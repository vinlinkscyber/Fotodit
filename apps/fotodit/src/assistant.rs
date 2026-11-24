//! Chat assistant bridge: composer prompts go to the SELECTED online AI provider
//! only. With no provider connected the chat says so and edits nothing: there is no
//! offline rule-based substitute, and manual editing is untouched. Plans are applied
//! against the app's admitted
//! command dispatch, per the `LiveEditSession` host contract: the app's real session is
//! borrowed for the duration of one operation, commands go through the admitted menu
//! path (authorization and discard interception preserved), and results are completed
//! values, never pending job receipts. The completed-value contract is enforced by
//! dispatching through `menus::invoke_completed`: a job-capable command (a filter, say)
//! runs inline even when the app has background jobs enabled, so the execution engine's
//! step sequencing, history binding and rollback view stay exact. The contract is gated
//! BEFORE any effect: a route that would open a dialog, workspace or picker, park an
//! unsaved-changes or rasterize prompt, start an interactive transform/session, run
//! recorded playback, or defer work to a background save/print worker is an explicit
//! pre-effect refusal (nothing opened, parked, started or queued), and any other route
//! outside the assistant's supported boundary is refused the same way rather than faked.
//! The supported boundary covers the command families plans emit (engine filters,
//! adjustment/fill layers, type, analysis, file.new with explicit parameters) plus
//! synchronous shell state (tools, panels, view, workspace, window).

use fotodit_automation::AutomationError;
use fotodit_automation::ai::{AiContext, AiEngine, EditingIntent, EditingPlan, GoalRequest, GoalStop, LiveEditSession, SemanticGoalRepair};
use fotodit_ui_egui::{AssistantBackend, FotoditApp};

/// Borrowed desktop editing destination: the app's live session through its admitted
/// command path, never a second raw session (see `LiveEditSession`'s contract).
struct GuiSession<'a> {
    app: &'a mut FotoditApp,
    ctx: &'a egui::Context,
}

impl LiveEditSession for GuiSession<'_> {
    fn inspect_active(&self) -> Result<Option<serde_json::Value>, AutomationError> {
        Ok(self.app.session.active().map(fotodit_engine::inspect::document))
    }

    fn command_run(&mut self, command: &str, params: serde_json::Value) -> Result<serde_json::Value, AutomationError> {
        fotodit_ui_egui::menus::invoke_completed(self.app, self.ctx, command, params).map_err(AutomationError::App)
    }

    fn visual_digest(&mut self, max_pixels: u64) -> Option<Result<serde_json::Value, String>> {
        // The same admitted menu path as every editing command: authorization
        // and discard interception are preserved exactly, and a refusal is
        // reported as capture failure - never bypassed.
        Some(fotodit_ui_egui::menus::invoke_completed(self.app, self.ctx, "image.analysis.pixelDigest", serde_json::json!({"maxPixels": max_pixels})))
    }
}

/// Panel backend holding the provider wiring and any in-flight provider goal.
pub struct GuiAssistant {
    planner_factory: Box<dyn Fn(&str) -> Result<(SelectedPlanner, [u8; 32]), String>>,
    identity: fn(&str) -> Result<[u8; 32], String>,
    pending: Option<PendingGoal>,
    /// Cross-turn context for this assistant session: completed prompts and
    /// plans, replayed into intent parsing and planning, and what bare
    /// "do that again" requests resolve against. Session-only, never persisted.
    context: AiContext,
    /// Document the context belongs to. A switch or close resets the context:
    /// recorded turns reference another document's layer ids and must never
    /// target the newly active one. Creation from no document keeps it.
    context_document: Option<String>,
}

impl GuiAssistant {
    /// Offline at startup. Only the in-app, session-only consent can enable a provider.
    pub fn new() -> Self {
        Self { planner_factory: Box::new(provider_planner), identity: provider_identity, pending: None, context: AiContext::new(), context_document: None }
    }

}

#[derive(Clone, Copy)]
enum PendingPhase {
    Planning,
    /// One result-check request is in flight; `check` is its 1-based number.
    Verifying { check: u32 },
}

enum WorkerResult {
    Plan(EditingPlan),
    Verdict(Box<fotodit_automation::ai::VisualVerdict>),
}

/// One provider goal in flight: a planning request, then optionally bounded
/// image result-checks while the consented budget lasts. Every apply re-runs
/// the full staleness and consent checks against the refreshed snapshot.
struct PendingGoal {
    phase: PendingPhase,
    receiver: std::sync::mpsc::Receiver<Result<WorkerResult, String>>,
    prompt: String,
    intent: EditingIntent,
    applied: Vec<String>,
    document: Option<serde_json::Value>,
    document_id: Option<String>,
    binding: (Option<&'static str>, Option<[u8; 32]>, u64),
    visual: bool,
}

/// Downscaled straight-alpha PNG render of the document composite for the
/// consented image path. Rendered locally on the UI thread; the bytes leave
/// the device only inside a request the image consent covers.
fn render_document_png(doc: &fotodit_doc::Document) -> Result<Vec<u8>, String> {
    const MAX_EDGE: u32 = 1024;
    let bounds = doc.bounds();
    let (w, h) = (bounds.width(), bounds.height());
    if w == 0 || h == 0 { return Err("The canvas is empty; no image was rendered.".into()); }
    let scale = (MAX_EDGE as f64 / f64::from(w.max(h))).min(1.0);
    let ow = ((w as f64 * scale).round() as u32).max(1);
    let oh = ((h as f64 * scale).round() as u32).max(1);
    let buf = fotodit_compose::render_reduced_rect(doc, bounds, ow, oh);
    let flat: Vec<f32> = buf.px.iter().flat_map(|p| *p).collect();
    let image = fotodit_codecs::Image::from_normalized(ow, oh, fotodit_codecs::ChannelLayout::Rgba,
        fotodit_codecs::SampleType::U8, &flat).map_err(|e| e.to_string())?;
    let bytes = fotodit_codecs::encode(&image, fotodit_codecs::Format::Png,
        &fotodit_codecs::EncodeOptions::default()).map_err(|e| e.to_string())?;
    if bytes.len() > fotodit_automation::budgets::MAX_PNG_BYTES {
        return Err("The rendered image exceeds the provider image budget; no image was sent.".into());
    }
    Ok(bytes)
}

fn document_id(app: &FotoditApp) -> Option<String> {
    app.session.active().map(|d| format!("{:?}", d.doc.id))
}

/// Name of the layer a completed plan last targeted, for cross-turn target
/// continuity. Ids are per-document state, so only the name crosses turns;
/// the provider always re-resolves targets against the current scene.
fn plan_target_name(plan: &EditingPlan, document: &Option<serde_json::Value>) -> Option<String> {
    let id = plan.steps.iter().rev().find_map(|s| s.params.get("layer").and_then(|v| v.as_u64()))?;
    document.as_ref()?.get("layers")?.as_array()?.iter()
        .find(|l| l.get("id").and_then(|v| v.as_u64()) == Some(id))?
        .get("name")?.as_str().map(|s| s.chars().take(64).collect())
}

struct FixedPlan(EditingPlan);
impl fotodit_automation::ai::AiPlanner for FixedPlan {
    fn plan(&self, _intent: &fotodit_automation::ai::EditingIntent, _scene: &fotodit_automation::ai::VisualScene,
        _context: &AiContext) -> Result<EditingPlan, fotodit_automation::ai::AiError> { Ok(self.0.clone()) }
}

impl AssistantBackend for GuiAssistant {
    fn reply(&mut self, app: &mut FotoditApp, ctx: &egui::Context, prompt: &str) -> Result<String, String> {
        if self.pending.is_some() { return Err("A provider request is already pending. Wait or cancel it first.".into()); }
        let binding = app.ai_providers.editing_consent().binding();
        if let Some(slot) = binding.0 {
            // Cross-turn context is per-document: switching to a different
            // document or closing it drops the recorded turns, so a follow-up
            // can never target another document's layers through stale ids.
            // Appearing from none (this assistant just created it) keeps it.
            let current_document = document_id(app);
            if match (&self.context_document, &current_document) {
                (Some(previous), Some(current)) => previous != current,
                (Some(_), None) => true,
                (None, _) => false,
            } { self.context = AiContext::new(); }
            self.context_document = current_document;
            // Each Send is one bounded request; session context resolves bare
            // replays locally instead of guessing what an earlier request meant.
            let lower = prompt.to_lowercase();
            // Bare conversational replays resolve against this session's own
            // context: the last completed prompt becomes the actual request.
            // Pronouns inside a full request (including the user's taxi-poster
            // wording) are ordinary language. With no completed turn yet there
            // is nothing to replay.
            let resolved;
            let prompt = if matches!(lower.trim().trim_end_matches(['.', '!', '?']), "do that again" | "same as previous" | "again" | "same" | "do it" | "do that") {
                match self.context.prompt_history.last() {
                    Some(previous) => { resolved = previous.clone(); resolved.as_str() }
                    None => return Err("There's nothing earlier in this session to repeat. State the full edit and name the target - nothing was sent.".into()),
                }
            } else { prompt };
            let document = app.session.active().map(fotodit_engine::inspect::document);
            let mut scene = document.as_ref().map(|doc| fotodit_automation::ai::PerceptionEngine::new().perceive(doc)).unwrap_or_default();
            if document.is_some() {
                // Local measurements ride with the scene: dominant colors from the
                // engine histogram, faces from the engine's own detector. Both are
                // bounded and budget-gated; an unavailable detector just leaves the
                // field empty - the contract tells the provider how to read that.
                let mut host = GuiSession { app: &mut *app, ctx };
                scene.dominant_colors = fotodit_automation::ai::computed_dominant_colors(&mut host);
                if u64::from(scene.width) * u64::from(scene.height) <= 4_000_000 {
                    if let Ok(result) = host.command_run("image.detectFaces", serde_json::json!({"minScore": 0.5, "sampleAllLayers": true})) {
                        if let Some(faces) = result.get("faces").and_then(serde_json::Value::as_array) {
                            scene.faces = faces.iter().filter_map(fotodit_automation::ai::FaceObservation::from_inspect).take(8).collect();
                        }
                    }
                }
            }
            let intent = fotodit_automation::ai::NaturalLanguageEngine::new().parse_intent(prompt, &scene, &self.context).map_err(|e| e.to_string())?;
            // With no document open there is no per-category gate: the provider
            // contract itself admits only plans whose first step is file.new on
            // an empty scene, so a non-creation plan is refused at validation
            // with nothing applied, whatever category the intent parsed to.
            let current_identity = match (self.identity)(slot) {
                Ok(identity) => identity,
                Err(error) => { app.ai_providers.editing_consent_mut().disable(); return Err(error); }
            };
            if binding.1 != Some(current_identity) {
                app.ai_providers.editing_consent_mut().disable();
                return Err("Provider credential or source changed. Enable editing again before sending.".into());
            }
            // Resolve only for this request; the planner and credential move to the worker.
            let (planner, loaded_identity) = match (self.planner_factory)(slot) {
                Ok(planner) => planner,
                Err(error) => { app.ai_providers.editing_consent_mut().disable(); return Err(error); }
            };
            // Detect replacement during resolution too, before dispatching anything.
            if loaded_identity != current_identity { app.ai_providers.editing_consent_mut().disable(); return Err("Provider credential changed. Enable editing again.".into()); }
            // Image-aware mode rides only with an open document and active Gemini
            // image consent; without it the metadata-only promise is unchanged.
            let visual = app.ai_providers.editing_consent().visual_active() && document.is_some();
            let image = if visual {
                match app.session.active().map(|d| render_document_png(&d.doc)) {
                    Some(Ok(png)) => Some(png),
                    Some(Err(error)) => return Err(error),
                    None => None,
                }
            } else { None };
            let (sender, receiver) = std::sync::mpsc::channel();
            let repaint = ctx.clone();
            let worker_intent = intent.clone();
            // The worker gets a snapshot of the session context as of dispatch;
            // turns completing while it runs record into the panel's own copy.
            let worker_context = self.context.clone();
            std::thread::spawn(move || {
                let result = planner.plan_visual(&worker_intent, &scene, &worker_context, image.as_deref())
                    .map(WorkerResult::Plan).map_err(|e| e.to_string());
                let _ = sender.send(result);
                repaint.request_repaint();
            });
            self.pending = Some(PendingGoal { phase: PendingPhase::Planning, receiver, prompt: prompt.into(),
                intent, applied: Vec::new(), document, document_id: document_id(app), binding, visual });
            Ok(if visual {
                format!("Asking {} to plan the edit, with the document image attached. No edits applied yet.", provider_name(slot))
            } else {
                format!("Asking {} to plan the edit. No edits applied yet.", provider_name(slot))
            })
        } else {
            // No offline rule-based substitute: AI natural editing is online-only.
            // Manual editing is untouched; say so plainly and edit nothing.
            Ok("AI editing isn't connected. It works online with an AI provider: open Edit > AI Providers to add a key and turn on AI editing, then ask again. Manual editing with the tools works offline as usual.".into())
        }
    }
    fn busy(&self) -> bool { self.pending.is_some() }
    fn cancel(&mut self) { self.pending = None; }
    fn poll(&mut self, app: &mut FotoditApp, ctx: &egui::Context) -> Option<String> {
        let pending = self.pending.as_ref()?;
        let slot = pending.binding.0?;
        if app.ai_providers.editing_consent().binding() != pending.binding
            || document_id(app) != pending.document_id {
            let verifying = matches!(pending.phase, PendingPhase::Verifying { .. });
            let visual = pending.visual;
            self.pending = None;
            // The already-dispatched worker cannot be cancelled: its request may still
            // finish (carrying the captured document image in visual mode), but its
            // result is never applied and no further requests start from this goal.
            let in_flight = match (verifying, visual) {
                (true, true) => " The review request already sent carries a captured image of the document and may still finish; no further requests will be made for this goal.",
                (true, false) => " The review request already sent may still finish; no further requests will be made for this goal.",
                (false, true) => " The planning request already sent carries a captured image of the document and may still finish; nothing from it will be applied and no further requests will be made for this goal.",
                (false, false) => " The planning request already sent may still finish; nothing from it will be applied and no further requests will be made for this goal.",
            };
            return Some(if verifying {
                format!("Edit cancelled because the document, key or AI permission changed. The changes already made stay in the document - use Undo if needed.{in_flight}")
            } else {
                format!("Edit cancelled because the document, key or AI permission changed. Nothing from it will be applied.{in_flight}")
            });
        }
        let result = match pending.receiver.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => return None,
            Err(_) => Err("Provider worker stopped. No further request was made.".into()),
        };
        let phase = pending.phase;
        let mut pending = self.pending.take()?;
        // Unseal/re-inspect once only, after a successful response and just before application.
        if (self.identity)(slot).ok() != pending.binding.1
            || app.session.active().map(fotodit_engine::inspect::document) != pending.document {
            app.ai_providers.editing_consent_mut().disable();
            return Some(if matches!(phase, PendingPhase::Verifying { .. }) {
                "Edit cancelled because the document or key changed. The changes already made stay in the document - use Undo if needed.".into()
            } else {
                "Edit cancelled because the document or key changed. Nothing from it will be applied.".into()
            });
        }
        let plan = match (phase, result) {
            (PendingPhase::Planning, Err(error)) => return Some(error),
            (PendingPhase::Verifying { .. }, Err(error)) => return Some(format!("The result review failed: {error} Changes made so far stay in the document - use Undo if needed.")),
            (PendingPhase::Planning, Ok(WorkerResult::Plan(plan))) => plan,
            (PendingPhase::Verifying { check: _ }, Ok(WorkerResult::Verdict(verdict))) => {
                let verdict = *verdict;
                if verdict.satisfied {
                    return Some(format!("Gemini reviewed the result and says it matches your request."));
                }
                if verdict.corrective.steps.is_empty() {
                    let note = if verdict.critique.is_empty() { String::new() } else { format!(": {}", verdict.critique) };
                    return Some(format!("Gemini reviewed the result and doesn't think it matches{note}. The changes made so far stay in the document - use Undo if needed."));
                }
                verdict.corrective
            }
            _ => return Some("The provider's reply didn't fit what I asked. No edits were applied.".into()),
        };
        // UI-thread application only, after the immutable snapshot and consent still match.
        // max_attempts 2 lets the LOCAL repair adapter (no extra provider request)
        // correct a completed-but-unmet attempt once; the goal coordinator's stop
        // reason, never raw execution success, decides what is reported (D4).
        let outcome = {
            let engine = AiEngine::with_planner(Box::new(FixedPlan(plan)));
            let mut session = GuiSession { app, ctx };
            let request = GoalRequest { max_attempts: 2, ..GoalRequest::new(&pending.prompt) };
            // The session context persists across turns: the goal session
            // records each completed prompt+plan into it, which is what later
            // bare replays and pronoun references resolve against.
            engine.process_goal_in(&mut session, &request, &mut self.context, &SemanticGoalRepair::new())
        };
        // Record every attempt that actually executed, including a repair attempt,
        // so the result review lists what was truly applied.
        for attempt in &outcome.attempts {
            if attempt.execution.as_ref().is_some_and(|e| e.is_success) {
                pending.applied.extend(attempt.plan.steps.iter().map(|s| {
                    let text = format!("{} {}", s.command, s.params);
                    text.chars().take(200).collect()
                }));
            }
        }
        let executed = outcome.attempts.iter().any(|a| a.execution.as_ref().is_some_and(|e| e.is_success));
        let completed = outcome.stop == GoalStop::Completed;
        if executed {
            // The apply changed the document: re-anchor the staleness snapshot so
            // the next round compares against the post-edit state, not the original.
            pending.document = app.session.active().map(fotodit_engine::inspect::document);
            pending.document_id = document_id(app);
            // Cross-turn target continuity: remember the last targeted layer's
            // name so the next turn's parsing can resolve "it"/"the heading".
            if let Some(attempt) = outcome.attempts.iter().rev().find(|a| a.execution.as_ref().is_some_and(|e| e.is_success)) {
                self.context.active_target = plan_target_name(&attempt.plan, &pending.document);
            }
            self.context_document = pending.document_id.clone();
        }
        let next_check = match phase { PendingPhase::Planning => 1, PendingPhase::Verifying { check } => check + 1 };
        // Requests actually sent for this goal so far; result checks only count once sent.
        let requests_made = match phase { PendingPhase::Planning => 1, PendingPhase::Verifying { check } => 1 + check };
        let can_check = pending.visual && executed
            && next_check <= fotodit_ui_egui::ai_providers_ui::VISUAL_MAX_CHECKS
            && app.ai_providers.editing_consent().visual_active();
        if can_check {
            let rendered = app.session.active().map(|d| render_document_png(&d.doc));
            match rendered {
                Some(Ok(image)) => {
                    let scene = pending.document.as_ref().map(|doc| fotodit_automation::ai::PerceptionEngine::new().perceive(doc)).unwrap_or_default();
                    let (planner, loaded_identity) = match (self.planner_factory)(slot) {
                        Ok(pair) => pair,
                        Err(error) => { app.ai_providers.editing_consent_mut().disable(); return Some(error); }
                    };
                    if Some(loaded_identity) != pending.binding.1 {
                        app.ai_providers.editing_consent_mut().disable();
                        return Some("The provider key changed between result reviews. Enable editing again; the changes made so far remain.".into());
                    }
                    let (sender, receiver) = std::sync::mpsc::channel();
                    let repaint = ctx.clone();
                    let prompt_text = pending.prompt.clone();
                    let applied = pending.applied.clone();
                    let intent = pending.intent.clone();
                    std::thread::spawn(move || {
                        let result = planner.verify_visual(&prompt_text, &applied, &intent, &scene, &image)
                            .map(|v| WorkerResult::Verdict(Box::new(v))).map_err(|e| e.to_string());
                        let _ = sender.send(result);
                        repaint.request_repaint();
                    });
                    pending.phase = PendingPhase::Verifying { check: next_check };
                    pending.receiver = receiver;
                    self.pending = Some(pending);
                    log::info!("result review {} of at most {} dispatched", next_check + 1, 1 + fotodit_ui_egui::ai_providers_ui::VISUAL_MAX_CHECKS);
                    return Some(format!("Done via {}: edits applied. Asking Gemini to review the result.", provider_name(slot)));
                }
                Some(Err(render_error)) => {
                    log::info!("{requests_made} provider request(s) were made for this goal");
                    return Some(format!("Done via {}: edits applied, but I couldn't render the result image for a review: {render_error}", provider_name(slot)));
                }
                None => {
                    return Some(format!("Done via {}: edits applied. The document is no longer open, so no result review was sent.", provider_name(slot)));
                }
            }
        }
        Some(if completed {
            if pending.visual && next_check > fotodit_ui_egui::ai_providers_ui::VISUAL_MAX_CHECKS {
                log::info!("result-review budget used: {requests_made} provider request(s) were made for this goal");
                format!("Done via {}: edits applied. I've used the result reviews available for this goal, so I'll stop here.", provider_name(slot))
            } else if pending.visual {
                log::info!("{requests_made} provider request(s) were made for this goal; image consent is no longer active");
                format!("Done via {}: edits applied. Image permission was turned off, so no result review was sent.", provider_name(slot))
            } else {
                format!("Done via {}: the edit steps are complete.", provider_name(slot))
            }
        } else if executed {
            // Steps ran but the goal coordinator did not verify the goal: say so
            // instead of calling the edit complete (D4).
            format!("Edits applied via {}, but the result isn't verified against your request: {} The changes made stay in the document - use Undo if needed.", provider_name(slot), outcome.detail)
        } else if outcome.attempts.iter().any(|a| !a.commands.is_empty()) {
            format!("Stopped partway via {} - some changes may be left in the document. Check the canvas and use Undo if needed. {}", provider_name(slot), outcome.detail)
        } else { format!("No edits were applied via {}. {}", provider_name(slot), outcome.detail) })
    }
}

struct SelectedPlanner {
    provider: fotodit_automation::ai::ProviderId,
    bridge: fotodit_automation::ai::AiProviderBridge,
}

impl fotodit_automation::ai::AiPlanner for SelectedPlanner {
    fn plan(&self, intent: &fotodit_automation::ai::EditingIntent, scene: &fotodit_automation::ai::VisualScene,
        context: &AiContext) -> Result<EditingPlan, fotodit_automation::ai::AiError> {
        self.bridge.plan_selected(self.provider, intent, scene, context)
    }
}

impl SelectedPlanner {
    fn plan_visual(&self, intent: &EditingIntent, scene: &fotodit_automation::ai::VisualScene,
        context: &AiContext, image: Option<&[u8]>) -> Result<EditingPlan, fotodit_automation::ai::AiError> {
        self.bridge.plan_visual(self.provider, intent, scene, context, image)
    }
    fn verify_visual(&self, prompt: &str, applied: &[String], intent: &EditingIntent,
        scene: &fotodit_automation::ai::VisualScene, image: &[u8])
        -> Result<fotodit_automation::ai::VisualVerdict, fotodit_automation::ai::AiError> {
        self.bridge.verify_visual(self.provider, prompt, applied, intent, scene, image)
    }
}

fn provider_planner(slot: &str) -> Result<(SelectedPlanner, [u8; 32]), String> {
    use fotodit_automation::ai::{AiProviderBridge, CommandCatalog, ProviderConfig};
    let (provider, config) = ProviderConfig::for_editing_slot(slot).map_err(|e| e.to_string())?;
    let identity = config_identity(slot, provider, &config)?;
    Ok((SelectedPlanner {
        provider, bridge: AiProviderBridge::new(config, CommandCatalog::new()),
    }, identity))
}

/// Binds consent to selected credential bytes, source and fixed destination without retaining a key.
pub fn provider_identity(slot: &str) -> Result<[u8; 32], String> {
    use fotodit_automation::ai::ProviderConfig;
    let (provider, config) = ProviderConfig::for_editing_slot(slot).map_err(|e| e.to_string())?;
    config_identity(slot, provider, &config)
}

fn config_identity(slot: &str, provider: fotodit_automation::ai::ProviderId,
    config: &fotodit_automation::ai::ProviderConfig) -> Result<[u8; 32], String> {
    use fotodit_automation::ai::ProviderId;
    let key = match provider {
        ProviderId::Gemini => config.gemini_api_key.as_deref(),
        ProviderId::Cohere => config.cohere_api_key.as_deref(),
        ProviderId::Lajiwa => config.lajiwa_api_key.as_deref(),
        _ => None,
    }.ok_or_else(|| "Selected key unavailable".to_string())?;
    let source = fotodit_keystore_source(slot)?;
    let mut hash = blake3::Hasher::new();
    hash.update(slot.as_bytes()); hash.update(source.as_bytes()); hash.update(key.as_bytes());
    Ok(*hash.finalize().as_bytes())
}

fn fotodit_keystore_source(slot: &str) -> Result<String, String> {
    // The backend's resolver already enforces the chosen source. The protected file also
    // carries source changes; stamp all of it so unrelated credential changes revoke consent too.
    let base = std::env::var_os("APPDATA").ok_or_else(|| "No protected provider store".to_string())?;
    let bytes = std::fs::read(std::path::PathBuf::from(base).join("Fotodit").join("ai_providers.v1.json"))
        .map_err(|_| "Provider store is missing or unreadable".to_string())?;
    Ok(format!("{slot}:{}", blake3::hash(&bytes).to_hex()))
}

/// The provider's display name in chat messages.
fn provider_name(slot: &str) -> &str {
    match slot {
        "gemini" => "Gemini",
        "cohere" => "Cohere",
        "lajiwa" => "Lajiwa",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fotodit_ui_egui::Services;

    struct ChatTransport {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail: bool,
    }
    impl fotodit_automation::ai::HttpTransport for ChatTransport {
        fn execute_json(&self, method: &str, url: &str, headers: &[(&str, &str)], body: Option<&serde_json::Value>,
            _timeout: std::time::Duration) -> Result<(u16, serde_json::Value), fotodit_automation::ai::AiError> {
            use std::sync::atomic::Ordering;
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(method, "POST");
            assert!(url.starts_with("https://generativelanguage.googleapis.com/"));
            assert_eq!(headers, [("x-goog-api-key", "fixture-not-a-real-key")]);
            assert!(body.unwrap().to_string().contains("chndrax"), "original chat prompt reaches provider");
            if self.fail { return Ok((429, serde_json::json!({"error":{"message":"fixture limit"}}))); }
            let plan = serde_json::json!({"title":"Provider fixture", "rationale":"mocked provider response",
                "steps":[{"command":"file.new","params":{"width":240,"height":320,"background":"transparent"}}]});
            Ok((200, serde_json::json!({"candidates":[{"content":{"parts":[{"text":plan.to_string()}]}}]})))
        }
        fn fetch_raw_bytes(&self, _url: &str, _headers: &[(&str, &str)], _timeout: std::time::Duration)
            -> Result<(u16, Vec<u8>), fotodit_automation::ai::AiError> {
            panic!("chat editing must not fetch images")
        }
    }

    fn chat_fixture(fail: bool) -> (FotoditApp, std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use fotodit_automation::ai::{AiProviderBridge, ProviderConfig, ProviderId, CommandCatalog};
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let calls = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let transport_calls = calls.clone();
        let config_loads = loads.clone();
        let mut assistant = GuiAssistant::new();
        assistant.identity = |_| Ok([7; 32]);
        assistant.planner_factory = Box::new(move |slot| {
            assert_eq!(slot, "gemini");
            config_loads.fetch_add(1, Ordering::SeqCst);
            let config = ProviderConfig { gemini_api_key: Some("fixture-not-a-real-key".into()), ..Default::default() };
            let bridge = AiProviderBridge::with_transport(config, CommandCatalog::new(),
                Box::new(ChatTransport { calls: transport_calls.clone(), fail }));
            Ok((SelectedPlanner { provider: ProviderId::Gemini, bridge }, [7; 32]))
        });
        let mut services = Services::default();
        services.assistant_backend = Some(Box::new(assistant));
        (FotoditApp::new(fotodit_engine::Session::new(), services), calls, loads)
    }

    fn poll_plan(app: &mut FotoditApp, ctx: &egui::Context) -> Option<String> {
        let mut backend = app.services.assistant_backend.take().unwrap();
        let result = backend.poll(app, ctx);
        app.services.assistant_backend = Some(backend);
        result
    }
    fn await_plan(app: &mut FotoditApp, ctx: &egui::Context) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(answer) = poll_plan(app, ctx) { return answer; }
            assert!(std::time::Instant::now() < deadline, "mock planning did not finish");
            std::thread::yield_now();
        }
    }

    #[test]
    fn complete_first_turn_requests_with_pronouns_reach_provider() {
        use std::sync::atomic::Ordering;
        for prompt in ["create a poster for my taxi busines to promote it its called chndrax",
            "create a poster that says chndrax", "make a poster for chndrax and give it blue text"] {
            let (mut app, calls, _) = chat_fixture(false);
            let ctx = egui::Context::default();
            app.ai_providers.editing_consent_mut().request("gemini");
            app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
            let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, prompt);
            assert!(initial.starts_with("Asking Gemini"), "{prompt}: {initial}");
            let _ = await_plan(&mut app, &ctx);
            assert_eq!(calls.load(Ordering::SeqCst), 1, "{prompt}");
        }
    }

    #[test]
    fn bare_replay_without_history_refuses_before_network() {
        use std::sync::atomic::Ordering;
        for prompt in ["do that again", "same as previous"] {
            let (mut app, calls, loads) = chat_fixture(false);
            app.ai_providers.editing_consent_mut().request("gemini");
            app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
            let reply = fotodit_ui_egui::panels::assistant::submit(&mut app, &egui::Context::default(), prompt);
            assert!(reply.contains("nothing was sent"), "{reply}");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(loads.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn no_document_non_creation_request_reaches_provider_and_creates() {
        use std::sync::atomic::Ordering;
        // The per-category gate is gone: with no document open the provider
        // contract itself admits only file.new-first plans, so any request may
        // be sent and a creation-shaped plan applies.
        let (mut app, calls, _) = chat_fixture(false);
        let ctx = egui::Context::default();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "make it blue, chndrax");
        assert!(initial.starts_with("Asking Gemini"), "{initial}");
        let _ = await_plan(&mut app, &ctx);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let doc = fotodit_engine::inspect::document(app.session.active().unwrap());
        assert_eq!(doc["width"], 240);
    }

    #[test]
    fn bare_replay_resolves_against_completed_turn() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = chat_fixture(false);
        let ctx = egui::Context::default();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        assert!(initial.starts_with("Asking Gemini"), "{initial}");
        let _ = await_plan(&mut app, &ctx);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // A bare replay resolves to the last completed prompt and reaches the
        // provider again with that wording (the fixture asserts it verbatim).
        let replay = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "do that again");
        assert!(replay.starts_with("Asking Gemini"), "{replay}");
        let _ = await_plan(&mut app, &ctx);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn document_switch_resets_cross_turn_context() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = chat_fixture(false);
        let ctx = egui::Context::default();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        let _ = await_plan(&mut app, &ctx);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Switching to another document drops the recorded turns: a replay
        // must never target the new document through the old one's context.
        app.session.execute("file.new", serde_json::json!({"width":16,"height":16})).unwrap();
        let reply = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "do that again");
        assert!(reply.contains("nothing was sent"), "{reply}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pending_plan_discarded_on_cancel_document_or_consent_change() {
        for change in ["cancel", "document", "consent"] {
            let (mut app, _, _) = chat_fixture(false);
            let ctx = egui::Context::default();
            app.ai_providers.editing_consent_mut().request("gemini");
            app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
            let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
            match change {
                "cancel" => app.services.assistant_backend.as_mut().unwrap().cancel(),
                "document" => { app.session.execute("file.new", serde_json::json!({"width":32,"height":32})).unwrap(); },
                _ => { app.ai_providers.editing_consent_mut().disable(); },
            }
            let answer = poll_plan(&mut app, &ctx);
            if change == "cancel" { assert!(answer.is_none()); }
            else {
                let answer = answer.unwrap();
                assert!(answer.contains("cancelled"), "{answer}");
                assert!(answer.contains("may still finish"), "the in-flight send is disclosed: {answer}");
                assert!(!answer.contains("captured document image"), "metadata-only goal carries no image: {answer}");
            }
            if change == "document" { assert_eq!(app.session.active().unwrap().doc.size.width,32); }
            else { assert!(app.session.active().is_none()); }
        }
    }

    #[test]
    fn composer_provider_enable_is_local_then_send_reaches_selected_mock() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, loads) = chat_fixture(false);
        let ctx = egui::Context::default();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "enabling sends nothing");
        assert_eq!(loads.load(Ordering::SeqCst), 0, "fixture enable does not build a provider planner");
        let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        assert!(initial.starts_with("Asking Gemini"), "{initial}");
        let answer = await_plan(&mut app, &ctx);
        // The fixture plan creates an empty document: execution succeeds but the
        // design goal (no type layers at all) is not verified, and the local
        // repair adapter stops because overwriting user text is unsafe. The chat
        // must report the unverified truth, never "complete" (D4).
        assert!(answer.starts_with("Edits applied via Gemini"), "{answer}");
        assert!(answer.contains("isn't verified"), "{answer}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        let doc = fotodit_engine::inspect::document(app.session.active().unwrap());
        assert_eq!(doc["width"], 240, "the provider's planned dimensions, not a local template default");
        assert_eq!(doc["height"], 320);
        let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        let _ = await_plan(&mut app, &ctx);
        assert_eq!(loads.load(Ordering::SeqCst), 2, "every Send reloads selected credentials");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        app.ai_providers.editing_consent_mut().disable();
        let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "disable prevents remote requests");
    }

    /// The open-document side of the creation boundary: when the user already has a
    /// document open and the provider answers with a leading file.new, the plan must be
    /// refused and the open document left untouched - never a silent replacement. The
    /// contract validator (provider_bridge validate_selected_plan, unchanged since the
    /// frozen candidate) refuses file.new on a non-empty scene before any execution, so
    /// the refusal surfaces as the plan-validation error.
    #[test]
    fn provider_led_file_new_never_replaces_an_open_document() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = chat_fixture(false);
        let ctx = egui::Context::default();
        app.session.execute("file.new", serde_json::json!({"width":64,"height":64})).unwrap();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        let answer = await_plan(&mut app, &ctx);
        assert!(answer.contains("Provider plan is outside the supported command, parameter or target contract"),
            "a provider-led file.new is refused while a document is open: {answer}");
        assert!(answer.contains("nothing was edited"), "{answer}");
        assert_eq!(app.session.active().unwrap().doc.size.width, 64, "the user's open document is untouched");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn composer_provider_failure_never_runs_offline_or_other_account() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = chat_fixture(true);
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let ctx = egui::Context::default();
        let _ = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster called chndrax");
        let answer = await_plan(&mut app, &ctx);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "one selected-provider attempt, no retry cascade");
        assert!(app.session.active().is_none(), "no fallback edits after provider refusal");
        assert!(answer.contains("No offline or other-provider fallback"), "{answer}");
    }

    /// The user's rule: AI natural editing is online-only. With no provider connected
    /// the chat says so plainly, edits nothing and leaves manual editing alone - no
    /// offline rule-based substitute runs, for a creation request or an open document.
    /// The backend must be installed (as in the real build) with no provider consent
    /// active; the fixture's transport counter proves no provider attempt is made.
    #[test]
    fn unconnected_chat_says_ai_is_offline_and_edits_nothing() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = chat_fixture(false);
        let ctx = egui::Context::default();
        let reply = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "create a poster for my taxi business called chndrax");
        assert!(reply.contains("AI editing isn't connected"), "{reply}");
        assert!(reply.contains("Manual editing"), "{reply}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no provider attempt without an active provider");
        assert!(app.session.active().is_none(), "no offline substitute created a document");
        app.session.execute("file.new", serde_json::json!({"width": 32, "height": 32, "background": "white"})).unwrap();
        let before = fotodit_engine::inspect::document(app.session.active().unwrap())["history"].as_array().unwrap().len();
        let reply = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "Enhance image quality");
        assert!(reply.contains("AI editing isn't connected"), "{reply}");
        let after = fotodit_engine::inspect::document(app.session.active().unwrap())["history"].as_array().unwrap().len();
        assert_eq!(before, after, "no offline substitute edited the open document");
    }

    #[test]
    fn composer_reloads_config_each_send_and_revokes_on_missing_key() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let loads = Arc::new(AtomicUsize::new(0));
        let load_count = loads.clone();
        let mut assistant = GuiAssistant::new();
        assistant.identity = |_| Ok([7; 32]);
        assistant.planner_factory = Box::new(move |_| {
            load_count.fetch_add(1, Ordering::SeqCst);
            Err("Selected key removed".into())
        });
        let mut app = FotoditApp::new(fotodit_engine::Session::new(), Services::default());
        app.services.assistant_backend = Some(Box::new(assistant));
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let answer = fotodit_ui_egui::panels::assistant::submit(&mut app, &egui::Context::default(), "create a poster called chndrax");
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(answer, "Selected key removed");
        assert_eq!(app.ai_providers.editing_consent().active(), None);
        assert!(app.session.active().is_none());
    }

    /// Records every request body; answers review bodies from a verdict queue
    /// and planning bodies with one vibrance plan. Never touches the network.
    struct VisualTransport {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        bodies: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
        verdicts: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<serde_json::Value>>>,
    }
    impl fotodit_automation::ai::HttpTransport for VisualTransport {
        fn execute_json(&self, method: &str, url: &str, _headers: &[(&str, &str)], body: Option<&serde_json::Value>,
            _timeout: std::time::Duration) -> Result<(u16, serde_json::Value), fotodit_automation::ai::AiError> {
            use std::sync::atomic::Ordering;
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(method, "POST");
            assert!(url.starts_with("https://generativelanguage.googleapis.com/"));
            let body = body.expect("visual fixture requests carry a body").clone();
            let instruction = body.pointer("/contents/0/parts/0/text").and_then(serde_json::Value::as_str).unwrap_or("").to_string();
            self.bodies.lock().unwrap().push(body);
            let payload = if instruction.contains("reviewing the result of an edit session") {
                self.verdicts.lock().unwrap().pop_front()
                    .unwrap_or_else(|| serde_json::json!({"satisfied": true, "critique": "", "corrective_steps": []}))
            } else {
                serde_json::json!({"title":"Visual fixture","rationale":"mocked provider response",
                    "steps":[{"command":"layer.newAdjustmentLayer.vibrance","params":{"vibrance":10,"saturation":5}}]})
            };
            Ok((200, serde_json::json!({"candidates":[{"content":{"parts":[{"text":payload.to_string()}]}}]})))
        }
        fn fetch_raw_bytes(&self, _url: &str, _headers: &[(&str, &str)], _timeout: std::time::Duration)
            -> Result<(u16, Vec<u8>), fotodit_automation::ai::AiError> {
            panic!("chat editing must not fetch images")
        }
    }

    fn visual_fixture(verdicts: Vec<serde_json::Value>)
        -> (FotoditApp, std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>) {
        use fotodit_automation::ai::{AiProviderBridge, CommandCatalog, ProviderConfig, ProviderId};
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let verdicts = std::sync::Arc::new(std::sync::Mutex::new(verdicts.into_iter().collect::<std::collections::VecDeque<_>>()));
        let (factory_calls, factory_bodies, factory_verdicts) = (calls.clone(), bodies.clone(), verdicts.clone());
        let mut assistant = GuiAssistant::new();
        assistant.identity = |_| Ok([7; 32]);
        assistant.planner_factory = Box::new(move |slot| {
            assert_eq!(slot, "gemini");
            let config = ProviderConfig { gemini_api_key: Some("fixture-not-a-real-key".into()), ..Default::default() };
            let bridge = AiProviderBridge::with_transport(config, CommandCatalog::new(), Box::new(VisualTransport {
                calls: factory_calls.clone(), bodies: factory_bodies.clone(), verdicts: factory_verdicts.clone(),
            }));
            Ok((SelectedPlanner { provider: ProviderId::Gemini, bridge }, [7; 32]))
        });
        let mut services = Services::default();
        services.assistant_backend = Some(Box::new(assistant));
        (FotoditApp::new(fotodit_engine::Session::new(), services), calls, bodies)
    }

    /// Records every request body; answers planning bodies with one fixed
    /// plan and review bodies from a verdict queue. Never touches the network.
    struct SymbolicPlanTransport {
        bodies: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
        plan: serde_json::Value,
        verdicts: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<serde_json::Value>>>,
    }
    impl fotodit_automation::ai::HttpTransport for SymbolicPlanTransport {
        fn execute_json(&self, method: &str, url: &str, _headers: &[(&str, &str)], body: Option<&serde_json::Value>,
            _timeout: std::time::Duration) -> Result<(u16, serde_json::Value), fotodit_automation::ai::AiError> {
            assert_eq!(method, "POST");
            assert!(url.starts_with("https://generativelanguage.googleapis.com/"));
            let body = body.expect("fixture requests carry a body").clone();
            let instruction = body.pointer("/contents/0/parts/0/text").and_then(serde_json::Value::as_str).unwrap_or("").to_string();
            self.bodies.lock().unwrap().push(body);
            let payload = if instruction.contains("reviewing the result of an edit session") {
                self.verdicts.lock().unwrap().pop_front()
                    .unwrap_or_else(|| serde_json::json!({"satisfied": true, "critique": "", "corrective_steps": []}))
            } else {
                self.plan.clone()
            };
            Ok((200, serde_json::json!({"candidates":[{"content":{"parts":[{"text":payload.to_string()}]}}]})))
        }
        fn fetch_raw_bytes(&self, _url: &str, _headers: &[(&str, &str)], _timeout: std::time::Duration)
            -> Result<(u16, Vec<u8>), fotodit_automation::ai::AiError> {
            panic!("chat editing must not fetch images")
        }
    }

    fn symbolic_fixture(plan: serde_json::Value, verdicts: Vec<serde_json::Value>)
        -> (FotoditApp, std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>) {
        use fotodit_automation::ai::{AiProviderBridge, CommandCatalog, ProviderConfig, ProviderId};
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let verdicts = std::sync::Arc::new(std::sync::Mutex::new(verdicts.into_iter().collect::<std::collections::VecDeque<_>>()));
        let (factory_bodies, factory_verdicts) = (bodies.clone(), verdicts.clone());
        let mut assistant = GuiAssistant::new();
        assistant.identity = |_| Ok([7; 32]);
        assistant.planner_factory = Box::new(move |slot| {
            assert_eq!(slot, "gemini");
            let config = ProviderConfig { gemini_api_key: Some("fixture-not-a-real-key".into()), ..Default::default() };
            let bridge = AiProviderBridge::with_transport(config, CommandCatalog::new(), Box::new(SymbolicPlanTransport {
                bodies: factory_bodies.clone(), plan: plan.clone(), verdicts: factory_verdicts.clone(),
            }));
            Ok((SelectedPlanner { provider: ProviderId::Gemini, bridge }, [7; 32]))
        });
        let mut services = Services::default();
        services.assistant_backend = Some(Box::new(assistant));
        (FotoditApp::new(fotodit_engine::Session::new(), services), bodies)
    }

    #[test]
    fn visual_loop_applies_intra_plan_references_and_reviews_the_refreshed_scene() {
        let plan = serde_json::json!({"title":"t","rationale":"r","steps":[
            {"command":"type.create","params":{"x":4,"y":16,"text":"Hi","size":14}},
            {"command":"type.setStyle","params":{"layer":{"$step":1},"size":30}},
            {"command":"edit.transform","params":{"layer":{"$step":1},"dx":3,"dy":2}}
        ]});
        let (mut app, bodies) = symbolic_fixture(plan, vec![serde_json::json!({"satisfied": true, "critique": "", "corrective_steps": []})]);
        let ctx = egui::Context::default();
        app.session.execute("file.new", serde_json::json!({"width":64,"height":64})).unwrap();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        app.ai_providers.editing_consent_mut().set_visual(true);
        let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "add the title Hi and make it bigger");
        assert!(initial.contains("with the document image attached"), "{initial}");
        let first = await_plan(&mut app, &ctx);
        assert!(first.contains("Asking Gemini to review the result"), "{first}");
        let second = await_plan(&mut app, &ctx);
        assert!(second.contains("says it matches your request"), "{second}");
        // The plan's create-then-style-then-move chain ran against the real id.
        let doc = fotodit_engine::inspect::document(app.session.active().unwrap());
        let layers = doc["layers"].as_array().expect("inspection lists layers");
        assert!(layers.iter().any(|l| l["kind"] == "Type"), "the created type layer exists after styling and transform");
        // The result check went out with the POST-APPLY scene: the provider saw
        // the created layer with its real id, so a corrective plan could target it.
        let review = bodies.lock().unwrap().iter()
            .filter_map(|b| b.pointer("/contents/0/parts/0/text").and_then(serde_json::Value::as_str).map(str::to_string))
            .find(|t| t.contains("reviewing the result of an edit session"))
            .expect("a result check was sent");
        let review: serde_json::Value = serde_json::from_str(&review).unwrap();
        let review_layers = review["scene"]["layers"].as_array().expect("review scene lists layers");
        assert!(review_layers.iter().any(|l| l["kind"] == "Type"),
            "the result-check scene shows the layer the plan created: {review_layers:?}");
    }

    fn submit_visual(app: &mut FotoditApp, ctx: &egui::Context) {
        app.session.execute("file.new", serde_json::json!({"width":32,"height":32})).unwrap();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        app.ai_providers.editing_consent_mut().set_visual(true);
        let initial = fotodit_ui_egui::panels::assistant::submit(app, ctx, "boost the colors");
        assert!(initial.contains("with the document image attached"), "{initial}");
    }

    #[test]
    fn visual_loop_plan_then_satisfied_check_costs_two_requests() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, bodies) = visual_fixture(vec![serde_json::json!({"satisfied": true, "critique": "looks good", "corrective_steps": []})]);
        let ctx = egui::Context::default();
        submit_visual(&mut app, &ctx);
        let first = await_plan(&mut app, &ctx);
        assert!(first.contains("Asking Gemini to review the result"), "{first}");
        let second = await_plan(&mut app, &ctx);
        assert!(second.contains("says it matches your request"), "{second}");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        let parts = bodies[0].pointer("/contents/0/parts").and_then(serde_json::Value::as_array).unwrap();
        assert_eq!(parts[1]["inline_data"]["mime_type"], "image/png", "planning request carries the canvas PNG");
        assert!(!parts[1]["inline_data"]["data"].as_str().unwrap().is_empty());
        assert!(bodies[1].to_string().contains("reviewing the result of an edit session"));
        assert!(bodies[1].to_string().contains("newAdjustmentLayer.vibrance"), "applied steps reach the result check");
    }

    #[test]
    fn visual_loop_budget_exhaustion_stops_at_three_requests() {
        use std::sync::atomic::Ordering;
        let corrective = serde_json::json!({"satisfied": false, "critique": "still flat", "corrective_steps": [
            {"command":"layer.newAdjustmentLayer.vibrance","params":{"vibrance":12,"saturation":6},"description":"more color"}]});
        let (mut app, calls, _) = visual_fixture(vec![corrective.clone(), corrective]);
        let ctx = egui::Context::default();
        submit_visual(&mut app, &ctx);
        let _ = await_plan(&mut app, &ctx); // plan applied, result check 1 dispatched
        let _ = await_plan(&mut app, &ctx); // verdict 1: corrective applied, result check 2 dispatched
        let final_msg = await_plan(&mut app, &ctx); // verdict 2: corrective applied, budget spent
        assert!(final_msg.contains("used the result reviews"), "{final_msg}");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(poll_plan(&mut app, &ctx).is_none(), "no fourth provider request");
    }

    #[test]
    fn visual_consent_off_sends_one_metadata_only_request() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, bodies) = visual_fixture(vec![]);
        let ctx = egui::Context::default();
        app.session.execute("file.new", serde_json::json!({"width":32,"height":32})).unwrap();
        app.ai_providers.editing_consent_mut().request("gemini");
        app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
        let initial = fotodit_ui_egui::panels::assistant::submit(&mut app, &ctx, "boost the colors");
        assert!(initial.starts_with("Asking Gemini"), "{initial}");
        assert!(!initial.contains("image attached"), "{initial}");
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("the edit steps are complete"), "{msg}");
        assert!(!msg.contains("verify the visual result"), "{msg} - no blanket verification disclaimer in chat");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].to_string().contains("inline_data"), "no image part without image consent");
    }

    #[test]
    fn visual_consent_withdrawn_before_apply_reports_actual_request_count() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, bodies) = visual_fixture(vec![]);
        let ctx = egui::Context::default();
        submit_visual(&mut app, &ctx);
        app.ai_providers.editing_consent_mut().set_visual(false);
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("Image permission was turned off"), "{msg}");
        assert!(!msg.contains("provider request(s)"), "{msg} - request counts move to the log");
        assert!(!msg.contains("used up"), "{msg}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].to_string().contains("inline_data"), "the plan request did carry the consented image");
    }

    #[test]
    fn stale_result_check_discard_keeps_applied_edits_wording() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = visual_fixture(vec![serde_json::json!({"satisfied": true, "critique": "", "corrective_steps": []})]);
        let ctx = egui::Context::default();
        submit_visual(&mut app, &ctx);
        let first = await_plan(&mut app, &ctx);
        assert!(first.contains("Asking Gemini to review the result"), "{first}");
        // Replace the document while the result check is in flight.
        app.session.execute("file.new", serde_json::json!({"width":16,"height":16})).unwrap();
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("Edit cancelled"), "{msg}");
        assert!(msg.contains("changes already made stay"), "{msg}");
        assert!(!msg.contains("Nothing from it will be applied"), "{msg} - the applied-edits branch, not the nothing-applied one");
        assert!(msg.contains("may still finish"), "the in-flight send is disclosed: {msg}");
        assert!(msg.contains("captured image of the document"), "the image the in-flight check carries is disclosed: {msg}");
        assert!(msg.contains("no further requests will be made"), "{msg}");
        // The result-check request was dispatched (worker thread spawned) before the
        // discard was observed; the send itself may still be in flight. Wait for it
        // before counting, so the assertion does not race thread scheduling.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while calls.load(Ordering::SeqCst) < 2 {
            assert!(std::time::Instant::now() < deadline, "the dispatched result check was never sent");
            std::thread::yield_now();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn render_document_png_refuses_an_empty_canvas_before_any_request() {
        let doc = fotodit_doc::Document::new("empty", fotodit_geom::Size::new(0, 0),
            fotodit_doc::ColorMode::Rgb, fotodit_doc::SampleType::U8);
        let error = render_document_png(&doc).expect_err("an empty canvas cannot produce a result image");
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn visual_goal_discarded_when_document_changes_before_apply() {
        use std::sync::atomic::Ordering;
        let (mut app, calls, _) = visual_fixture(vec![]);
        let ctx = egui::Context::default();
        submit_visual(&mut app, &ctx);
        app.session.execute("file.new", serde_json::json!({"width":16,"height":16})).unwrap();
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("cancelled because the document"), "{msg}");
        assert!(msg.contains("may still finish"), "the in-flight send is disclosed: {msg}");
        assert!(msg.contains("captured image of the document"), "the image the in-flight plan carries is disclosed: {msg}");
        assert!(msg.contains("no further requests will be made"), "{msg}");
        // The plan request was dispatched (worker thread spawned at submit) before the
        // discard was observed; the send itself may still be in flight. Wait for it
        // before counting, so the assertion does not race thread scheduling.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while calls.load(Ordering::SeqCst) < 1 {
            assert!(std::time::Instant::now() < deadline, "the plan request was never sent");
            std::thread::yield_now();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the plan request was sent but nothing applied");
    }

}
