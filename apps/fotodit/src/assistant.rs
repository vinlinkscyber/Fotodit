//! Chat assistant bridge: composer prompts run through `fotodit_automation`'s AI
//! pipeline (deterministic offline planner by default) against the app's admitted
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
use fotodit_automation::ai::{AiContext, AiEngine, EditingIntent, EditingPlan, ExecutionEvidence, GoalOutcome, GoalRequest, GoalStop, IntentCategory, LiveEditSession, SemanticGoalRepair, VisualEvidenceClass};
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

/// Panel backend holding the long-lived engine and conversational context across prompts.
pub struct GuiAssistant {
    engine: AiEngine,
    context: AiContext,
    planner_factory: Box<dyn Fn(&str) -> Result<(SelectedPlanner, [u8; 32]), String>>,
    identity: fn(&str) -> Result<[u8; 32], String>,
    pending: Option<PendingGoal>,
}

impl GuiAssistant {
    /// Offline at startup. Only the in-app, session-only consent can enable a provider.
    pub fn new() -> Self {
        Self { engine: AiEngine::new(), context: AiContext::new(), planner_factory: Box::new(provider_planner), identity: provider_identity, pending: None }
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
            // This bounded route is single-turn. Do not guess what an earlier request meant.
            let lower = prompt.to_lowercase();
            // Only bare conversational references are refused. Pronouns inside a full
            // request (including the user's taxi-poster wording) are ordinary language.
            if matches!(lower.trim().trim_end_matches(['.', '!', '?']), "do that again" | "same as previous" | "again" | "same" | "do it" | "do that") {
                return Err("Provider editing is single-turn and does not share earlier chat. Restate the complete edit and name the target explicitly; nothing was sent.".into());
            }
            let document = app.session.active().map(fotodit_engine::inspect::document);
            let scene = document.as_ref().map(|doc| fotodit_automation::ai::PerceptionEngine::new().perceive(doc)).unwrap_or_default();
            let intent = fotodit_automation::ai::NaturalLanguageEngine::new().parse_intent(prompt, &scene, &AiContext::new()).map_err(|e| e.to_string())?;
            if document.is_none() && intent.category != IntentCategory::GraphicDesign {
                return Err("Open a document before editing, or make an explicit poster/banner creation request. Nothing was sent.".into());
            }
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
            std::thread::spawn(move || {
                let result = planner.plan_visual(&worker_intent, &scene, &AiContext::new(), image.as_deref())
                    .map(WorkerResult::Plan).map_err(|e| e.to_string());
                let _ = sender.send(result);
                repaint.request_repaint();
            });
            self.pending = Some(PendingGoal { phase: PendingPhase::Planning, receiver, prompt: prompt.into(),
                intent, applied: Vec::new(), document, document_id: document_id(app), binding, visual });
            Ok(if visual {
                format!("Provider planning ({slot}) with the document image attached. No edit has been applied yet.")
            } else {
                format!("Provider planning ({slot}). No edit has been applied yet.")
            })
        } else {
            let mut session = GuiSession { app, ctx };
            self.run(&mut session, prompt).map(|answer| format!("Offline rules (no provider contacted). {answer}"))
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
                (true, true) => " The result check already sent may still finish; it carries a captured document image, and no further requests will be made for this goal.",
                (true, false) => " The result check already sent may still finish, and no further requests will be made for this goal.",
                (false, true) => " The planning request already sent may still finish; it carries the captured document image. Nothing from it will be applied and no further requests will be made for this goal.",
                (false, false) => " The planning request already sent may still finish; nothing from it will be applied and no further requests will be made for this goal.",
            };
            return Some(if verifying {
                format!("Result check discarded because the document, credential or consent changed. Edits already applied remain in the document; nothing was reverted.{in_flight} Inspect the canvas and use Undo if needed.")
            } else {
                format!("Pending edit discarded because the document, credential or consent changed. No response was applied.{in_flight}")
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
                "Result check discarded because the document or credential changed. Edits already applied remain in the document; nothing was reverted. Inspect the canvas and use Undo if needed.".into()
            } else {
                "Pending edit discarded because the document or credential changed. No response was applied.".into()
            });
        }
        let plan = match (phase, result) {
            (PendingPhase::Planning, Err(error)) => return Some(error),
            (PendingPhase::Verifying { .. }, Err(error)) => return Some(format!("The result check failed: {error} Changes made so far remain; inspect the canvas and use Undo as needed.")),
            (PendingPhase::Planning, Ok(WorkerResult::Plan(plan))) => plan,
            (PendingPhase::Verifying { check }, Ok(WorkerResult::Verdict(verdict))) => {
                let verdict = *verdict;
                if verdict.satisfied {
                    return Some(format!("Gemini reviewed the edited image and reported the request satisfied (result check {check}). That is the provider's own judgment of the image, not independent proof. Inspect the canvas; use Undo if needed."));
                }
                if verdict.corrective.steps.is_empty() {
                    let note = if verdict.critique.is_empty() { String::new() } else { format!(": {}", verdict.critique) };
                    return Some(format!("Gemini reviewed the edited image and reported the goal not met, with no correction to offer{note}. Changes made so far remain; inspect the canvas and use Undo as needed."));
                }
                verdict.corrective
            }
            _ => return Some("Provider response did not match the pending phase. No edits were applied.".into()),
        };
        pending.applied.extend(plan.steps.iter().map(|s| {
            let text = format!("{} {}", s.command, s.params);
            text.chars().take(200).collect()
        }));
        // UI-thread application only, after the immutable snapshot and consent still match.
        let outcome = {
            let engine = AiEngine::with_planner(Box::new(FixedPlan(plan)));
            let mut session = GuiSession { app, ctx };
            let request = GoalRequest { max_attempts: 1, ..GoalRequest::new(&pending.prompt) };
            engine.process_goal_in(&mut session, &request, &mut AiContext::new(), &SemanticGoalRepair::new())
        };
        let completed = outcome.attempts.iter().any(|a| a.execution.as_ref().is_some_and(|e| e.is_success));
        if completed {
            // The apply changed the document: re-anchor the staleness snapshot so
            // the next round compares against the post-edit state, not the original.
            pending.document = app.session.active().map(fotodit_engine::inspect::document);
            pending.document_id = document_id(app);
        }
        let next_check = match phase { PendingPhase::Planning => 1, PendingPhase::Verifying { check } => check + 1 };
        // Requests actually sent for this goal so far; result checks only count once sent.
        let requests_made = match phase { PendingPhase::Planning => 1, PendingPhase::Verifying { check } => 1 + check };
        let can_check = pending.visual && completed
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
                        return Some("Provider credential changed between result checks. Enable editing again; changes made so far remain.".into());
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
                    return Some(format!("Provider editing ({slot}): commands executed. Asking Gemini to check the result image (request {} of at most {}). Inspect the canvas; use Undo if needed. {}",
                        next_check + 1, 1 + fotodit_ui_egui::ai_providers_ui::VISUAL_MAX_CHECKS, outcome.detail));
                }
                Some(Err(render_error)) => {
                    return Some(format!("Provider editing ({slot}): commands executed; {requests_made} provider request(s) were made. The result image could not be rendered, so no result check was sent: {render_error} The requested result and visual quality are not independently verified. Inspect the canvas; use Undo if needed. {}", outcome.detail));
                }
                None => {
                    return Some(format!("Provider editing ({slot}): commands executed; {requests_made} provider request(s) were made. The document is no longer open, so no result check was sent. Inspect the canvas; use Undo if needed. {}", outcome.detail));
                }
            }
        }
        Some(if completed {
            if pending.visual && next_check > fotodit_ui_egui::ai_providers_ui::VISUAL_MAX_CHECKS {
                format!("Provider editing ({slot}): commands executed. The consented result checks are used up for this goal ({requests_made} requests were made). The requested result and visual quality are not independently verified. Inspect the canvas; use Undo if needed. {}", outcome.detail)
            } else if pending.visual {
                format!("Provider editing ({slot}): commands executed; {requests_made} provider request(s) were made. Image consent is no longer active, so no result check was sent. The requested result and visual quality are not independently verified. Inspect the canvas; use Undo if needed. {}", outcome.detail)
            } else {
                format!("Provider editing ({slot}): commands executed. The requested result and visual quality are not independently verified. Inspect the canvas; use Undo if needed. {}", outcome.detail)
            }
        } else if outcome.attempts.iter().any(|a| !a.commands.is_empty()) {
            format!("Provider editing ({slot}): execution stopped and may have left partial changes. Inspect the canvas and use Undo as needed. {}", outcome.detail)
        } else { format!("Provider editing ({slot}): no edits applied. {}", outcome.detail) })
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

impl GuiAssistant {
    /// Execute a request through the bounded goal coordinator: the original prompt is
    /// parsed once and never re-parsed from feedback, every attempt runs through the
    /// admitted completed-value dispatch above, and only a completed-but-unmet attempt
    /// invokes the semantic repair adapter - which rewrites type layers this goal's own
    /// attempts created, never pre-existing user text. There is no retry after an
    /// execution error, and a pending/dialog receipt stops dispatch with the final state
    /// marked unverified rather than reported as done.
    fn run(&mut self, session: &mut GuiSession<'_>, prompt: &str) -> Result<String, String> {
        let outcome = self.run_goal(session, prompt)?;
        Ok(summarize_goal(&outcome))
    }

    /// The single wiring behind the chat reply, exposed privately so tests assert on the
    /// ACTUAL outcome of the same run rather than re-invoking the coordinator a different
    /// way: one bounded goal run through this assistant's engine and context with the
    /// semantic repair adapter. No behavior change from inlining it in `run`.
    fn run_goal(&mut self, session: &mut GuiSession<'_>, prompt: &str) -> Result<GoalOutcome, String> {
        let repair = SemanticGoalRepair::new();
        let outcome = self.engine.process_goal_in(session, &GoalRequest::new(prompt), &mut self.context, &repair);
        if outcome.stop == GoalStop::InvalidRequest {
            return Err(outcome.detail);
        }
        Ok(outcome)
    }
}

/// Edits big enough that the design belongs in the reply next to the outcome.
fn is_significant(plan: &EditingPlan) -> bool {
    plan.creates_document || plan.steps.len() > 1 || plan.category == IntentCategory::GraphicDesign
}

/// The executed design in plain language: what was planned and the ordered steps.
fn propose(plan: &EditingPlan) -> String {
    let steps = plan.steps.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s.description)).collect::<Vec<_>>().join(" ");
    format!("Plan: {}. {} Steps: {}", plan.title, plan.rationale, steps)
}

/// The chat reply for a finished goal run. The stop reason decides what may honestly
/// be said: only a Completed run reports satisfaction (evaluator evidence, never
/// independent visual-quality proof) and may include the pixel observation; a run
/// whose final state is unverified says so and withholds it. A significant executed
/// design (new document, multi-step work, creative design) is named next to the outcome.
fn summarize_goal(outcome: &GoalOutcome) -> String {
    let attempts = outcome.attempts.len();
    let last = outcome.attempts.last();
    let mut text = match outcome.stop {
        GoalStop::Completed => {
            let title = last.map(|a| a.plan.title.as_str()).unwrap_or("the edit");
            let steps = last.map(|a| a.plan.steps.len()).unwrap_or(0);
            let mut t = if attempts > 1 { format!("Done: {title} ({steps} steps, {attempts} attempts).") } else { format!("Done: {title} ({steps} steps).") };
            t.push_str(" The evaluator reported the goal satisfied; that is evaluator evidence, not independent visual-quality proof.");
            if let Some(observation) = last.and_then(|a| a.evidence.as_ref()).and_then(pixel_observation) {
                t.push(' ');
                t.push_str(&observation);
            }
            t
        }
        GoalStop::NeedsInput if outcome.detail == "no document open" => "Open a document first, or ask me to create one.".to_string(),
        GoalStop::NeedsInput => format!("I need more input: {}", outcome.detail),
        GoalStop::Unmet => format!("The goal is not met: {}", outcome.detail),
        GoalStop::BudgetExhausted => format!("Stopped: {}", outcome.detail),
        GoalStop::RepeatedPlan => format!("Stopped before repeating the same edit: {}", outcome.detail),
        GoalStop::PlanningFailed => format!("I could not plan that: {}", outcome.detail),
        GoalStop::InvalidRequest => outcome.detail.clone(),
        GoalStop::FinalStateUnverified => {
            format!("The final document state is unverified - check the canvas before relying on it. {}", outcome.detail)
        }
    };
    if let Some(plan) = outcome.attempts.first().map(|a| &a.plan).filter(|p| is_significant(p)) {
        text = format!("{} {text}", propose(plan));
    }
    text
}

/// One bounded, honest line of observed-composite evidence: what a before/after
/// pixel digest OBSERVED, explicitly not a verdict on the intended result (the
/// evaluation above owns intent). Nothing is reported when no step was checked.
///
/// Aggregate cost behind this line: one engine pass is bounded at 8 digest
/// renders + 2 color-statistics renders (about 640,000,000 decimal bytes of
/// cumulative composite output; not peak, no time bound), and this assistant
/// can run two passes per prompt (initial + one refinement), doubling the
/// aggregate (16 digests, 4 statistics renders, ~1.28 billion decimal bytes).
fn pixel_observation(evidence: &ExecutionEvidence) -> Option<String> {
    let mut changed = 0usize;
    let mut no_change = 0usize;
    let mut unknown = 0usize;
    let mut first_unknown: Option<&str> = None;
    for step in &evidence.steps {
        match step.visual.class {
            VisualEvidenceClass::Changed => changed += 1,
            VisualEvidenceClass::NoChangeObserved => no_change += 1,
            VisualEvidenceClass::Unknown => {
                unknown += 1;
                first_unknown.get_or_insert(step.visual.detail.as_str());
            }
            VisualEvidenceClass::NotApplicable => {}
        }
    }
    if changed + no_change + unknown == 0 {
        return None;
    }
    let mut line = "Pixel check (observation, not intent verification): ".to_string();
    let mut segments = 0;
    if changed > 0 {
        line.push_str(&format!("sampled composite data/geometry changed on {changed} visual step(s); a data difference is not proof of visible improvement, intent or quality"));
        segments += 1;
    }
    if no_change > 0 {
        if segments > 0 {
            line.push_str("; ");
        }
        line.push_str(&format!("no difference observed on the sampled, 8-bit-quantized grid on {no_change} step(s); sampling and quantization can hide changes, and idempotent steps legitimately land here"));
        segments += 1;
    }
    if unknown > 0 {
        if segments > 0 {
            line.push_str("; ");
        }
        line.push_str(&format!("{unknown} step(s) unverified"));
        if let Some(reason) = first_unknown {
            line.push_str(&format!(" ({reason})"));
        }
    }
    line.push('.');
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fotodit_automation::ai::{GoalAttempt, StepVisualEvidence, VisualEvidence};
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
            assert!(initial.starts_with("Provider planning"), "{prompt}: {initial}");
            let _ = await_plan(&mut app, &ctx);
            assert_eq!(calls.load(Ordering::SeqCst), 1, "{prompt}");
        }
    }

    #[test]
    fn provider_no_document_or_ambiguous_followup_refuses_before_network() {
        use std::sync::atomic::Ordering;
        for prompt in ["enhance image quality", "do that again", "make it blue", "same as previous"] {
            let (mut app, calls, loads) = chat_fixture(false);
            app.ai_providers.editing_consent_mut().request("gemini");
            app.ai_providers.editing_consent_mut().confirm_identity([7; 32]);
            let reply = fotodit_ui_egui::panels::assistant::submit(&mut app, &egui::Context::default(), prompt);
            assert!(reply.contains("Nothing was sent") || reply.contains("nothing was sent"), "{reply}");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(loads.load(Ordering::SeqCst), 0);
        }
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
                assert!(answer.contains("discarded"), "{answer}");
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
        assert!(initial.starts_with("Provider planning (gemini)"), "{initial}");
        let answer = await_plan(&mut app, &ctx);
        assert!(answer.starts_with("Provider editing (gemini):"), "{answer}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        let doc = fotodit_engine::inspect::document(app.session.active().unwrap());
        assert_eq!(doc["width"], 240, "provider plan, not offline 1200px poster");
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

    fn attempt(title: &str, classes: &[VisualEvidenceClass]) -> GoalAttempt {
        GoalAttempt {
            plan: EditingPlan { title: title.to_string(), rationale: String::new(), category: IntentCategory::PhotoEnhancement, steps: vec![], creates_document: false, target_size: None },
            execution: None,
            evaluation: None,
            evidence: Some(evidence(classes)),
            commands: vec![],
            error: None,
        }
    }

    fn outcome(stop: GoalStop, detail: &str, attempts: Vec<GoalAttempt>) -> GoalOutcome {
        GoalOutcome {
            original_prompt: "p".to_string(),
            original_intent: None,
            attempts,
            stop,
            detail: detail.to_string(),
            final_state_unverified: stop == GoalStop::FinalStateUnverified,
            planned_commands: 0,
        }
    }

    fn evidence(classes: &[VisualEvidenceClass]) -> ExecutionEvidence {
        ExecutionEvidence {
            steps: classes
                .iter()
                .enumerate()
                .map(|(i, class)| StepVisualEvidence {
                    step_index: i,
                    visual: VisualEvidence {
                        class: class.clone(),
                        detail: if *class == VisualEvidenceClass::Unknown { "capture budget (8) is exhausted".to_string() } else { "sampled".to_string() },
                    },
                })
                .collect(),
        }
    }

    #[test]
    fn summarize_goal_completed_is_evaluator_evidence_not_visual_proof() {
        let text = summarize_goal(&outcome(GoalStop::Completed, "completed", vec![attempt("enhance", &[VisualEvidenceClass::Changed])]));
        assert!(text.contains("Done: enhance"), "{text}");
        assert!(text.contains("evaluator evidence, not independent visual-quality proof"), "{text}");
        assert!(text.contains("Pixel check"), "{text}");
    }

    #[test]
    fn summarize_goal_unverified_withholds_pixel_observation() {
        let text = summarize_goal(&outcome(GoalStop::FinalStateUnverified, "host returned a pending receipt", vec![attempt("enhance", &[VisualEvidenceClass::Changed])]));
        assert!(text.contains("final document state is unverified"), "{text}");
        assert!(text.contains("pending receipt"), "{text}");
        assert!(!text.contains("Pixel check"), "{text}");
        assert!(!text.contains("Done:"), "{text}");
    }

    #[test]
    fn summarize_goal_needs_input_without_document_is_the_open_first_line() {
        let text = summarize_goal(&outcome(GoalStop::NeedsInput, "no document open", vec![]));
        assert_eq!(text, "Open a document first, or ask me to create one.");
    }

    #[test]
    fn summarize_goal_repeated_plan_and_unmet_keep_the_detail() {
        let text = summarize_goal(&outcome(GoalStop::RepeatedPlan, "same sequence proposed again", vec![attempt("enhance", &[])]));
        assert!(text.contains("Stopped before repeating the same edit"), "{text}");
        assert!(text.contains("same sequence proposed again"), "{text}");
        let text = summarize_goal(&outcome(GoalStop::Unmet, "need another strategy", vec![attempt("enhance", &[])]));
        assert!(text.contains("The goal is not met: need another strategy"), "{text}");
    }

    /// Outcome-level assertions through the ACTUAL run wiring (the `run_goal` seam the
    /// chat reply itself uses, not a second coordinator invocation path): the user's
    /// verbatim poster brief reaches GoalStop::Completed in ONE attempt, the plan creates
    /// the document and both text layers with the user's strings, the executed document
    /// matches, and the real output is rendered to /tmp/poster_render.png for inspection.
    #[test]
    fn goal_outcome_poster_completed_with_interpreted_purpose_and_measured_geometry() {
        let s = fotodit_engine::Session::new();
        let mut app = FotoditApp::new(s, Services::default());
        app.background_jobs = true;
        let ctx = egui::Context::default();
        let mut assistant = GuiAssistant::new();
        let outcome = {
            let mut session = GuiSession { app: &mut app, ctx: &ctx };
            assistant.run_goal(&mut session, "create a poster for my taxi busines to promote it its called chndrax").unwrap()
        };
        assert_eq!(outcome.stop, GoalStop::Completed, "{}", outcome.detail);
        assert_eq!(outcome.attempts.len(), 1, "the poster goal must complete without repair: {}", outcome.detail);
        assert!(!outcome.final_state_unverified);
        let steps = &outcome.attempts[0].plan.steps;
        let commands: Vec<&str> = steps.iter().map(|s| s.command.as_str()).collect();
        assert_eq!(commands, ["file.new", "layer.newFillLayer.solidColor", "type.create", "type.create"], "{commands:?}");
        let texts: Vec<&str> = steps.iter().filter(|s| s.command == "type.create").filter_map(|s| s.params["text"].as_str()).collect();
        assert_eq!(texts, ["chndrax", "Taxi business"], "the user's name lands verbatim and the stated purpose lands as interpreted display copy: {texts:?}");
        // Execution, not just planning: the document itself carries both layers.
        let document = fotodit_engine::inspect::document(app.session.active().unwrap());
        assert_eq!(document["width"].as_u64(), Some(1200));
        assert_eq!(document["height"].as_u64(), Some(1600));
        let doc_text: String = document["layers"].as_array().unwrap().iter().filter_map(|l| l["text"]["text"].as_str()).collect::<Vec<_>>().join(" ");
        assert!(doc_text.contains("chndrax") && doc_text.contains("Taxi business"), "{doc_text}");
        // Rendered-geometry assertions on the REAL layer content bounds (document
        // pixels): the headline must be legible at canvas scale, horizontally
        // centered, and the subheading below it without overlap - the defect the
        // user hit was text rendering tiny and scattered on a blank canvas.
        let layers = document["layers"].as_array().unwrap();
        let bounds_of = |text: &str| -> [i64; 4] {
            let b = layers.iter().find(|l| l["text"]["text"].as_str() == Some(text)).unwrap_or_else(|| panic!("layer '{text}'"))["bounds"].as_array().unwrap().clone();
            [b[0].as_i64().unwrap(), b[1].as_i64().unwrap(), b[2].as_i64().unwrap(), b[3].as_i64().unwrap()]
        };
        let actual_headline = layers.iter().find(|l| l["text"]["text"].as_str() == Some("chndrax")).unwrap();
        let planned_size = steps.iter().find(|s| s.command == "type.create").unwrap().params["size"].as_f64().unwrap();
        assert_eq!(actual_headline["text"]["sizePt"].as_f64(), Some(planned_size), "native size key must reach the model, not silently default to 12pt");
        let [hx, hy, hw, hh] = bounds_of("chndrax");
        assert!(hh as f64 >= 1600.0 * 0.035, "headline ink must be legible at canvas scale: {hh}px of 1600px");
        assert!(hx >= 0 && hx + hw <= 1200, "headline ink on-canvas horizontally: x={hx} w={hw}");
        let center_offset = ((2 * hx + hw) - 1200).abs();
        assert!(center_offset <= 1200 / 16, "headline horizontally centered: offset {}px", center_offset / 2);
        let [sx, sy, sw, sh] = bounds_of("Taxi business");
        assert!(sy >= hy + hh, "subheading below the headline without overlap: headline y={hy} h={hh}, sub y={sy}");
        assert!(sy + sh <= 1600, "subheading on-canvas vertically: y={sy} h={sh}");
        assert!(sw > 0 && sx >= 0 && sx + sw <= 1200, "subheading ink on-canvas: x={sx} w={sw}");
        // Export actual output pixels and verify a temporary PNG write/readback.
        let png = fotodit_io::export(&app.session.active().unwrap().doc, "png", &fotodit_io::ExportOptions::default()).unwrap();
        assert!(png.bytes.starts_with(&[0x89, b'P', b'N', b'G']), "a real PNG render of the poster");
        let output = tempfile::tempdir().unwrap();
        let path = output.path().join("poster_render.png");
        std::fs::write(&path, &png.bytes).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), png.bytes, "rendered PNG is written intact");
    }

    #[test]
    fn goal_outcome_long_copy_wraps_with_readable_rendered_bounds() {
        let mut app = FotoditApp::new(fotodit_engine::Session::new(), Services::default());
        app.background_jobs = true;
        let ctx = egui::Context::default();
        let mut assistant = GuiAssistant::new();
        let outcome = {
            let mut session = GuiSession { app: &mut app, ctx: &ctx };
            assistant.run_goal(&mut session, "Create an elegant wedding invitation").unwrap()
        };
        assert_eq!(outcome.stop, GoalStop::Completed, "{} {:?}", outcome.detail, outcome.attempts);
        let document = fotodit_engine::inspect::document(app.session.active().unwrap());
        for layer in document["layers"].as_array().unwrap().iter().filter(|l| l["kind"] == "Type") {
            let b = layer["bounds"].as_array().unwrap();
            let (x,y,w,h) = (b[0].as_i64().unwrap(),b[1].as_i64().unwrap(),b[2].as_i64().unwrap(),b[3].as_i64().unwrap());
            assert!(x >= 0 && y >= 0 && x+w <= 1200 && y+h <= 1600, "wrapped text must stay on canvas: {layer}");
        }
        let png = fotodit_io::export(&app.session.active().unwrap().doc, "png", &fotodit_io::ExportOptions::default()).unwrap();
        let output = tempfile::tempdir().unwrap();
        let path = output.path().join("wedding_render.png");
        std::fs::write(&path, &png.bytes).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), png.bytes, "rendered PNG is written intact");
    }

    /// The original failing workflow at outcome level: "Enhance image quality" on a plain
    /// open document runs the three-step enhancement plan inline; the same goal aimed at a
    /// poster's type layer is REFUSED honestly (no edit to the type layer, the user text
    /// intact) instead of faking success.
    #[test]
    fn goal_outcome_enhance_runs_on_pixels_and_refuses_type_layer() {
        let mut s = fotodit_engine::Session::new();
        s.execute("file.new", serde_json::json!({"width": 32, "height": 32, "background": "white"})).unwrap();
        let mut app = FotoditApp::new(s, Services::default());
        app.background_jobs = true;
        let ctx = egui::Context::default();
        let mut assistant = GuiAssistant::new();
        let outcome = {
            let mut session = GuiSession { app: &mut app, ctx: &ctx };
            assistant.run_goal(&mut session, "Enhance image quality").unwrap()
        };
        assert_eq!(outcome.stop, GoalStop::Completed, "{}", outcome.detail);
        let commands: Vec<&str> = outcome.attempts[0].plan.steps.iter().map(|s| s.command.as_str()).collect();
        assert_eq!(commands, ["filter.sharpen.unsharpMask", "layer.newAdjustmentLayer.curves", "layer.newAdjustmentLayer.vibrance"], "{commands:?}");
        let doc = app.session.active().unwrap().doc.id;
        assert!(app.session.job_on(doc).is_none(), "no job may remain running");

        // A poster's type layer is not an enhancement target: the refusal must be explicit
        // and the user's text untouched. Fresh app with only the poster document.
        let mut app2 = FotoditApp::new(fotodit_engine::Session::new(), Services::default());
        app2.background_jobs = true;
        let mut assistant2 = GuiAssistant::new();
        {
            let mut session = GuiSession { app: &mut app2, ctx: &ctx };
            assistant2.run_goal(&mut session, "create a poster for my taxi busines to promote it its called chndrax").unwrap();
        }
        let outcome = {
            let mut session = GuiSession { app: &mut app2, ctx: &ctx };
            assistant2.run_goal(&mut session, "Enhance image quality").unwrap()
        };
        assert_eq!(outcome.stop, GoalStop::FinalStateUnverified, "{}", outcome.detail);
        assert!(outcome.detail.contains("not an eligible pixel layer"), "{}", outcome.detail);
        let document = fotodit_engine::inspect::document(app2.session.active().unwrap());
        let doc_text: String = document["layers"].as_array().unwrap().iter().filter_map(|l| l["text"]["text"].as_str()).collect::<Vec<_>>().join(" ");
        assert!(doc_text.contains("chndrax") && doc_text.contains("Taxi business"), "the refusal must not touch the text: {doc_text}");
    }

    /// Integrated flow through the REAL app: a poster goal with no document open runs the
    /// coordinator end to end (document creation, type layers, evaluation, repair) against
    /// the admitted completed-value dispatch with background jobs enabled.
    #[test]
    fn goal_flow_creates_chndrax_poster_end_to_end() {
        let s = fotodit_engine::Session::new();
        let mut app = FotoditApp::new(s, Services::default());
        app.background_jobs = true;
        let ctx = egui::Context::default();
        let mut assistant = GuiAssistant::new();
        let reply = assistant.reply(&mut app, &ctx, "create a poster for my taxi busines to promote it its called chndrax").unwrap();
        assert!(!reply.contains("unverified"), "{reply}");
        let document = fotodit_engine::inspect::document(app.session.active().expect("the goal must leave a document"));
        let text: String = document["layers"].as_array().unwrap().iter().filter_map(|layer| layer["text"]["text"].as_str()).collect::<Vec<_>>().join(" ");
        assert!(text.to_lowercase().contains("chndrax"), "the requested name must land: {text}");
        assert!(app.ui.dialogs.is_empty(), "the goal flow must not open dialogs");
    }

    /// The original failing workflow: an enhancement goal on an open document completes
    /// inline even with background jobs enabled - no pending receipt, no queued job.
    #[test]
    fn goal_flow_enhances_open_document_inline_despite_background_jobs() {
        let mut s = fotodit_engine::Session::new();
        s.execute("file.new", serde_json::json!({"width": 32, "height": 32, "background": "white"})).unwrap();
        let mut app = FotoditApp::new(s, Services::default());
        app.background_jobs = true;
        let ctx = egui::Context::default();
        let mut assistant = GuiAssistant::new();
        let before = fotodit_engine::inspect::document(app.session.active().unwrap())["history"].as_array().unwrap().len();
        let reply = assistant.reply(&mut app, &ctx, "Enhance image quality").unwrap();
        assert!(!reply.contains("unverified"), "{reply}");
        let document = fotodit_engine::inspect::document(app.session.active().unwrap());
        assert!(document["history"].as_array().unwrap().len() > before, "the enhancement must record history");
        let doc = app.session.active().unwrap().doc.id;
        assert!(app.session.job_on(doc).is_none(), "no job may remain running");
        assert!(app.ui.dialogs.is_empty(), "the goal flow must not open dialogs");
    }

    #[test]
    fn pixel_observation_changed_wording_is_data_only() {
        let line = pixel_observation(&evidence(&[VisualEvidenceClass::Changed])).unwrap();
        assert!(line.contains("sampled composite data/geometry changed"), "{line}");
        assert!(line.contains("not proof of visible improvement, intent or quality"), "{line}");
        assert!(!line.contains("canvas change observed"), "{line}");
    }

    #[test]
    fn pixel_observation_no_change_keeps_sampling_caveat() {
        let line = pixel_observation(&evidence(&[VisualEvidenceClass::NoChangeObserved])).unwrap();
        assert!(line.contains("sampling and quantization can hide changes"), "{line}");
        assert!(line.contains("idempotent"), "{line}");
        assert!(!line.contains("image equality"), "{line}");
    }

    #[test]
    fn pixel_observation_reports_mixed_classes_and_first_unknown_reason() {
        let line = pixel_observation(&evidence(&[VisualEvidenceClass::Changed, VisualEvidenceClass::NoChangeObserved, VisualEvidenceClass::Unknown])).unwrap();
        assert!(line.contains("data/geometry changed on 1 visual step(s)"), "{line}");
        assert!(line.contains("no difference observed on the sampled, 8-bit-quantized grid on 1 step(s)"), "{line}");
        assert!(line.contains("1 step(s) unverified (capture budget (8) is exhausted)"), "{line}");
    }

    #[test]
    fn pixel_observation_is_none_without_checked_steps() {
        assert!(pixel_observation(&evidence(&[VisualEvidenceClass::NotApplicable])).is_none());
        assert!(pixel_observation(&evidence(&[])).is_none());
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
        assert!(first.contains("Asking Gemini to check the result image"), "{first}");
        let second = await_plan(&mut app, &ctx);
        assert!(second.contains("reported the request satisfied"), "{second}");
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
        assert!(first.contains("request 2 of at most 3"), "{first}");
        let second = await_plan(&mut app, &ctx);
        assert!(second.contains("reported the request satisfied"), "{second}");
        assert!(second.contains("not independent proof"), "{second}");
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
        assert!(final_msg.contains("result checks are used up"), "{final_msg}");
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
        assert!(initial.starts_with("Provider planning"), "{initial}");
        assert!(!initial.contains("image attached"), "{initial}");
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("commands executed"), "{msg}");
        assert!(msg.contains("not independently verified"), "{msg}");
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
        assert!(msg.contains("Image consent is no longer active"), "{msg}");
        assert!(msg.contains("1 provider request(s) were made"), "{msg}");
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
        assert!(first.contains("request 2 of at most 3"), "{first}");
        // Replace the document while the result check is in flight.
        app.session.execute("file.new", serde_json::json!({"width":16,"height":16})).unwrap();
        let msg = await_plan(&mut app, &ctx);
        assert!(msg.contains("Result check discarded"), "{msg}");
        assert!(msg.contains("Edits already applied remain"), "{msg}");
        assert!(!msg.contains("No response was applied"), "{msg}");
        assert!(msg.contains("may still finish"), "the in-flight send is disclosed: {msg}");
        assert!(msg.contains("captured document image"), "the image the in-flight check carries is disclosed: {msg}");
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
        assert!(msg.contains("discarded because the document"), "{msg}");
        assert!(msg.contains("may still finish"), "the in-flight send is disclosed: {msg}");
        assert!(msg.contains("captured document image"), "the image the in-flight plan carries is disclosed: {msg}");
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
