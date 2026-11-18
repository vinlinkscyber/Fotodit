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
use fotodit_automation::ai::{AiContext, AiEngine, EditingPlan, ExecutionEvidence, GoalOutcome, GoalRequest, GoalStop, IntentCategory, LiveEditSession, SemanticGoalRepair, VisualEvidenceClass};
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
}

impl GuiAssistant {
    /// Default: the offline deterministic planner. Provider planning is opt-in via the
    /// FOTODIT_AI_PROVIDER=provider environment setting so no billable provider call can
    /// happen unless explicitly enabled; without provider keys in the environment the
    /// deterministic planner is always used.
    pub fn new() -> Self {
        let provider_opt_in = std::env::var("FOTODIT_AI_PROVIDER").map(|v| v.eq_ignore_ascii_case("provider")).unwrap_or(false);
        let config = fotodit_automation::ai::ProviderConfig::from_env();
        let has_keys = [fotodit_automation::ai::ProviderId::Gemini, fotodit_automation::ai::ProviderId::OpenRouter, fotodit_automation::ai::ProviderId::Cohere, fotodit_automation::ai::ProviderId::Pollinations, fotodit_automation::ai::ProviderId::Lajiwa]
            .iter()
            .any(|p| config.has_key(*p));
        let engine = if provider_opt_in && has_keys { AiEngine::with_provider_config(config) } else { AiEngine::new() };
        Self { engine, context: AiContext::new() }
    }
}

impl AssistantBackend for GuiAssistant {
    fn reply(&mut self, app: &mut FotoditApp, ctx: &egui::Context, prompt: &str) -> Result<String, String> {
        let mut session = GuiSession { app, ctx };
        self.run(&mut session, prompt)
    }
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

}
