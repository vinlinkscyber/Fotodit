//! Chat assistant bridge: composer prompts run through `fotodit_automation`'s AI
//! pipeline (deterministic offline planner by default) against the app's admitted
//! command dispatch, per the `LiveEditSession` host contract: the app's real session is
//! borrowed for the duration of one operation, commands go through the admitted menu
//! path (authorization and discard interception preserved), and results are completed
//! values, never pending job receipts.

use fotodit_automation::AutomationError;
use fotodit_automation::ai::{AiContext, AiEngine, AiError, AiResult, EditingPlan, ExecutionEvidence, IntentCategory, LiveEditSession, VisualEvidenceClass};
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
        fotodit_ui_egui::menus::invoke(self.app, self.ctx, command, params).map_err(AutomationError::App)
    }

    fn visual_digest(&mut self, max_pixels: u64) -> Option<Result<serde_json::Value, String>> {
        // The same admitted menu path as every editing command: authorization
        // and discard interception are preserved exactly, and a refusal is
        // reported as capture failure - never bypassed.
        Some(fotodit_ui_egui::menus::invoke(self.app, self.ctx, "image.analysis.pixelDigest", serde_json::json!({"maxPixels": max_pixels})))
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
        // Plan first: significant edits (new documents, multi-step work, creative design) are
        // planned against the live document BEFORE anything executes, and the reply reports
        // that design next to the verified outcome. Non-destructive work runs autonomously.
        let planned = match self.engine.plan_instruction_in(&session, prompt, &self.context) {
            Ok(plan) => Some(plan),
            Err(AiError::NoDocumentOpen) => return Err("Open a document first, or ask me to create one.".to_string()),
            Err(_) => None,
        };
        let significant = planned.as_ref().is_some_and(is_significant);
        match self.run(&mut session, prompt) {
            Ok(done) if significant => Ok(format!("{} {}", propose(planned.as_ref().expect("checked above")), done)),
            other => other,
        }
    }
}

impl GuiAssistant {
    /// Execute a request through the full inspect/plan/execute/evaluate loop, then run one
    /// bounded refinement pass when execution succeeded but the evaluation found the intent
    /// unmet. The second pass re-inspects the live document and carries the evaluator's
    /// findings, and execution rolls back atomically on failure.
    fn run(&mut self, session: &mut GuiSession<'_>, prompt: &str) -> Result<String, String> {
        // The GUI explicitly opts into observed-composite evidence; the legacy
        // entry point stays render-free for other hosts.
        match self.engine.process_instruction_in_with_evidence(session, prompt, &mut self.context) {
            Ok((result, evidence)) => {
                if result.execution.is_success && !result.evaluation.intent_satisfied && !result.evaluation.observations.is_empty() {
                    let follow_up = format!(
                        "The previous edit (\"{prompt}\") is partially done. Without repeating steps that already succeeded, apply only the corrections needed for these unmet points: {}.",
                        result.evaluation.observations.join("; ")
                    );
                    let refined = self.engine.process_instruction_in_with_evidence(session, &follow_up, &mut self.context);
                    return Ok(summarize(&refinement_report((result, evidence), refined)));
                }
                Ok(summarize(&RunReport::Final(result, evidence)))
            }
            Err(AiError::NoDocumentOpen) => Err("Open a document first, or ask me to create one.".to_string()),
            Err(error) => Err(format!("I couldn't complete that edit: {error}")),
        }
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

/// What the chat reply may honestly describe after an optional refinement
/// attempt: either the run whose mutations ARE the final canvas, or a completed
/// first pass followed by a failed refinement whose end state cannot be known.
enum RunReport {
    /// The latest SUCCESSFUL run and its evidence: its mutations are the canvas.
    Final(AiResult, ExecutionEvidence),
    /// The first pass completed, then the refinement failed - at the engine
    /// level or mid-plan. Rollback is best-effort and may be incomplete, so the
    /// first pass's evidence may be STALE: the reply retains the failure detail
    /// and withholds the pixel observation instead of presenting it as current.
    Unverified { first: AiResult, detail: String },
}

/// Fold the refinement outcome into what may honestly be reported. A successful
/// refinement has already mutated the document, so the latest successful
/// execution and ITS evidence are reported regardless of evaluator scores
/// (heuristics about quality, not about which mutations exist). A failed
/// refinement CANNOT be assumed rolled back, so the final state is unverified.
fn refinement_report(first: (AiResult, ExecutionEvidence), refined: Result<(AiResult, ExecutionEvidence), AiError>) -> RunReport {
    match refined {
        Ok((refined_result, refined_evidence)) if refined_result.execution.is_success => RunReport::Final(refined_result, refined_evidence),
        Ok((refined_result, _)) => RunReport::Unverified {
            first: first.0,
            detail: format!(
                "refinement stopped at step {} of {}: {}",
                refined_result.execution.executed_steps + 1,
                refined_result.execution.total_steps,
                refined_result.execution.error.as_deref().unwrap_or("unknown error")
            ),
        },
        Err(error) => RunReport::Unverified { first: first.0, detail: error.to_string() },
    }
}

/// Human-readable chat reply for a completed run: what was done, whether the visual
/// evaluation thinks the intent was met, and what the pixel observation saw.
fn summarize(report: &RunReport) -> String {
    let (result, evidence) = match report {
        RunReport::Final(result, evidence) => (result, evidence),
        RunReport::Unverified { first, detail } => {
            return format!(
                "Done: {} ({} of {} steps), but the result may not fully match the intent. A follow-up refinement was attempted and failed ({detail}); the final document state is unverified - check the canvas before relying on it.",
                first.plan.title, first.execution.executed_steps, first.execution.total_steps
            );
        }
    };
    let execution = &result.execution;
    if execution.is_success {
        let mut text = format!("Done: {} ({} of {} steps).", result.plan.title, execution.executed_steps, execution.total_steps);
        if !result.evaluation.intent_satisfied {
            text.push_str(" The result may not fully match the intent yet.");
        }
        if let Some(observation) = pixel_observation(evidence) {
            text.push(' ');
            text.push_str(&observation);
        }
        text
    } else {
        format!(
            "The edit stopped at step {} of {}: {}",
            execution.executed_steps + 1,
            execution.total_steps,
            execution.error.as_deref().unwrap_or("unknown error")
        )
    }
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
    use fotodit_automation::ai::{EvaluationReport, ExecutionResult, StepVisualEvidence, VisualEvidence, VisualScene};

    fn result(title: &str, success: bool, intent: bool, confidence: f32) -> AiResult {
        AiResult {
            prompt: "p".to_string(),
            initial_scene: VisualScene::default(),
            plan: EditingPlan { title: title.to_string(), rationale: String::new(), category: IntentCategory::PhotoEnhancement, steps: vec![], creates_document: false, target_size: None },
            execution: ExecutionResult {
                total_steps: 1,
                executed_steps: usize::from(success),
                step_executions: vec![],
                is_success: success,
                error: (!success).then(|| "boom".to_string()),
                undos_performed: 0,
            },
            evaluation: EvaluationReport { confidence_score: confidence, intent_satisfied: intent, contrast_check_passed: true, hierarchy_valid: true, layout_balance_score: 0.5, observations: vec![] },
            revision: 1,
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
    fn refinement_report_keeps_successful_refinement_regardless_of_scores() {
        // A successful refinement already mutated the canvas: the report must
        // describe it even when every heuristic score got WORSE. Reporting the
        // discarded first run would describe a canvas that no longer exists.
        let current = (result("first pass", true, false, 0.9), evidence(&[VisualEvidenceClass::NoChangeObserved]));
        let refined = Ok((result("refinement", true, false, 0.1), evidence(&[VisualEvidenceClass::Changed])));
        let RunReport::Final(kept, kept_evidence) = refinement_report(current, refined) else { panic!("successful refinement must be final") };
        assert_eq!(kept.plan.title, "refinement");
        assert_eq!(kept_evidence.steps[0].visual.class, VisualEvidenceClass::Changed);
    }

    #[test]
    fn refinement_report_marks_incomplete_refinement_unverified() {
        // Mid-plan failure: rollback may be incomplete, so the reply keeps the
        // first pass's completion, retains the failure detail, marks the final
        // state unverified, and WITHHOLDS the stale pixel observation.
        let current = (result("first pass", true, false, 0.4), evidence(&[VisualEvidenceClass::Changed]));
        let refined = Ok((result("refinement", false, false, 0.0), evidence(&[])));
        let text = summarize(&refinement_report(current, refined));
        assert!(text.contains("Done: first pass (1 of 1 steps)"), "{text}");
        assert!(text.contains("refinement stopped at step 1 of 1: boom"), "{text}");
        assert!(text.contains("final document state is unverified"), "{text}");
        assert!(!text.contains("Pixel check"), "{text}");
    }

    #[test]
    fn refinement_report_marks_engine_error_unverified() {
        // ACTUAL engine-Err path (e.g. inspection failure AFTER the refinement
        // already mutated the document): not a fabricated is_success=false.
        // Same contract: failure detail retained, final state unverified, stale
        // pixel observation withheld.
        let current = (result("first pass", true, false, 0.4), evidence(&[VisualEvidenceClass::Changed]));
        let refined: Result<(AiResult, ExecutionEvidence), AiError> = Err(AiError::PerceptionError("document switched mid-refinement".to_string()));
        let text = summarize(&refinement_report(current, refined));
        assert!(text.contains("Done: first pass (1 of 1 steps)"), "{text}");
        assert!(text.contains("visual perception failed: document switched mid-refinement"), "{text}");
        assert!(text.contains("final document state is unverified"), "{text}");
        assert!(!text.contains("Pixel check"), "{text}");
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

    #[test]
    fn summarize_reports_final_canvas_after_successful_unimproved_refinement() {
        // Successful refinement, evaluator still unsatisfied: the summary names
        // the refinement's plan (the canvas that exists) and stays honest that
        // intent may be unmet, with data-only pixel wording.
        let current = (result("first pass", true, false, 0.9), evidence(&[]));
        let refined = Ok((result("refinement pass", true, false, 0.2), evidence(&[VisualEvidenceClass::Changed])));
        let text = summarize(&refinement_report(current, refined));
        assert!(text.contains("refinement pass"), "{text}");
        assert!(text.contains("may not fully match the intent"), "{text}");
        assert!(text.contains("not proof of visible improvement"), "{text}");
    }
}
