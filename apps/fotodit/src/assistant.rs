//! Chat assistant bridge: composer prompts run through `fotodit_automation`'s AI
//! pipeline (deterministic offline planner by default) against the app's admitted
//! command dispatch, per the `LiveEditSession` host contract: the app's real session is
//! borrowed for the duration of one operation, commands go through the admitted menu
//! path (authorization and discard interception preserved), and results are completed
//! values, never pending job receipts.

use fotodit_automation::AutomationError;
use fotodit_automation::ai::{AiContext, AiEngine, AiError, AiResult, EditingPlan, IntentCategory, LiveEditSession};
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
        match self.engine.process_instruction_in(session, prompt, &mut self.context) {
            Ok(mut result) => {
                if result.execution.is_success && !result.evaluation.intent_satisfied && !result.evaluation.observations.is_empty() {
                    let follow_up = format!(
                        "The previous edit (\"{prompt}\") is partially done. Without repeating steps that already succeeded, apply only the corrections needed for these unmet points: {}.",
                        result.evaluation.observations.join("; ")
                    );
                    if let Ok(refined) = self.engine.process_instruction_in(session, &follow_up, &mut self.context)
                        && refined.execution.is_success
                        && (refined.evaluation.intent_satisfied
                            || refined.evaluation.confidence_score > result.evaluation.confidence_score)
                    {
                        result = refined;
                    }
                }
                Ok(summarize(&result))
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

/// Human-readable chat reply for a completed run: what was done, and whether the visual
/// evaluation thinks the intent was met.
fn summarize(result: &AiResult) -> String {
    let execution = &result.execution;
    if execution.is_success {
        let mut text = format!("Done: {} ({} of {} steps).", result.plan.title, execution.executed_steps, execution.total_steps);
        if !result.evaluation.intent_satisfied {
            text.push_str(" The result may not fully match the intent yet.");
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
