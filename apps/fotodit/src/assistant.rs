//! Chat assistant bridge: composer prompts run through `fotodit_automation`'s AI
//! pipeline (deterministic offline planner by default) against the app's admitted
//! command dispatch, per the `LiveEditSession` host contract: the app's real session is
//! borrowed for the duration of one operation, commands go through the admitted menu
//! path (authorization and discard interception preserved), and results are completed
//! values, never pending job receipts.

use fotodit_automation::AutomationError;
use fotodit_automation::ai::{AiContext, AiEngine, AiError, AiResult, LiveEditSession};
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
    pub fn new() -> Self {
        Self { engine: AiEngine::new(), context: AiContext::new() }
    }
}

impl AssistantBackend for GuiAssistant {
    fn reply(&mut self, app: &mut FotoditApp, ctx: &egui::Context, prompt: &str) -> Result<String, String> {
        let mut session = GuiSession { app, ctx };
        match self.engine.process_instruction_in(&mut session, prompt, &mut self.context) {
            Ok(result) => Ok(summarize(&result)),
            Err(AiError::NoDocumentOpen) => Err("Open a document first, or ask me to create one.".to_string()),
            Err(error) => Err(format!("I couldn't complete that edit: {error}")),
        }
    }
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
