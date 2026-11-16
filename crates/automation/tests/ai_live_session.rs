//! Offline contract tests. The borrowed Session adapter below is TEST ONLY: it is
//! not a production GUI dispatcher and does not implement desktop modal/job policy.

use std::cell::Cell;

use fotodit_automation::ai::{AiContext, AiEngine, AiError, EditingPlan, ExecutionEngine, IntentCategory, LiveEditSession, PlanStep};
use fotodit_automation::{AutomationError, Headless};
use fotodit_engine::Session;
use serde_json::{Value, json};

#[derive(Default)]
struct MockSession {
    document: Option<Value>,
    calls: Vec<(String, Value)>,
    fail_command: Option<&'static str>,
    inspection_fails: bool,
    inspections: Cell<usize>,
}

impl LiveEditSession for MockSession {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        self.inspections.set(self.inspections.get() + 1);
        if self.inspection_fails {
            // The old substring-based path must not mistake a real error for absence.
            return Err(AutomationError::BadRequest("no document open: injected inspection failure".into()));
        }
        Ok(self.document.clone())
    }

    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
        self.calls.push((command.to_string(), params));
        if self.fail_command == Some(command) {
            return Err(AutomationError::BadRequest("injected command refusal".into()));
        }
        Ok(json!({"command": command}))
    }
}

fn plan(commands: &[&str]) -> EditingPlan {
    EditingPlan {
        title: "Offline boundary test".into(),
        rationale: "Exercise the host dispatcher without a provider".into(),
        category: IntentCategory::PhotoEnhancement,
        steps: commands.iter().map(|command| PlanStep {
            command: (*command).to_string(),
            params: json!({"marker": command}),
            description: (*command).to_string(),
        }).collect(),
        creates_document: false,
        target_size: None,
    }
}

#[test]
fn execution_routes_commands_and_parameters_through_host_trait_object() {
    let mut host = MockSession::default();
    let erased: &mut dyn LiveEditSession = &mut host;
    let result = ExecutionEngine::execute_plan_in(erased, &plan(&["first", "second"])).unwrap();
    assert!(result.is_success);
    assert_eq!(result.executed_steps, 2);
    assert_eq!(host.calls, vec![
        ("first".into(), json!({"marker": "first"})),
        ("second".into(), json!({"marker": "second"})),
    ]);
}

#[test]
fn failed_step_uses_same_host_for_existing_undo_attempts() {
    let mut host = MockSession { fail_command: Some("fail"), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["first", "second", "fail", "never"])).unwrap_err();
    assert!(matches!(error, AiError::ExecutionError { step: 2, .. }));
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["first", "second", "fail", "edit.undo", "edit.undo"]);
    assert_eq!(host.calls[3].1, json!({}));
}

#[test]
fn first_step_refusal_does_not_undo_preexisting_history() {
    let mut host = MockSession { fail_command: Some("fail"), ..Default::default() };
    assert!(ExecutionEngine::execute_plan_in(&mut host, &plan(&["fail"])).is_err());
    assert_eq!(host.calls.len(), 1);
}

#[test]
fn inspection_error_is_not_treated_as_no_document() {
    let ai = AiEngine::new();
    let mut host = MockSession { inspection_fails: true, ..Default::default() };
    let mut context = AiContext::new();
    assert!(matches!(ai.process_instruction_in(&mut host, "Create an elegant wedding invitation", &mut context), Err(AiError::Automation(_))));
    assert!(matches!(ai.plan_instruction_in(&host, "Create an elegant wedding invitation", &context), Err(AiError::Automation(_))));
    assert!(matches!(ai.perceive_document_in(&host), Err(AiError::Automation(_))));
    assert!(host.calls.is_empty());
    assert!(context.prompt_history.is_empty());
}

#[test]
fn no_document_is_explicit_and_planning_never_dispatches() {
    let ai = AiEngine::new();
    let mut host = MockSession::default();
    let mut context = AiContext::new();
    assert!(matches!(ai.perceive_document_in(&host), Err(AiError::NoDocumentOpen)));
    assert!(matches!(ai.process_instruction_in(&mut host, "Remove the background", &mut context), Err(AiError::NoDocumentOpen)));
    let draft = ai.plan_instruction_in(&host, "Create an elegant wedding invitation", &context).unwrap();
    assert!(draft.creates_document);
    assert_eq!(draft.steps[0].command, "file.new");
    assert!(host.calls.is_empty());
    assert!(context.executed_plans.is_empty());
}

#[test]
fn headless_impl_preserves_denied_files_and_null_normalization() {
    let mut headless = Headless::new();
    assert!(LiveEditSession::inspect_active(&headless).unwrap().is_none());
    assert!(LiveEditSession::command_run(&mut headless, "file.open", json!({"path": "forbidden.png"})).is_err());
    assert!(LiveEditSession::command_run(&mut headless, "file.save", json!({"path": "forbidden.plyra"})).is_err());
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    assert!(LiveEditSession::inspect_active(&headless).unwrap().is_some());
    LiveEditSession::command_run(&mut headless, "image.adjustments.invert", json!({})).unwrap();
    LiveEditSession::command_run(&mut headless, "edit.undo", Value::Null).unwrap();
}

/// Test-only adapter borrowing the original session, never cloning or swapping it.
struct BorrowedTestSession<'a> {
    session: &'a mut Session,
}

impl LiveEditSession for BorrowedTestSession<'_> {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        Ok(self.session.active().map(fotodit_engine::inspect::document))
    }

    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
        let params = if params.is_null() { json!({}) } else { params };
        fotodit_automation::workspace::authorize_desktop_engine_command(command, &params)?;
        // Existing Session.authorize remains in place and is checked by engine dispatch.
        self.session.execute(command, params).map_err(AutomationError::from)
    }
}

fn pixel(session: &Session) -> [f32; 4] {
    let active = session.active().unwrap();
    active.doc.layer(active.active_layer.unwrap()).unwrap().surface().unwrap().rgba(0, 0)
}

fn refuse_every_command(_command: &str, _params: &Value) -> fotodit_engine::Result<()> {
    Err(fotodit_engine::EngineError::Other("existing custom gate".into()))
}

#[test]
fn borrowed_adapter_changes_original_document_not_a_second_session_and_supports_undo() {
    let mut original = Session::new();
    original.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let mut untouched = Session::new();
    untouched.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let before = fotodit_engine::inspect::document(original.active().unwrap());
    let other_before = fotodit_engine::inspect::document(untouched.active().unwrap());
    let original_id = original.active().unwrap().doc.id;
    let original_pixel = pixel(&original);
    {
        let mut host = BorrowedTestSession { session: &mut original };
        let plan = EditingPlan {
            title: "Invert".into(), rationale: "Offline command routing".into(),
            category: IntentCategory::PhotoEnhancement,
            steps: vec![PlanStep {
                command: "image.adjustments.invert".into(), params: json!({}), description: "Invert active document".into(),
            }], creates_document: false, target_size: None,
        };
        ExecutionEngine::execute_plan_in(&mut host, &plan).unwrap();
        let after = host.inspect_active().unwrap().unwrap();
        assert_ne!(after["revision"], before["revision"]);
        assert_ne!(pixel(host.session), original_pixel);
        host.command_run("edit.undo", json!({})).unwrap();
    }
    assert!(original.active().unwrap().history.can_redo());
    assert_eq!(pixel(&original), original_pixel);
    assert_eq!(original.active().unwrap().doc.id, original_id);
    assert_eq!(fotodit_engine::inspect::document(untouched.active().unwrap()), other_before);
    assert_eq!(original.active().unwrap().doc.layers.len(), before["layers"].as_array().unwrap().len());
}

#[test]
fn borrowed_adapter_keeps_existing_custom_authorization_hook() {
    let mut original = Session::new();
    original.authorize = Some(refuse_every_command);
    let mut host = BorrowedTestSession { session: &mut original };
    let error = host.command_run("file.new", json!({"width": 8, "height": 8})).unwrap_err();
    assert!(error.to_string().contains("existing custom gate"));
    assert!(host.inspect_active().unwrap().is_none());
    assert!(host.session.authorize.is_some());
}

#[test]
fn existing_headless_api_remains_function_pointer_compatible() {
    let _process: fn(&AiEngine, &mut Headless, &str, &mut AiContext) -> Result<fotodit_automation::AiResult, AiError> = AiEngine::process_instruction;
    let _plan: fn(&AiEngine, &Headless, &str, &AiContext) -> Result<EditingPlan, AiError> = AiEngine::plan_instruction;
    let _perceive: fn(&AiEngine, &Headless) -> Result<fotodit_automation::ai::VisualScene, AiError> = AiEngine::perceive_document;
    let _execute: fn(&mut Headless, &EditingPlan) -> Result<fotodit_automation::ai::ExecutionResult, AiError> = ExecutionEngine::execute_plan;
}

#[test]
fn missing_document_after_execution_is_reported_without_recording_success() {
    let ai = AiEngine::new();
    // This mock accepts commands but deliberately never creates a document.
    let mut host = MockSession::default();
    let mut context = AiContext::new();
    let error = ai.process_instruction_in(&mut host, "Create an elegant wedding invitation", &mut context).unwrap_err();
    assert!(matches!(error, AiError::NoDocumentOpen));
    assert!(!host.calls.is_empty());
    assert_eq!(host.calls[0].0, "file.new");
    assert_eq!(host.inspections.get(), 2);
    assert!(context.prompt_history.is_empty());
}

#[test]
fn headless_wrapper_and_generic_planning_agree() {
    let ai = AiEngine::new();
    let headless = Headless::new();
    let context = AiContext::new();
    let prompt = "Create an elegant wedding invitation";
    let legacy = ai.plan_instruction(&headless, prompt, &context).unwrap();
    let generic = ai.plan_instruction_in(&headless, prompt, &context).unwrap();
    assert_eq!(legacy, generic);
    assert!(headless.session.active().is_none());
}

#[test]
fn successful_generic_process_inspects_host_twice_and_records_context() {
    let mut fixture = Session::new();
    fixture.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let document = fotodit_engine::inspect::document(fixture.active().unwrap());
    let mut host = MockSession { document: Some(document), ..Default::default() };
    let ai = AiEngine::new();
    let mut context = AiContext::new();
    let prompt = "Make this photograph look professional and enhance clarity";
    let result = ai.process_instruction_in(&mut host, prompt, &mut context).unwrap();
    assert!(result.execution.is_success);
    assert_eq!(host.inspections.get(), 2);
    assert_eq!(host.calls.len(), result.plan.steps.len());
    assert_eq!(context.prompt_history, vec![prompt.to_string()]);
    assert_eq!(context.executed_plans, vec![result.plan]);
}
