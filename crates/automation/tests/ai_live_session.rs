//! Offline contract tests. The borrowed Session adapter below is TEST ONLY: it is
//! not a production GUI dispatcher and does not implement desktop modal/job policy.

use std::cell::Cell;

use fotodit_automation::ai::{AiContext, AiEngine, AiError, EditingPlan, ExecutionEngine, IntentCategory, LiveEditSession, PlanStep, computed_dominant_colors};
use fotodit_automation::{AutomationError, Headless};
use fotodit_engine::Session;
use serde_json::{Value, json};

#[derive(Default)]
struct MockSession {
    document: Option<Value>,
    calls: Vec<(String, Value)>,
    fail_command: Option<&'static str>,
    /// The failing command journals an entry before erroring (mutate-then-Err).
    mutate_before_fail: bool,
    /// After the first successful command the document identity changes, as a
    /// document-switching side effect would produce.
    identity_sabotage: bool,
    /// file.new actually creates a document (the default mock deliberately
    /// never creates one, preserving the no-document coverage).
    file_new_creates: bool,
    /// edit.undo reports success but removes nothing.
    noop_undo: bool,
    /// A journaling command replaces the history list with foreign entries
    /// instead of appending.
    replace_history: bool,
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
            // mutate-then-Err: the failure still leaves its journaled entry.
            if self.mutate_before_fail {
                self.push_history(command);
            }
            return Err(AutomationError::BadRequest("injected command refusal".into()));
        }
        // Simulate the engine's history contract using its real registry: journaling
        // commands append one History panel entry; queries do not; edit.undo removes
        // the latest undoable entry and is refused when only the current
        // (non-undoable) state remains.
        if command == "file.new" && self.file_new_creates {
            self.document = Some(mock_document("Untitled-1", &["New"]));
        } else if command == "edit.undo" {
            let history = self.document.as_mut().and_then(|doc| doc.get_mut("history")).and_then(Value::as_array_mut);
            match history {
                _ if self.noop_undo => {}
                Some(h) if h.len() > 1 => {
                    h.pop();
                }
                _ => return Err(AutomationError::BadRequest("nothing to undo".into())),
            }
        } else if self.replace_history {
            if let Some(h) = self.document.as_mut().and_then(|doc| doc.get_mut("history")).and_then(Value::as_array_mut) {
                *h = vec![json!("OtherUserEdit"), json!("ForeignEdit")];
            }
        } else if fotodit_engine::command_specs().iter().any(|s| s.id == command && s.journal) {
            self.push_history(command);
        }
        if self.identity_sabotage {
            self.document = Some(mock_document("sabotaged.plyra", &["Open", "Stroke", "Fill"]));
        }
        Ok(json!({"command": command}))
    }
}

impl MockSession {
    fn push_history(&mut self, label: &str) {
        if let Some(h) = self.document.as_mut().and_then(|doc| doc.get_mut("history")).and_then(Value::as_array_mut) {
            h.push(json!(label));
        }
    }
}

fn mock_document(name: &str, history: &[&str]) -> Value {
    json!({"name": name, "path": null, "revision": history.len() as u64, "history": history})
}

fn plan(commands: &[&str]) -> EditingPlan {
    EditingPlan {
        title: "Offline boundary test".into(),
        rationale: "Exercise the host dispatcher without a provider".into(),
        category: IntentCategory::GeneralEditing,
        steps: commands
            .iter()
            .map(|command| PlanStep { command: (*command).to_string(), params: json!({"marker": command}), description: (*command).to_string() })
            .collect(),
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
    assert_eq!(host.calls, vec![("first".into(), json!({"marker": "first"})), ("second".into(), json!({"marker": "second"})),]);
}

#[test]
fn failed_step_uses_same_host_for_existing_undo_attempts() {
    let mut host = MockSession { fail_command: Some("fail"), document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "image.adjustments.invert", "fail", "never"])).unwrap_err();
    let AiError::ExecutionError { step, error, .. } = &error else { panic!("execution error") };
    assert_eq!(*step, 2);
    assert!(error.contains("history boundary is verified at its pre-plan entries"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "image.adjustments.invert", "fail", "edit.undo", "edit.undo"]);
    assert_eq!(host.calls[3].1, json!({}));
    // Both plan entries are gone; the pre-existing "New" entry was never touched.
    let history = host.document.unwrap()["history"].as_array().unwrap().clone();
    assert_eq!(history, vec![json!("New")]);
}

#[test]
fn failed_step_entry_is_reconciled_and_undone_first() {
    // mutate-then-Err: the failed command journaled an entry before erroring.
    // Rollback must account that entry to the failed step and undo it first.
    let mut host = MockSession { fail_command: Some("stroke"), mutate_before_fail: true, document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "stroke"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rolled back 2 entr"), "{error}");
    assert!(error.contains("history boundary is verified at its pre-plan entries"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "stroke", "edit.undo", "edit.undo"]);
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("New")]);
}

#[test]
fn plans_changing_document_identity_are_refused_before_execution() {
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "file.close"])).unwrap_err();
    let AiError::ExecutionError { step, error, .. } = &error else { panic!("execution error") };
    assert_eq!(*step, 1);
    assert!(error.contains("changes which document is active"), "{error}");
    assert!(host.calls.is_empty(), "nothing may execute before the refusal");
}

#[test]
fn creation_plan_rolls_back_inside_the_created_document() {
    let mut host = MockSession { fail_command: Some("fail"), file_new_creates: true, ..Default::default() };
    let mut plan = plan(&["file.new", "image.adjustments.invert", "fail"]);
    plan.creates_document = true;
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    // The created document's edits are undone, but the run is honestly
    // incomplete relative to the original state: the document stays open.
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("plan-created document remains open"), "{error}");
    assert!(error.contains("rolled back 1 of 1"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["file.new", "image.adjustments.invert", "fail", "edit.undo"]);
    // The created document remains open (creation is not a history entry) with
    // only its own initial entry.
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("New")]);
}

#[test]
fn rollback_fails_closed_when_history_prefix_is_replaced() {
    // Prefix attack: the step keeps name/path and grows the list by one, but the
    // prior entries were REPLACED by foreign ones. Not one foreign entry may be
    // undone.
    let mut host = MockSession { fail_command: Some("fail"), replace_history: true, document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("rolled back 0"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail"]);
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("OtherUserEdit"), json!("ForeignEdit")]);
}

#[test]
fn noop_undo_is_reported_as_zero_verified_undos() {
    let mut host = MockSession { fail_command: Some("fail"), noop_undo: true, document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("rolled back 0 of 1"), "{error}");
    assert!(error.contains("undo verification failed"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail", "edit.undo"]);
    // The plan's entry remains, reported rather than silently consumed.
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("New"), json!("image.adjustments.invert")]);
}

#[test]
fn malformed_inspection_metadata_closes_rollback() {
    // history missing from the inspection JSON: unreadable, never assumed empty.
    let mut host = MockSession { fail_command: Some("fail"), document: Some(json!({"name": "mock.plyra"})), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("could not be inspected"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail"]);
}

#[test]
fn rollback_fails_closed_when_active_document_changes() {
    let mut host = MockSession { fail_command: Some("fail"), identity_sabotage: true, document: Some(mock_document("first.plyra", &["New"])), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("changed the active document"), "{error}");
    // Fail-closed: no blind undo is issued.
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail"]);
}

#[test]
fn rollback_stops_safely_when_inspection_is_unreadable() {
    // No document JSON is readable: the steps' effect on history cannot be
    // verified, so rollback must undo nothing and say so, never undo blindly.
    let mut host = MockSession { fail_command: Some("fail"), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("rollback incomplete"), "{error}");
    assert!(error.contains("could not be inspected"), "{error}");
    assert!(error.contains("rolled back 0"), "{error}");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "image.adjustments.invert", "fail"]);
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
            title: "Invert".into(),
            rationale: "Offline command routing".into(),
            category: IntentCategory::GeneralEditing,
            steps: vec![PlanStep { command: "image.adjustments.invert".into(), params: json!({}), description: "Invert active document".into() }],
            creates_document: false,
            target_size: None,
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
    // Legitimate delta: rollback safety reads the document mark once before the
    // plan and once after each executed step (2 + steps + 1).
    assert_eq!(host.inspections.get(), 7);
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
fn successful_enhancement_process_rechecks_host_before_dispatch_and_records_context() {
    let mut fixture = Session::new();
    fixture.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let document = fotodit_engine::inspect::document(fixture.active().unwrap());
    let mut host = MockSession { document: Some(document), ..Default::default() };
    let ai = AiEngine::new();
    let mut context = AiContext::new();
    let prompt = "Make this photograph look professional and enhance clarity";
    let result = ai.process_instruction_in(&mut host, prompt, &mut context).unwrap();
    assert!(result.execution.is_success);
    // Legitimate delta: rollback safety reads the document mark once before the
    // plan and once after each executed step (3 + steps + 1).
    assert_eq!(host.inspections.get(), 7);
    // Color stats are computed from the host before planning and after execution; the
    // remaining calls are exactly the plan steps.
    let stats_calls = host.calls.iter().filter(|(id, _)| id == "image.analysis.colorStats").count();
    assert_eq!(stats_calls, 2);
    assert_eq!(host.calls.len() - stats_calls, result.plan.steps.len());
    assert_eq!(context.prompt_history, vec![prompt.to_string()]);
    assert_eq!(context.executed_plans, vec![result.plan]);
}

#[test]
fn computed_dominant_colors_survives_unsupported_or_failing_stats_command() {
    let mut fixture = Session::new();
    fixture.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let document = fotodit_engine::inspect::document(fixture.active().unwrap());
    // A host without image.analysis.colorStats (or one that refuses it) still processes
    // the instruction: colors stay honestly empty instead of failing or being invented.
    let mut host = MockSession { document: Some(document), fail_command: Some("image.analysis.colorStats"), ..Default::default() };
    let ai = AiEngine::new();
    let mut context = AiContext::new();
    let result = ai.process_instruction_in(&mut host, "Make this photograph look professional and enhance clarity", &mut context).unwrap();
    assert!(result.execution.is_success);
    assert!(result.initial_scene.dominant_colors.is_empty());
}

#[test]
fn computed_dominant_colors_uses_engine_stats_and_stays_honest() {
    let mut headless = Headless::new();
    // No document open: the command cannot run, and the result stays empty
    // rather than carrying invented colors.
    assert!(computed_dominant_colors(&mut headless).is_empty());
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    // Real engine pixels: a white canvas computes to a single white dominant.
    assert_eq!(computed_dominant_colors(&mut headless), vec!["#FFFFFF".to_string()]);
}

#[test]
fn rollback_undoes_only_journaling_plan_steps_not_preexisting_history() {
    let mut headless = Headless::new();
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    // A pre-existing user edit that no plan rollback may consume.
    LiveEditSession::command_run(&mut headless, "image.adjustments.invert", json!({})).unwrap();
    assert_eq!(computed_dominant_colors(&mut headless), vec!["#000000".to_string()]);

    let plan = EditingPlan {
        title: "Mixed journaling plan".into(),
        rationale: "A non-journaling analysis step precedes the failing step".into(),
        category: IntentCategory::GeneralEditing,
        steps: vec![
            PlanStep { command: "image.analysis.colorStats".into(), params: json!({"maxColors": 5, "maxPixels": 4_000_000}), description: "analysis only".into() },
            PlanStep { command: "image.adjustments.invert".into(), params: json!({}), description: "plan edit".into() },
            PlanStep { command: "test.nonexistent.command".into(), params: json!({}), description: "fails".into() },
        ],
        creates_document: false,
        target_size: None,
    };
    let error = ExecutionEngine::execute_plan_in(&mut headless, &plan).unwrap_err();
    assert!(matches!(error, AiError::ExecutionError { step: 2, .. }));
    // Only the plan's own journaling edit may be rolled back: the pre-existing
    // inversion (black canvas) survives because the analysis step journaled nothing.
    assert_eq!(computed_dominant_colors(&mut headless), vec!["#000000".to_string()]);
}

#[test]
fn save_reopen_roundtrip_preserves_pixels_through_authorized_session_path() {
    // Durability through the authorized Headless save/open path (the admitted
    // command path refuses file.save/file.open; nothing here weakens that).
    // trusted_local: the authorized test-session filesystem (Headless::new has
    // no filesystem grant; the admitted command path's refusal is untouched).
    let mut headless = Headless::trusted_local();
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    LiveEditSession::command_run(&mut headless, "image.adjustments.invert", json!({})).unwrap();
    assert_eq!(computed_dominant_colors(&mut headless), vec!["#000000".to_string()]);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let path = std::env::temp_dir().join(format!("fotodit_ai_roundtrip_{}_{}.plyra", std::process::id(), nanos));
    headless.save(None, Some(&path), Some("plyra"), &Default::default()).unwrap();
    headless.open(&path).unwrap();
    // The reopened document is the active one and keeps the edited pixels.
    assert_eq!(computed_dominant_colors(&mut headless), vec!["#000000".to_string()]);
    let _ = std::fs::remove_file(&path);
}
