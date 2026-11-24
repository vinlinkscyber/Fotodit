//! Offline contract tests. The borrowed Session adapter below is TEST ONLY: it is
//! not a production GUI dispatcher and does not implement desktop modal/job policy.

use std::cell::Cell;

use fotodit_automation::ai::{AiContext, AiEngine, AiError, EditingPlan, ExecutionEngine, IntentCategory, LiveEditSession, PlanStep, VisualEvidenceClass, computed_dominant_colors};
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
    /// Scripted digest captures for a host WITH pixel capability: each
    /// `visual_digest` call pops the front value. An empty script means the
    /// host offers no capability (the trait default is exercised instead).
    digest_script: std::collections::VecDeque<Result<Value, String>>,
    digest_calls: Cell<usize>,
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

    fn visual_digest(&mut self, _max_pixels: u64, _region: Option<[i64; 4]>) -> Option<Result<Value, String>> {
        if self.digest_script.is_empty() {
            // No-capability host: the trait default, exercised on purpose.
            return None;
        }
        self.digest_calls.set(self.digest_calls.get() + 1);
        Some(self.digest_script.pop_front().expect("test digest script exhausted"))
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

/// A well-formed `image.analysis.pixelDigest` output for the scripted host:
/// the full fixed schema with a grid-consistent sample count.
fn digest_json(tag: u64, width: u64, height: u64, stride: u64) -> Value {
    let sampled = ((width - 1) / stride + 1) * ((height - 1) / stride + 1);
    json!({"schema": 1, "hash": "fnv1a64", "quantization": "8bit/channel", "digest": format!("{tag:016x}"), "domain": "canvas", "width": width, "height": height, "stride": stride, "sampledPixels": sampled, "revision": 1})
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

    fn visual_digest(&mut self, max_pixels: u64, _region: Option<[i64; 4]>) -> Option<Result<Value, String>> {
        // Source-positive admitted path: exactly the same authorization gates as
        // any other command, including a custom Session.authorize hook.
        Some(self.command_run("image.analysis.pixelDigest", json!({"maxPixels": max_pixels})).map_err(|e| e.to_string()))
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


// ------------------------------------------------------------------ visual result evidence
//
// Evidence is collected through the additive `execute_plan_*_with_evidence`
// API: `StepExecution` stays byte-frozen and evidence never gates success,
// rollback or dispatch. These tests focus on schema, domain, capture count,
// denial, budget, position and quantization honesty.

/// Build a single-step plan with real-engine-compatible empty params.
fn one_step_plan(command: &str) -> EditingPlan {
    EditingPlan {
        title: "Evidence test".into(),
        rationale: "Observed-composite evidence".into(),
        category: IntentCategory::GeneralEditing,
        steps: vec![PlanStep { command: command.to_string(), params: json!({}), description: command.to_string() }],
        creates_document: false,
        target_size: None,
    }
}

#[test]
fn visual_evidence_unknown_when_host_has_no_digest_capability() {
    // Default trait behavior: a host without pixel access yields honest Unknown,
    // never invented change, and no phantom commands reach the editing path.
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps.len(), 1);
    assert_eq!(evidence.steps[0].step_index, 0);
    let visual = &evidence.steps[0].visual;
    assert_eq!(visual.class, VisualEvidenceClass::Unknown);
    assert!(visual.detail.contains("no pixel digest capability"), "{}", visual.detail);
    assert_eq!(host.digest_calls.get(), 0);
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert"]);
    // The pre-existing API is a thin wrapper returning the identical execution.
    let mut host2 = MockSession { document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let result2 = ExecutionEngine::execute_plan_in(&mut host2, &plan(&["image.adjustments.invert"])).unwrap();
    assert_eq!(result2, result);
}

#[test]
fn nonvisual_and_unclassified_commands_are_honestly_classified() {
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.analysis.colorStats", "first"])).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::NotApplicable);
    let unclassified = &evidence.steps[1].visual;
    assert_eq!(unclassified.class, VisualEvidenceClass::Unknown);
    assert!(unclassified.detail.contains("not in the visual-evidence classification table"), "{}", unclassified.detail);
    assert_eq!(host.digest_calls.get(), 0);
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.analysis.colorStats", "first"]);
}

#[test]
fn invented_id_under_a_visual_prefix_is_unclassified_not_visual() {
    // Registry grounding: image.adjustments.* is visual only when the id is a
    // registered journaling spec. An invented id under the prefix must not
    // inherit the prefix's admission: unknown-safe, no expansion.
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.notARealAdjustment"])).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::Unknown);
    assert!(evidence.steps[0].visual.detail.contains("not in the visual-evidence classification table"), "{}", evidence.steps[0].visual.detail);
    assert_eq!(host.digest_calls.get(), 0);
}

#[test]
fn digest_marks_classify_changed_and_no_change_observed() {
    // Chained captures: step 1's after-mark is step 2's baseline (k+1 captures
    // for k visual steps). Equal digests over an identical grid are reported as
    // no-change-observed (a legitimate idempotent outcome), never as failure.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert", "image.adjustments.invert"])).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::Changed);
    let noop = &evidence.steps[1].visual;
    assert_eq!(noop.class, VisualEvidenceClass::NoChangeObserved);
    assert!(noop.detail.contains("sampling, quantization and hash collisions can hide changes"), "{}", noop.detail);
    assert_eq!(host.digest_calls.get(), 3);
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "image.adjustments.invert"]);
}

#[test]
fn geometry_change_is_changed_not_a_digest_mismatch() {
    // A dimension change is an observable change of its own; digests over
    // different dimensions are never compared.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(9, 16, 16, 1))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (_result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    let visual = &evidence.steps[0].visual;
    assert_eq!(visual.class, VisualEvidenceClass::Changed);
    assert!(visual.detail.contains("the document dimensions changed (8x8 -> 16x16)"), "{}", visual.detail);
}

#[test]
fn stride_change_at_identical_dimensions_is_unknown_incomparable() {
    // Same dimensions, different stride: the sampling grid changed, so the
    // digests are incomparable. This is NOT a pixel change - honest Unknown.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(1, 8, 8, 2))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (_result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    let visual = &evidence.steps[0].visual;
    assert_eq!(visual.class, VisualEvidenceClass::Unknown);
    assert!(visual.detail.contains("the sampling grid changed (stride 1 -> 2)"), "{}", visual.detail);
}

#[test]
fn malformed_digest_outputs_fail_schema_validation() {
    // The host's JSON is unauthenticated: every fixed-schema violation -
    // domain, version, hash, quantization, digest syntax, stride/dims, and
    // sample-count grid consistency - is rejected, never trusted.
    let cases: Vec<Value> = [
        ("domain", json!("layer")),
        ("schema", json!(2)),
        ("hash", json!("sha256")),
        ("quantization", json!("16bit/channel")),
        ("digest", json!("xyz")),
        ("digest", json!("00000000000000001")),
        ("stride", json!(0)),
        ("width", json!(0)),
        ("height", json!(0)),
        ("sampledPixels", json!(63)),
    ]
    .into_iter()
    .map(|(key, value)| {
        let mut bad = digest_json(1, 8, 8, 1);
        bad[key] = value;
        bad
    })
    .chain([
        {
            let mut missing = digest_json(1, 8, 8, 1);
            missing.as_object_mut().expect("object").remove("digest");
            missing
        },
        // Dimensions whose area overflows u64.
        json!({"schema": 1, "hash": "fnv1a64", "quantization": "8bit/channel", "digest": "0000000000000001", "domain": "canvas", "width": u64::MAX, "height": 2, "stride": 1, "sampledPixels": 64}),
        // Grid-consistent but over the requested capture budget.
        json!({"schema": 1, "hash": "fnv1a64", "quantization": "8bit/channel", "digest": "0000000000000001", "domain": "canvas", "width": 5000, "height": 5000, "stride": 1, "sampledPixels": 25_000_000u64}),
    ])
    .collect();
    for bad in cases {
        let script = [Ok(bad.clone()), Ok(digest_json(2, 8, 8, 1))];
        let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
        let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
        assert!(result.is_success);
        let visual = &evidence.steps[0].visual;
        assert_eq!(visual.class, VisualEvidenceClass::Unknown, "{bad}");
        assert!(visual.detail.contains("failed schema validation"), "{bad}: {}", visual.detail);
    }
}

#[test]
fn digest_hex_is_compared_numerically_not_textually() {
    // "AB" and "ab" are the same number: a case-only difference must never
    // register as a change.
    let mut upper = digest_json(0xAB, 8, 8, 1);
    upper["digest"] = json!("00000000000000AB");
    let script = [Ok(upper), Ok(digest_json(0xAB, 8, 8, 1))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (_result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::NoChangeObserved, "{}", evidence.steps[0].visual.detail);
}

#[test]
fn execute_plan_in_stays_render_free_for_existing_callers() {
    // Strict opt-in: the legacy entry point makes zero digest captures even
    // when the host offers the capability.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let result = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    assert!(result.is_success);
    assert_eq!(host.digest_calls.get(), 0);
}

#[test]
fn capture_failure_is_unknown_and_not_masked() {
    // The first failure is the honest reason; a later unavailable capture must
    // not mask it, and evidence never affects the step's success.
    let script = [Err("refused by host policy".to_string())];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert"])).unwrap();
    assert!(result.is_success);
    let visual = &evidence.steps[0].visual;
    assert_eq!(visual.class, VisualEvidenceClass::Unknown);
    assert!(visual.detail.contains("the baseline digest capture failed: refused by host policy"), "{}", visual.detail);
}

#[test]
fn capture_budget_bounds_aggregate_render_cost() {
    // 8 visual steps chain 9 captures; the budget is 8, so the last step's
    // after-capture is refused with honest Unknown instead of unbounded cost.
    let script: Vec<Result<Value, String>> = (1..=8u64).map(|tag| Ok(digest_json(tag, 8, 8, 1))).collect();
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let commands = ["image.adjustments.invert"; 8];
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&commands)).unwrap();
    assert!(result.is_success);
    for step in &evidence.steps[..7] {
        assert_eq!(step.visual.class, VisualEvidenceClass::Changed, "step {}: {}", step.step_index, step.visual.detail);
    }
    let last = &evidence.steps[7].visual;
    assert_eq!(last.class, VisualEvidenceClass::Unknown);
    assert!(last.detail.contains("capture budget (8) is exhausted"), "{}", last.detail);
    assert_eq!(host.digest_calls.get(), 8);
}

#[test]
fn failed_step_drops_the_bundle_and_rollback_is_unchanged() {
    // Legacy entry point (evidence OFF): failure handling must be identical to
    // the pre-evidence behavior - no captures, same rollback. The opt-in
    // failure contract is covered separately below.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1))];
    let mut host = MockSession { fail_command: Some("fail"), document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("history boundary is verified at its pre-plan entries"), "{error}");
    assert_eq!(host.digest_calls.get(), 0, "legacy path must not capture");
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail", "edit.undo"]);
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("New")]);
}

#[test]
fn failed_step_opt_in_captures_pre_failure_then_drops_bundle() {
    // Same failure through the OPT-IN API: the completed visual step is
    // captured (before + after) BEFORE the failing step runs, the partial
    // bundle is dropped with the error, and rollback matches the legacy path.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1))];
    let mut host = MockSession { fail_command: Some("fail"), document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let error = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert", "fail"])).unwrap_err();
    let AiError::ExecutionError { error, .. } = &error else { panic!("execution error") };
    assert!(error.contains("history boundary is verified at its pre-plan entries"), "{error}");
    // Captures for the completed visual step happened before the failure.
    assert_eq!(host.digest_calls.get(), 2);
    assert_eq!(host.calls.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["image.adjustments.invert", "fail", "edit.undo"]);
    assert_eq!(host.document.unwrap()["history"].as_array().unwrap().clone(), vec![json!("New")]);
}

#[test]
fn unclassified_step_breaks_the_comparison_chain() {
    // An unobserved middle step means the next visual step re-baselines instead
    // of attributing the middle step's possible changes to itself.
    let script = [Ok(digest_json(1, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1)), Ok(digest_json(2, 8, 8, 1)), Ok(digest_json(3, 8, 8, 1))];
    let mut host = MockSession { document: Some(mock_document("mock.plyra", &["New"])), digest_script: script.into_iter().collect(), ..Default::default() };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan(&["image.adjustments.invert", "mystery", "image.adjustments.invert"])).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::Changed);
    assert_eq!(evidence.steps[1].visual.class, VisualEvidenceClass::Unknown);
    // The third step re-baselined (tag 2) and observed its own change (2 -> 3).
    assert_eq!(evidence.steps[2].visual.class, VisualEvidenceClass::Changed);
    assert_eq!(host.digest_calls.get(), 4);
}

fn refuse_digest_only(command: &str, _params: &Value) -> fotodit_engine::Result<()> {
    if command == "image.analysis.pixelDigest" {
        Err(fotodit_engine::EngineError::Other("custom gate denies digest".into()))
    } else {
        Ok(())
    }
}

#[test]
fn denied_digest_hook_is_neither_bypassed_nor_fails_editing() {
    // A host whose custom authorization hook refuses the digest command: the
    // refusal surfaces as honest unknown evidence, the editing step still
    // succeeds through the admitted path, and no side channel bypasses the
    // gate or reroutes the editing command.
    let mut original = Session::new();
    original.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    original.authorize = Some(refuse_digest_only);
    let before = pixel(&original);
    let mut host = BorrowedTestSession { session: &mut original };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut host, &one_step_plan("image.adjustments.invert")).unwrap();
    assert!(result.is_success);
    assert_eq!(evidence.steps[0].visual.class, VisualEvidenceClass::Unknown);
    assert!(evidence.steps[0].visual.detail.contains("custom gate denies digest"), "{}", evidence.steps[0].visual.detail);
    // Editing was neither failed nor rerouted by the denied evidence capture,
    // and the custom gate itself is untouched.
    assert_ne!(pixel(host.session), before);
    assert!(host.session.authorize.is_some());
}

#[test]
fn process_instruction_in_stays_render_free_and_evidence_is_opt_in() {
    // The host OFFERS digest capability; the legacy AiEngine entry point must
    // not use it, and the additive entry point must actually collect.
    let mut fixture = Session::new();
    fixture.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let document = fotodit_engine::inspect::document(fixture.active().unwrap());
    let script: Vec<Result<Value, String>> = (1..=16u64).map(|tag| Ok(digest_json(tag, 8, 8, 1))).collect();
    let mut host = MockSession { document: Some(document.clone()), digest_script: script.into_iter().collect(), ..Default::default() };
    let ai = AiEngine::new();
    let mut context = AiContext::new();
    let prompt = "Make this photograph look professional and enhance clarity";
    let result = ai.process_instruction_in(&mut host, prompt, &mut context).unwrap();
    assert!(result.execution.is_success);
    assert_eq!(host.digest_calls.get(), 0, "the legacy entry point must not capture evidence");

    let script2: Vec<Result<Value, String>> = (1..=16u64).map(|tag| Ok(digest_json(tag, 8, 8, 1))).collect();
    let mut host2 = MockSession { document: Some(document), digest_script: script2.into_iter().collect(), ..Default::default() };
    let mut context2 = AiContext::new();
    let (result2, evidence) = ai.process_instruction_in_with_evidence(&mut host2, prompt, &mut context2).unwrap();
    assert!(result2.execution.is_success);
    assert_eq!(evidence.steps.len(), result2.execution.step_executions.len());
    assert!(host2.digest_calls.get() >= 1, "the opt-in path actually captures");
}

#[test]
fn headless_visual_evidence_reflects_real_pixel_changes() {
    // End to end through the admitted Headless path: the real engine digest
    // observes a real inversion, and repeat captures of an unchanged document
    // are deterministic (no-op renders compare equal).
    let mut headless = Headless::new();
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let (result, evidence) = ExecutionEngine::execute_plan_with_evidence(&mut headless, &one_step_plan("image.adjustments.invert")).unwrap();
    assert!(result.is_success);
    let visual = &evidence.steps[0].visual;
    assert_eq!(visual.class, VisualEvidenceClass::Changed, "{}", visual.detail);
    let first = LiveEditSession::visual_digest(&mut headless, 4_000_000, None).unwrap().unwrap();
    let second = LiveEditSession::visual_digest(&mut headless, 4_000_000, None).unwrap().unwrap();
    assert_eq!(first["digest"], second["digest"]);
    assert_eq!(first["schema"], json!(1));
    assert_eq!(first["hash"], json!("fnv1a64"));
    assert_eq!(first["quantization"], json!("8bit/channel"));
    assert_eq!(first["width"], json!(8));
    assert_eq!(first["stride"], json!(1));
    assert_eq!(first["sampledPixels"], json!(64));
}

fn paint(session: &mut Session, x0: i32, y0: i32, x1: i32, y1: i32, rgba: [f32; 4]) {
    session
        .edit("paint", move |doc, _| {
            let surf = doc.layers[0].surface_mut().unwrap();
            let px = fotodit_raster::from_rgba(&surf.format(), rgba);
            surf.fill_rect(fotodit_geom::Rect::new(x0, y0, x1, y1), &px);
            Ok(())
        })
        .unwrap();
}

fn real_digest(session: &mut Session) -> String {
    let mut host = BorrowedTestSession { session };
    let value = host.visual_digest(4_000_000, None).unwrap().unwrap();
    value["digest"].as_str().unwrap().to_string()
}

#[test]
fn real_digest_is_position_sensitive_and_deterministic() {
    // Two composites with identical histograms (one red pixel each, at
    // different positions) must digest differently; an unchanged document
    // re-captures identically.
    let mut original = Session::new();
    original.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let plain = real_digest(&mut original);
    assert_eq!(real_digest(&mut original), plain, "repeat captures of an unchanged document are deterministic");
    paint(&mut original, 0, 0, 1, 1, [1.0, 0.0, 0.0, 1.0]);
    let at_origin = real_digest(&mut original);
    assert_ne!(at_origin, plain);
    paint(&mut original, 0, 0, 1, 1, [1.0, 1.0, 1.0, 1.0]);
    paint(&mut original, 1, 0, 2, 1, [1.0, 0.0, 0.0, 1.0]);
    let shifted = real_digest(&mut original);
    assert_ne!(shifted, at_origin, "identical histogram, different position: digests must differ");
    assert_ne!(shifted, plain);
}

#[test]
fn real_digest_quantization_hides_hdr_and_sub_byte_differences() {
    // Documented claim boundary: the digest clamps to 8 bits per channel, so
    // HDR values above 1.0 are indistinguishable from 1.0 - an equal digest
    // is not proof of no change.
    let mut original = Session::new();
    original.execute("file.new", json!({"width": 8, "height": 8, "depth": 32, "background": "white"})).unwrap();
    let plain = real_digest(&mut original);
    paint(&mut original, 0, 0, 1, 1, [2.0, 2.0, 2.0, 1.0]);
    let hdr = real_digest(&mut original);
    assert_eq!(hdr, plain, "2.0 clamps to the same byte as 1.0: the HDR difference is invisible to the digest");
    paint(&mut original, 0, 0, 1, 1, [3.0, 3.0, 3.0, 1.0]);
    assert_eq!(real_digest(&mut original), plain, "3.0 clamps identically: still invisible");
    // A visible-scale difference IS observed on the same document.
    paint(&mut original, 0, 0, 1, 1, [0.0, 0.0, 0.0, 1.0]);
    assert_ne!(real_digest(&mut original), plain);
}

/// REAL-ENGINE FIXTURE (authored now; runs only under an approved CI run):
/// a plan that selects before editing yields a target-selection capture
/// taken IMMEDIATELY BEFORE the edit whose effect is evaluated - document
/// identity, bounds, the selection's exact per-pixel coverage, the exact
/// before-pixels, the region digest, protected-complement state and
/// coverage-aware colour statistics - binding the measured criteria to the
/// plan's actual target, not to any pre-existing selection, with the exact
/// coverage preserved through the selection's consumption.
#[test]
fn selection_stage_capture_binds_the_plans_target_before_the_edit() {
    let mut headless = Headless::new();
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 64, "height": 64, "background": "white"})).unwrap();
    // A pre-existing selection is deliberately absent: the plan's own
    // selection step is what defines the target.
    let plan = EditingPlan {
        title: "Invert the selection".into(),
        rationale: "select, then edit".into(),
        category: IntentCategory::ColorStyling,
        steps: vec![
            PlanStep { command: "select.all".into(), params: json!({}), description: "target the canvas".into() },
            PlanStep { command: "image.adjustments.invert".into(), params: json!({}), description: "the edit".into() },
        ],
        creates_document: false,
        target_size: None,
    };
    let (result, evidence) = ExecutionEngine::execute_plan_in_with_evidence(&mut headless, &plan).unwrap();
    assert!(result.is_success);
    let capture = evidence.selection_capture.expect("a plan with a selection step yields a target-selection capture");
    assert_eq!(capture.step_index, 1, "the capture is taken immediately before the edit step it evaluates");
    assert_eq!(capture.bounds, [0, 0, 64, 64]);
    assert!(!capture.document_name.is_empty(), "the document identity rides the capture");
    let coverage = capture.coverage.expect("the selection's exact per-pixel coverage is preserved before the edit");
    assert_eq!(coverage.get("format").and_then(Value::as_str), Some("coverage-f32-rowmajor"));
    assert_eq!(coverage.get("coverage").and_then(Value::as_array).map(Vec::len), Some(64 * 64));
    let before = capture.before_pixels.expect("the exact before-pixels are captured before the edit");
    assert_eq!(before.get("format").and_then(Value::as_str), Some("rgba-f32-rowmajor"));
    assert_eq!(before.get("pixels").and_then(Value::as_array).map(Vec::len), Some(64 * 64 * 4));
    assert!(capture.region.is_some(), "the selection's bounding region is digested before the edit");
    assert!(capture.protected.is_empty(), "a full-canvas selection leaves no protected complement to measure");
    let stats = capture.color_stats.expect("coverage-aware colour statistics ride the capture");
    assert_eq!(stats.get("domain").and_then(Value::as_str), Some("selection"));
    assert_eq!(stats.get("meanColor").and_then(Value::as_str), Some("#FFFFFF"),
        "the pre-edit selection colour is white; the capture happened BEFORE the invert");
}

/// REAL-ENGINE FIXTURE: exact-rect digests and region colour statistics
/// measure the requested pixels, labeled by domain.
#[test]
fn region_digest_and_color_stats_measure_exact_rects() {
    let mut headless = Headless::new();
    LiveEditSession::command_run(&mut headless, "file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let digest = LiveEditSession::visual_digest(&mut headless, 4_000_000, Some([0, 0, 4, 4])).unwrap().unwrap();
    assert_eq!(digest.get("domain").and_then(Value::as_str), Some("region"));
    assert_eq!(digest.get("stride").and_then(Value::as_u64), Some(1), "a 16-px region samples every pixel");
    assert_eq!(digest.get("sampledPixels").and_then(Value::as_u64), Some(16));
    let stats = LiveEditSession::command_run(&mut headless, "image.analysis.colorStats",
        json!({"x": 0, "y": 0, "width": 8, "height": 8})).unwrap();
    assert_eq!(stats.get("domain").and_then(Value::as_str), Some("region"));
    assert_eq!(stats.get("meanColor").and_then(Value::as_str), Some("#FFFFFF"));
    // All-or-none region params are refused, never partially applied.
    assert!(LiveEditSession::command_run(&mut headless, "image.analysis.colorStats", json!({"x": 0})).is_err());
}
