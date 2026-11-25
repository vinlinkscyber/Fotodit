//! Goal coordination tests use scripted planning and admitted host boundaries only.
//! No provider requests, sleeps, UI automation or external services.
use std::cell::{Cell, RefCell};

use fotodit_automation::ai::{AiContext, AiEngine, AiError, AiPlanner, EditingIntent,
    EditingPlan, GoalRepairAdapter, GoalRepairDecision, GoalRepairInput, GoalRequest,
    GoalStop, IntentCategory, LiveEditSession, NoGoalRepair, PlanStep, VisualScene};
use fotodit_automation::{AutomationError, Headless};
use serde_json::{Value, json};

const GOAL: &str = "create a poster for my taxi busines to promote it its called chndrax";

#[derive(Clone)]
struct InitialPlan(EditingPlan);
impl AiPlanner for InitialPlan {
    fn plan(&self, _intent: &EditingIntent, _scene: &VisualScene, _context: &AiContext) -> Result<EditingPlan, AiError> {
        Ok(self.0.clone())
    }
}

struct Repair {
    plan: EditingPlan,
    calls: Cell<usize>,
    prompts: RefCell<Vec<String>>,
    layer_counts: RefCell<Vec<usize>>,
}
impl Repair {
    fn new(plan: EditingPlan) -> Self {
        Self { plan, calls: Cell::new(0), prompts: RefCell::new(Vec::new()), layer_counts: RefCell::new(Vec::new()) }
    }
}
impl GoalRepairAdapter for Repair {
    fn plan_repair(&self, input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError> {
        self.calls.set(self.calls.get() + 1);
        self.prompts.borrow_mut().push(input.original_intent.raw_prompt.clone());
        self.layer_counts.borrow_mut().push(input.current_scene.layers.len());
        assert!(!input.attempts.is_empty());
        assert!(input.attempts.last().unwrap().execution.as_ref().unwrap().is_success);
        assert!(!input.attempts.last().unwrap().evaluation.as_ref().unwrap().intent_satisfied);
        Ok(GoalRepairDecision::Plan(self.plan.clone()))
    }
}

fn step(command: &str, params: Value) -> PlanStep {
    PlanStep { command: command.into(), params, description: command.into() }
}
fn plan(steps: Vec<PlanStep>) -> EditingPlan {
    EditingPlan { title: "Scripted goal attempt".into(), rationale: "Offline coordinator test".into(),
        category: IntentCategory::GraphicDesign, steps, creates_document: false, target_size: None }
}
fn duplicate() -> EditingPlan { plan(vec![step("layer.duplicate", json!({}))]) }
fn text() -> EditingPlan { plan(vec![step("type.create", json!({"x": 1, "y": 1, "text": "chndrax taxi", "size": 12}))]) }
fn engine(plan: EditingPlan) -> AiEngine { AiEngine::with_planner(Box::new(InitialPlan(plan))) }

#[derive(Clone, Copy, PartialEq)]
enum Mode { Normal, Pending, Dialog, ErrorAfterMutation, InspectionAfterMutation, SwitchDocument, DigestPending, DigestChanges }

struct Host {
    document: Value,
    calls: Vec<String>,
    mode: Mode,
    mutated: bool,
}
impl Host {
    fn new(mode: Mode) -> Self {
        let mut s = fotodit_engine::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
        Self { document: fotodit_engine::inspect::document(s.active().unwrap()), calls: Vec::new(), mode, mutated: false }
    }
    fn mutate(&mut self, command: &str) {
        let length = {
            let history = self.document["history"].as_array_mut().unwrap();
            history.push(json!(command));
            history.len()
        };
        self.document["revision"] = json!(length);
        let mut layer = self.document["layers"][0].clone();
        layer["id"] = json!(self.document["layers"].as_array().unwrap().len() + 1);
        if command == "type.create" {
            layer["kind"] = json!("Type");
            layer["text"] = json!({"text": "chndrax taxi", "font": "Inter", "sizePt": 12});
        }
        self.document["layers"].as_array_mut().unwrap().push(layer);
        self.mutated = true;
    }
}
impl LiveEditSession for Host {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        if self.mode == Mode::InspectionAfterMutation && self.mutated {
            return Err(AutomationError::App("inspection unavailable after edit".into()));
        }
        Ok(Some(self.document.clone()))
    }
    fn command_run(&mut self, command: &str, _params: Value) -> Result<Value, AutomationError> {
        // Read-only measurement probes are not edits: they neither count as
        // dispatched commands nor mutate the scripted document.
        if command.starts_with("image.analysis.") { return Ok(json!({})); }
        self.calls.push(command.into());
        if command == "edit.undo" {
            self.document["history"].as_array_mut().unwrap().pop();
            return Ok(json!({}));
        }
        self.mutate(command);
        match self.mode {
            Mode::Pending => Ok(json!({"pending": true, "job": 7})),
            Mode::Dialog => Ok(json!({"dialog": 7})),
            Mode::ErrorAfterMutation => Err(AutomationError::App("mutated then failed".into())),
            Mode::SwitchDocument => { self.document["name"] = json!("other document"); Ok(json!({})) }
            _ => Ok(json!({})),
        }
    }
    fn visual_digest(&mut self, _max_pixels: u64, _region: Option<[i64; 4]>) -> Option<Result<Value, String>> {
        if self.mode == Mode::DigestPending {
            Some(Ok(json!({"pending": true, "job": 9})))
        } else if self.mode == Mode::DigestChanges {
            let tag = self.document["history"].as_array().unwrap().len();
            Some(Ok(json!({"domain": "canvas", "schema": 1, "hash": "fnv1a64",
                "quantization": "8bit/channel", "digest": format!("{tag:016x}"),
                "width": 8, "height": 8, "stride": 1, "sampledPixels": 64})))
        } else { None }
    }
}

#[test]
fn completed_but_unmet_repairs_with_fresh_scene_and_immutable_original_goal() {
    let mut host = Host::new(Mode::Normal);
    let initial_layers = host.document["layers"].as_array().unwrap().len();
    let repair = Repair::new(text());
    let mut context = AiContext::new();
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut context, &repair);
    assert_eq!(repair.calls.get(), 1);
    assert_eq!(&*repair.prompts.borrow(), &[GOAL.to_string()]);
    assert_eq!(&*repair.layer_counts.borrow(), &[initial_layers + 1]);
    assert_eq!(outcome.attempts.len(), 2);
    assert!(outcome.attempts[1].execution.as_ref().unwrap().is_success);
    // The evaluator accepts the repaired result, but completion now requires
    // exact per-pixel grounding and this scripted host serves no pixel data
    // (probes answer with an empty object), so the honest stop is unverified.
    assert!(outcome.attempts[1].evaluation.as_ref().unwrap().intent_satisfied);
    assert_eq!(outcome.stop, GoalStop::FinalStateUnverified, "{}", outcome.detail);
    assert!(outcome.final_state_unverified);
    assert!(context.prompt_history.iter().all(|prompt| prompt == GOAL));
}

#[test]
fn repeated_plan_ignores_labels_and_object_key_order_before_second_mutation() {
    let mut first = duplicate();
    first.steps[0].params = serde_json::from_str("{\"layer\":1,\"unused\":{\"b\":2,\"a\":1}}").unwrap();
    let mut repeated = first.clone();
    repeated.title = "Different title must not bypass loop guard".into();
    repeated.steps[0].description = "A new label".into();
    repeated.steps[0].params = serde_json::from_str("{\"unused\":{\"a\":1,\"b\":2},\"layer\":1}").unwrap();
    let mut host = Host::new(Mode::Normal);
    let outcome = engine(first).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &Repair::new(repeated));
    assert_eq!(outcome.stop, GoalStop::RepeatedPlan);
    assert_eq!(outcome.attempts.len(), 1);
    assert_eq!(host.calls, vec!["layer.duplicate"]);
}

#[test]
fn command_budget_rejects_entire_plan_before_dispatch() {
    let mut host = Host::new(Mode::Normal);
    let mut request = GoalRequest::new(GOAL);
    request.max_commands = 1;
    let outcome = engine(plan(vec![step("layer.duplicate", json!({})), step("layer.duplicate", json!({}))]))
        .process_goal_in(&mut host, &request, &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::BudgetExhausted);
    assert!(host.calls.is_empty());
    assert!(outcome.attempts.is_empty());
}

#[test]
fn attempt_budget_does_not_invoke_repair_after_last_attempt() {
    let mut host = Host::new(Mode::Normal);
    let repair = Repair::new(text());
    let mut request = GoalRequest::new(GOAL);
    request.max_attempts = 1;
    let outcome = engine(duplicate()).process_goal_in(&mut host, &request, &mut AiContext::new(), &repair);
    assert_eq!(outcome.stop, GoalStop::BudgetExhausted);
    assert_eq!(repair.calls.get(), 0);
    assert_eq!(host.calls, vec!["layer.duplicate"]);
}

#[test]
fn pending_or_dialog_receipt_poison_stops_all_further_dispatch_not_cancellation() {
    for mode in [Mode::Pending, Mode::Dialog] {
        let mut host = Host::new(mode);
        let repair = Repair::new(text());
        let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &repair);
        assert_eq!(outcome.stop, GoalStop::FinalStateUnverified);
        assert!(outcome.final_state_unverified);
        assert_eq!(repair.calls.get(), 0);
        assert_eq!(host.calls, vec!["layer.duplicate"]);
        let edits: Vec<_> = outcome.attempts[0].commands.iter()
            .filter(|call| call.command == "layer.duplicate" && !call.observation).collect();
        assert_eq!(edits.len(), 1, "exactly one actual duplicate dispatch must be ledgered");
        let expected = if mode == Mode::Pending { json!({"pending": true, "job": 7}) }
            else { json!({"dialog": 7}) };
        assert_eq!(edits[0].output.as_ref(), Some(&expected));
        assert!(host.mutated, "a late receipt is not proof of no mutation or cancellation");
    }
}

#[test]
fn mutation_then_error_never_replans_and_retains_actual_error_ledger() {
    let mut host = Host::new(Mode::ErrorAfterMutation);
    let repair = Repair::new(text());
    let mut context = AiContext::new();
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut context, &repair);
    assert_eq!(outcome.stop, GoalStop::FinalStateUnverified);
    assert_eq!(repair.calls.get(), 0);
    assert!(outcome.detail.contains("mutated then failed"));
    assert!(outcome.attempts[0].commands.iter().any(|call| call.error.is_some()));
    assert!(context.executed_plans.is_empty());
}

#[test]
fn post_mutation_inspection_failure_and_identity_change_stop_without_repair() {
    for mode in [Mode::InspectionAfterMutation, Mode::SwitchDocument] {
        let mut host = Host::new(mode);
        let repair = Repair::new(text());
        let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &repair);
        assert_eq!(outcome.stop, GoalStop::FinalStateUnverified);
        assert_eq!(repair.calls.get(), 0);
        assert_eq!(host.calls, vec!["layer.duplicate"]);
    }
}

#[test]
fn deferred_observation_stops_before_any_edit_dispatch() {
    let mut host = Host::new(Mode::DigestPending);
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::FinalStateUnverified);
    assert!(host.calls.is_empty());
    assert!(outcome.attempts[0].commands.iter().any(|call| call.observation));
}

#[test]
fn no_repair_adapter_asks_for_input_instead_of_repeating_old_planner() {
    let mut host = Host::new(Mode::Normal);
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::NeedsInput);
    assert_eq!(host.calls, vec!["layer.duplicate"]);
}

#[test]
fn invalid_request_and_invalid_command_stop_before_dispatch() {
    let mut host = Host::new(Mode::Normal);
    let mut request = GoalRequest::new(GOAL);
    request.max_attempts = 9;
    let outcome = engine(duplicate()).process_goal_in(&mut host, &request, &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::InvalidRequest);
    let outcome = engine(plan(vec![step("made.up.command", json!({}))]))
        .process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::PlanningFailed);
    assert!(host.calls.is_empty());
}

#[test]
fn real_admitted_headless_commands_complete_through_existing_engine() {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert!(matches!(outcome.stop, GoalStop::NeedsInput | GoalStop::BudgetExhausted));
    assert!(outcome.attempts[0].execution.as_ref().unwrap().is_success);
    assert!(outcome.attempts[0].evidence.is_some());
    assert!(!outcome.final_state_unverified);
}

#[test]
fn inserted_creation_counts_against_budget_before_any_real_host_dispatch() {
    let mut host = Headless::trusted_local();
    let mut request = GoalRequest::new(GOAL);
    request.max_commands = 1;
    let outcome = engine(text()).process_goal_in(&mut host, &request, &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::BudgetExhausted);
    assert!(host.inspect_active().unwrap().is_none());
}

#[test]
fn non_creation_request_with_no_document_never_fabricates_a_document() {
    // A recolour/enhance-style request with no document open and a plan that
    // never asked for creation must stop for input, not auto-create an
    // unrelated document. (Audit of the removed keyword gate.)
    let mut host = Headless::trusted_local();
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new("make it brighter"), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::NeedsInput, "{}", outcome.detail);
    assert!(host.inspect_active().unwrap().is_none(), "no document was fabricated");
}

#[test]
fn planner_led_creation_without_design_keywords_is_allowed_with_no_document() {
    // Creation is a capability, not a keyword whitelist: a non-design request
    // whose plan leads with file.new still creates, whatever the category.
    let mut host = Headless::trusted_local();
    let outcome = engine(plan(vec![step("file.new", json!({"width": 8, "height": 8, "background": "white"}))]))
        .process_goal_in(&mut host, &GoalRequest::new("make it brighter"), &mut AiContext::new(), &NoGoalRepair);
    assert!(host.inspect_active().unwrap().is_some(), "planner-led creation proceeds: {}", outcome.detail);
}

struct StopRepair(bool);
impl GoalRepairAdapter for StopRepair {
    fn plan_repair(&self, _input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError> {
        if self.0 { Ok(GoalRepairDecision::Stop("Need another strategy, not another edit".into())) }
        else { Err(AiError::PlanningError("repair planner unavailable".into())) }
    }
}

#[test]
fn explicit_adapter_stop_or_planning_error_preserves_completed_attempt_without_retry() {
    for stop in [true, false] {
        let mut host = Host::new(Mode::Normal);
        let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &StopRepair(stop));
        assert_eq!(outcome.stop, if stop { GoalStop::Unmet } else { GoalStop::PlanningFailed });
        assert_eq!(outcome.attempts.len(), 1);
        assert!(!outcome.final_state_unverified);
        assert_eq!(host.calls, vec!["layer.duplicate"]);
    }
}

#[test]
fn changed_pixel_digest_and_completed_steps_do_not_substitute_for_goal_satisfaction() {
    let mut host = Host::new(Mode::DigestChanges);
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::NeedsInput);
    let attempt = &outcome.attempts[0];
    assert!(attempt.execution.as_ref().unwrap().is_success);
    assert!(!attempt.evaluation.as_ref().unwrap().intent_satisfied);
    assert_eq!(attempt.evidence.as_ref().unwrap().steps[0].visual.class,
        fotodit_automation::ai::VisualEvidenceClass::Changed);
}

#[test]
fn unsupported_digest_is_explicit_unknown_not_a_success_measure() {
    let mut host = Host::new(Mode::Normal);
    let outcome = engine(duplicate()).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::NeedsInput);
    assert_eq!(outcome.attempts[0].evidence.as_ref().unwrap().steps[0].visual.class,
        fotodit_automation::ai::VisualEvidenceClass::Unknown);
}

/// Regression for the composer failure seen in the Windows CI run: a planner that
/// itself leads with `file.new` when no document is open is the same creation case as
/// an inserted step. Before the normalization fix, the transactional identity guard
/// rejected the first step, the goal stopped FinalStateUnverified with zero commands,
/// and the composer reported "no edits applied" with no document - the test then
/// panicked unwrapping the missing active document.
#[test]
fn planner_led_file_new_with_no_document_is_a_creation_plan_not_an_identity_change() {
    struct CreationHost {
        session: fotodit_engine::Session,
        calls: Vec<String>,
    }
    impl LiveEditSession for CreationHost {
        fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
            Ok(self.session.active().map(fotodit_engine::inspect::document))
        }
        fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
            self.calls.push(command.into());
            self.session.execute(command, params).map_err(|error| AutomationError::App(error.to_string()))
        }
    }
    let mut host = CreationHost { session: fotodit_engine::Session::new(), calls: Vec::new() };
    let initial = plan(vec![step("file.new", json!({"width": 240, "height": 320, "background": "transparent"}))]);
    assert!(!initial.creates_document, "the planner flag is false, exactly as the provider bridge parses it");
    let outcome = engine(initial).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(host.calls, vec!["file.new"], "the leading creation step must dispatch, not stop at the identity guard: {}", outcome.detail);
    assert!(outcome.attempts[0].execution.as_ref().unwrap().is_success, "{}", outcome.detail);
    assert_ne!(outcome.stop, GoalStop::FinalStateUnverified, "{}", outcome.detail);
    let document = fotodit_engine::inspect::document(host.session.active().expect("the created document stays open"));
    assert_eq!(document["width"], json!(240));
    assert_eq!(document["height"], json!(320));
}

/// The open-document side of the same boundary: a planner-led file.new while a
/// document is open keeps creates_document=false, so the identity guard still refuses
/// it and the user's document is never replaced.
#[test]
fn planner_led_file_new_with_an_open_document_stays_refused() {
    let mut host = Host::new(Mode::Normal);
    let before = host.document.clone();
    let initial = plan(vec![step("file.new", json!({"width": 240, "height": 320, "background": "transparent"}))]);
    let outcome = engine(initial).process_goal_in(&mut host, &GoalRequest::new(GOAL), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::FinalStateUnverified, "{}", outcome.detail);
    assert!(host.calls.is_empty(), "nothing dispatched: {calls:?}", calls = host.calls);
    assert_eq!(host.document, before, "the open document is untouched");
}

fn tonal_observations(outcome: &fotodit_automation::ai::GoalOutcome) -> Vec<String> {
    outcome.attempts.iter()
        .filter_map(|a| a.evaluation.as_ref())
        .flat_map(|e| e.observations.iter().cloned())
        .filter(|o| o.contains("tonal"))
        .collect()
}

fn tonal_plan() -> EditingPlan {
    plan(vec![step("layer.newAdjustmentLayer.brightnessContrast", json!({"brightness": 60, "contrast": 0}))])
}

/// Real engine, real exact canvas pixels: a sole tonal request is measured on
/// the canvas and the direction is judged (met or not met, never skipped).
#[test]
fn real_engine_measures_a_sole_tonal_request_on_exact_canvas_pixels() {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 16, "height": 16, "background": "black"})).unwrap();
    let outcome = engine(tonal_plan()).process_goal_in(&mut host, &GoalRequest::new("make this photo brighter"), &mut AiContext::new(), &NoGoalRepair);
    assert!(outcome.attempts[0].execution.as_ref().unwrap().is_success);
    let obs = tonal_observations(&outcome);
    assert!(obs.iter().any(|o| o.contains("global tonal criterion")), "the tonal criterion must run on exact pixels: {obs:?}");
}

/// A multi-part request: one tonal match is evidence for that property only and
/// can never clear the request's other requirements or complete the goal.
#[test]
fn real_engine_tonal_match_does_not_complete_a_multi_part_request() {
    for prompt in ["make this photo brighter and add a vignette", "make it brighter, then crop it square", "don't make this photo brighter"] {
        let mut host = black_canvas_host();
        let outcome = engine(tonal_plan()).process_goal_in(&mut host, &GoalRequest::new(prompt), &mut AiContext::new(), &NoGoalRepair);
        assert_ne!(outcome.stop, GoalStop::Completed, "{prompt}");
        let obs = tonal_observations(&outcome);
        assert!(!obs.iter().any(|o| o.contains("global tonal criterion met")), "{prompt}: {obs:?}");
    }
}

fn black_canvas_host() -> Headless {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 16, "height": 16, "background": "black"})).unwrap();
    host
}

/// Provider-admitted commands that have no CommandCatalog entry (shape, layer
/// styles, type anti-alias, image rotation) must reach the real engine through
/// the goal loop, and engine-side parameter constraints still apply.
#[test]
fn real_engine_runs_uncatalogued_admitted_commands_through_the_goal_loop() {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 32, "height": 24, "background": "white"})).unwrap();
    let steps = vec![
        step("shape.create", json!({"kind": "rect", "rect": [4, 4, 16, 10], "fill": "#cc2200"})),
        step("layer.layerStyle.dropShadow", json!({"layer": {"$step": 1}, "distance": 3, "size": 4})),
        step("layer.layerStyle.stroke", json!({"layer": {"$step": 1}, "size": 2, "position": "outside", "color": "#000000"})),
        step("image.imageRotation.90cw", json!({})),
    ];
    let outcome = engine(plan(steps)).process_goal_in(&mut host, &GoalRequest::new("add a red box with a shadow and an outline, then rotate the canvas"), &mut AiContext::new(), &NoGoalRepair);
    assert_ne!(outcome.stop, GoalStop::PlanningFailed, "{:?}", outcome.stop);
    let execution = outcome.attempts[0].execution.as_ref().expect("the plan must reach execution");
    assert!(execution.is_success, "uncatalogued admitted commands must execute: {execution:?}");
    let doc = host.inspect_active().unwrap().expect("document stays open");
    let text = doc.to_string();
    assert!(text.contains("\"width\":24") && text.contains("\"height\":32"), "quarter turn swaps the canvas size: {text}");
}

/// Admission is by id only: a made-up id is still refused before dispatch, and
/// the engine still rejects out-of-range parameters of an admitted command.
#[test]
fn admission_refuses_unknown_ids_and_engine_still_rejects_bad_style_params() {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 16, "height": 16, "background": "white"})).unwrap();
    let outcome = engine(plan(vec![step("layer.layerStyle.notAStyle", json!({}))]))
        .process_goal_in(&mut host, &GoalRequest::new("add a style"), &mut AiContext::new(), &NoGoalRepair);
    assert_eq!(outcome.stop, GoalStop::PlanningFailed);
    assert!(outcome.attempts.is_empty() || outcome.attempts[0].execution.is_none());
    let bad = engine(plan(vec![step("layer.layerStyle.stroke", json!({"layer": 1, "size": 100000, "position": "sideways"}))]))
        .process_goal_in(&mut host, &GoalRequest::new("add an outline"), &mut AiContext::new(), &NoGoalRepair);
    assert!(bad.attempts.first().and_then(|a| a.execution.as_ref()).is_none_or(|e| !e.is_success), "bad style params must not succeed");
}

/// Invalid parameters on an admitted, uncatalogued command are refused before
/// dispatch for the whole plan: an earlier valid step must not have edited the
/// document.
#[test]
fn invalid_admitted_params_are_refused_before_any_earlier_step_edits_the_document() {
    for bad in [
        json!({"layer": {"$step": 1}, "size": 100000, "position": "outside"}),
        json!({"layer": {"$step": 1}, "size": 2, "position": "above"}),
        json!({"layer": {"$step": 1}, "size": 2, "notAStyleKey": true}),
    ] {
        let mut host = Headless::trusted_local();
        host.command_run("file.new", json!({"width": 32, "height": 24, "background": "white"})).unwrap();
        let before = host.inspect_active().unwrap();
        let steps = vec![
            step("shape.create", json!({"kind": "rect", "rect": [4, 4, 16, 10], "fill": "#cc2200"})),
            step("layer.layerStyle.stroke", bad.clone()),
        ];
        let outcome = engine(plan(steps)).process_goal_in(&mut host, &GoalRequest::new("add a red box with an outline"), &mut AiContext::new(), &NoGoalRepair);
        assert_eq!(outcome.stop, GoalStop::PlanningFailed, "{bad}");
        assert!(outcome.attempts.iter().all(|a| a.execution.is_none()), "{bad}: nothing may dispatch");
        assert_eq!(host.inspect_active().unwrap(), before, "{bad}: the document must be untouched");
    }
}
