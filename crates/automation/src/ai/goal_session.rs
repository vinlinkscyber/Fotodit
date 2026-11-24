//! Bounded synchronous goal attempts, not an autonomous UI or a new planner.
//!
//! Hosts must honor LiveEditSession's stable-document, admitted synchronous-command
//! contract. Name/path checks are hints, not unique document identity or state proof.
//! An observed pending/dialog receipt stops further dispatch but cannot cancel work
//! a nonconforming host already started. No retry follows an execution error.

use std::cell::RefCell;
use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::{AiContext, AiEngine, AiError, EditingIntent, EditingPlan, EvaluationReport,
    ExecutionEngine, ExecutionEvidence, ExecutionResult, IntentCategory, LiveEditSession,
    PlanStep, VisualScene};
use super::execution::{DigestMark, EVIDENCE_MAX_PIXELS};
use super::{VisualEvidence, VisualEvidenceClass};
use crate::AutomationError;

/// Limits bound planned edit steps, not renderer work, time, money or provider calls.
/// Inspection, optional digest attempts and engine rollback are not edit-budget steps.
#[derive(Clone, Debug)]
pub struct GoalRequest {
    pub original_prompt: String,
    pub max_attempts: usize,
    pub max_commands: usize,
}

impl GoalRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { original_prompt: prompt.into(), max_attempts: 2, max_commands: 32 }
    }
}

/// Explicit input to a caller-supplied repair planner. The original intent is never
/// reparsed from feedback, and previous observations are not new user instructions.
pub struct GoalRepairInput<'a> {
    pub original_intent: &'a EditingIntent,
    pub current_scene: &'a VisualScene,
    pub context: &'a AiContext,
    pub attempts: &'a [GoalAttempt],
}

pub enum GoalRepairDecision {
    Plan(EditingPlan),
    NeedsInput(String),
    Stop(String),
}

/// Does not widen AiPlanner or implicitly enable a remote planner. Implementations
/// are responsible for their own planning cost and permissions, just like AiPlanner.
pub trait GoalRepairAdapter {
    fn plan_repair(&self, input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError>;
}

/// Safe opt-out: the old planner is not presumed to understand structured repair.
pub struct NoGoalRepair;
impl GoalRepairAdapter for NoGoalRepair {
    fn plan_repair(&self, _input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError> {
        Ok(GoalRepairDecision::NeedsInput("The completed attempt did not verify the goal; no repair adapter was supplied.".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalStop {
    /// Evaluator reported satisfied after completed execution. Not independent
    /// proof of appearance, legibility, quality or every part of the user's goal.
    Completed,
    NeedsInput,
    Unmet,
    BudgetExhausted,
    RepeatedPlan,
    PlanningFailed,
    InvalidRequest,
    FinalStateUnverified,
}

#[derive(Clone, Debug)]
pub struct GoalCommandRecord {
    pub command: String,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub observation: bool,
}

#[derive(Clone, Debug)]
pub struct GoalAttempt {
    pub plan: EditingPlan,
    pub execution: Option<ExecutionResult>,
    pub evaluation: Option<EvaluationReport>,
    pub evidence: Option<ExecutionEvidence>,
    /// Actual host calls, including rollback and optional observation calls.
    /// A failed attempt may have no ExecutionResult but still have mutations.
    pub commands: Vec<GoalCommandRecord>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GoalOutcome {
    pub original_prompt: String,
    pub original_intent: Option<EditingIntent>,
    pub attempts: Vec<GoalAttempt>,
    pub stop: GoalStop,
    pub detail: String,
    /// True after uncertain execution/inspection/host-contract failure. Older
    /// successful attempt evidence must not be shown as the current canvas.
    pub final_state_unverified: bool,
    pub planned_commands: usize,
}

impl GoalOutcome {
    fn stop(mut self, stop: GoalStop, detail: impl Into<String>) -> Self {
        self.stop = stop;
        self.detail = detail.into();
        self.final_state_unverified = stop == GoalStop::FinalStateUnverified;
        self
    }
}

impl AiEngine {
    /// Additive goal coordinator. First plan uses this engine's existing planner;
    /// only a completed-but-unmet attempt can invoke the explicit repair adapter.
    /// Existing process_instruction* APIs and their behavior are unchanged.
    ///
    /// Initial/final scenes use inspection only here: optional color-statistics
    /// dispatch is deliberately omitted, rather than spending before plan admission.
    /// Pixel evidence remains opt-in through this API's execution path and retains
    /// its per-plan capture cap/unknown semantics. This is not an async GUI runner.
    pub fn process_goal_in<S: LiveEditSession + ?Sized>(
        &self,
        session: &mut S,
        request: &GoalRequest,
        context: &mut AiContext,
        repair: &dyn GoalRepairAdapter,
    ) -> GoalOutcome {
        let mut outcome = GoalOutcome {
            original_prompt: request.original_prompt.clone(), original_intent: None,
            attempts: Vec::new(), stop: GoalStop::InvalidRequest, detail: String::new(),
            final_state_unverified: false, planned_commands: 0,
        };
        // Hard ceilings prevent accidental unbounded loops. These are dispatch
        // limits only; they are not a grant for external planning or paid work.
        if request.original_prompt.trim().is_empty() || !(1..=8).contains(&request.max_attempts)
            || !(1..=128).contains(&request.max_commands)
        {
            return outcome.stop(GoalStop::InvalidRequest, "Give me a clear, non-empty edit request.");
        }
        let document = match session.inspect_active() {
            Ok(document) => document,
            Err(error) => return outcome.stop(GoalStop::FinalStateUnverified, error.to_string()),
        };
        let anchor = match identity_hint(&document) {
            Ok(anchor) => anchor,
            Err(error) => return outcome.stop(GoalStop::FinalStateUnverified, error),
        };
        let first_scene = document.as_ref().map(|doc| self.perception_engine.perceive(doc)).unwrap_or_default();
        let intent = match self.nl_engine.parse_intent(&request.original_prompt, &first_scene, context) {
            Ok(intent) => intent,
            Err(error) => return outcome.stop(GoalStop::PlanningFailed, error.to_string()),
        };
        outcome.original_intent = Some(intent.clone());
        // No per-category gate with no document open: a planner may lead with
        // file.new for ANY request below (creation is a capability, not a
        // keyword whitelist), and the transactional guard is keyed on
        // allow_creation. What is not allowed is the engine fabricating a
        // document for a request that never asked for one - see the guarded
        // auto-insert below. The GUI provider contract independently enforces
        // file.new-first on empty scenes before a plan ever reaches this point.
        let mut host = GoalHost { inner: session, anchor: RefCell::new(anchor),
            poison: RefCell::new(None), calls: Vec::new(), allow_creation: document.is_none() };
        let mut fingerprints = BTreeSet::new();
        for attempt_index in 0..request.max_attempts {
            let current = match host.inspect_active() {
                Ok(document) => document,
                Err(error) => return outcome.stop(GoalStop::FinalStateUnverified, error.to_string()),
            };
            let scene = current.as_ref().map(|doc| self.perception_engine.perceive(doc)).unwrap_or_default();
            let mut plan = if attempt_index == 0 {
                match self.planner.plan(&intent, &scene, context) {
                    Ok(plan) => plan,
                    Err(error) => return outcome.stop(GoalStop::PlanningFailed, error.to_string()),
                }
            } else {
                match repair.plan_repair(GoalRepairInput {
                    original_intent: &intent, current_scene: &scene, context, attempts: &outcome.attempts,
                }) {
                    Ok(GoalRepairDecision::Plan(plan)) => plan,
                    Ok(GoalRepairDecision::NeedsInput(reason)) => return outcome.stop(GoalStop::NeedsInput, reason),
                    Ok(GoalRepairDecision::Stop(reason)) => return outcome.stop(GoalStop::Unmet, reason),
                    Err(error) => return outcome.stop(GoalStop::PlanningFailed, error.to_string()),
                }
            };
            if attempt_index == 0 && current.is_none() && !plan.steps.iter().any(|step| step.command == "file.new") {
                // Auto-create only when creation was requested (GraphicDesign)
                // or the planner signalled it (target size / creates_document).
                // A non-creation request must never fabricate an unrelated
                // document: stop for input instead, with nothing dispatched.
                if intent.category != IntentCategory::GraphicDesign
                    && plan.target_size.is_none() && !plan.creates_document {
                    return outcome.stop(GoalStop::NeedsInput, AiError::NoDocumentOpen.to_string());
                }
                let [width, height] = plan.target_size.unwrap_or([1200, 1600]);
                plan.steps.insert(0, PlanStep { command: "file.new".into(),
                    params: json!({"width": width, "height": height, "background": "transparent"}),
                    description: "Create the requested document".into() });
                plan.creates_document = true;
            }
            // A planner may itself lead with file.new when no document is open; that is the
            // same creation case as an inserted step. Mark the plan accordingly or the
            // transactional guard would reject its first step as identity-changing.
            // With a document open the flag stays false, so a provider-led file.new there
            // is still refused instead of replacing the user's document.
            if attempt_index == 0 && current.is_none() && plan.steps.first().is_some_and(|step| step.command == "file.new") {
                plan.creates_document = true;
            }
            if attempt_index > 0 && (plan.creates_document || plan.steps.iter().any(|step| step.command == "file.new")) {
                return outcome.stop(GoalStop::PlanningFailed, "the correction would have replaced the document, which is not safe, so I stopped");
            }
            if plan.steps.is_empty() {
                return outcome.stop(GoalStop::NeedsInput, "no concrete edit steps came out of that");
            }
            if plan.steps.len() > request.max_commands.saturating_sub(outcome.planned_commands) {
                return outcome.stop(GoalStop::BudgetExhausted, "the plan grew too large to run safely, so I stopped before running it");
            }
            let fingerprint = plan_fingerprint(&plan);
            if !fingerprints.insert(fingerprint) {
                return outcome.stop(GoalStop::RepeatedPlan, "my next plan was the same as the last one, so I stopped instead of repeating it");
            }
            for step in &plan.steps {
                if let Err(error) = self.catalog.validate_command(&step.command, &step.params) {
                    return outcome.stop(GoalStop::PlanningFailed, error.to_string());
                }
            }
            outcome.planned_commands += plan.steps.len();
            let region_watch = RegionWatch::new(&scene, &plan);
            let region_before = region_watch.capture(&mut host);
            let start = host.calls.len();
            let mut attempt = GoalAttempt { plan: plan.clone(), execution: None, evaluation: None,
                evidence: None, commands: Vec::new(), error: None };
            match ExecutionEngine::execute_plan_in_with_evidence(&mut host, &plan) {
                Err(error) => {
                    attempt.error = Some(error.to_string());
                    attempt.commands = host.calls[start..].to_vec();
                    outcome.attempts.push(attempt);
                    return outcome.stop(GoalStop::FinalStateUnverified, error.to_string());
                }
                Ok((execution, evidence)) => {
                    attempt.execution = Some(execution.clone());
                    attempt.evidence = Some(evidence);
                    attempt.commands = host.calls[start..].to_vec();
                    if !execution.is_success {
                        outcome.attempts.push(attempt);
                        return outcome.stop(GoalStop::FinalStateUnverified, "the edit did not finish running");
                    }
                    let final_document = match host.inspect_active() {
                        Ok(Some(document)) => document,
                        other => {
                            let detail = match other { Err(error) => error.to_string(), _ => "The completed attempt left no inspectable document.".into() };
                            attempt.error = Some(detail.clone());
                            outcome.attempts.push(attempt);
                            return outcome.stop(GoalStop::FinalStateUnverified, detail);
                        }
                    };
                    let final_scene = self.perception_engine.perceive(&final_document);
                    let measured = region_watch.measure(&mut host, region_before);
                    let mut evaluation = self.eval_engine.evaluate(&intent, &scene, &final_scene, &execution);
                    measured.apply(&mut evaluation);
                    let satisfied = evaluation.intent_satisfied;
                    attempt.evaluation = Some(evaluation);
                    outcome.attempts.push(attempt);
                    // Record only completed executions, with the real original goal.
                    context.record_interaction(request.original_prompt.clone(), plan);
                    if satisfied {
                        return outcome.stop(GoalStop::Completed, "All edit steps completed.");
                    }
                }
            }
        }
        outcome.stop(GoalStop::BudgetExhausted, "I made several attempts without getting there, so I stopped")
    }
}

// A stable canonical JSON encoding, independent of map insertion order. No
// lossy hash: comparison uses the full ordered command/parameter sequence.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: std::collections::BTreeMap<_, _> = map.iter().map(|(key, value)| (key.clone(), canonical(value))).collect();
            let mut object = serde_json::Map::new();
            for (key, value) in sorted { object.insert(key, value); }
            Value::Object(object)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        value => value.clone(),
    }
}

fn plan_fingerprint(plan: &EditingPlan) -> String {
    canonical(&Value::Array(plan.steps.iter().map(|step| json!({"command": step.command, "params": step.params})).collect())).to_string()
}

fn identity_hint(document: &Option<Value>) -> Result<Option<Value>, String> {
    match document {
        None => Ok(None),
        Some(document) => {
            let name = document.get("name").and_then(Value::as_str).ok_or("Document name hint is missing.")?;
            let path = document.get("path").filter(|path| path.is_null() || path.is_string()).ok_or("Document path hint is missing.")?;
            Ok(Some(json!({"name": name, "path": path})))
        }
    }
}

fn deferred(value: &Value) -> bool {
    value.get("pending").and_then(Value::as_bool) == Some(true) || value.get("dialog").is_some()
}

struct GoalHost<'a, S: LiveEditSession + ?Sized> {
    inner: &'a mut S,
    anchor: RefCell<Option<Value>>,
    poison: RefCell<Option<String>>,
    calls: Vec<GoalCommandRecord>,
    allow_creation: bool,
}

impl<S: LiveEditSession + ?Sized> GoalHost<'_, S> {
    fn fail(&self, detail: impl Into<String>) -> AutomationError {
        let detail = detail.into();
        *self.poison.borrow_mut() = Some(detail.clone());
        AutomationError::App(detail)
    }

    fn check_document(&self, document: &Option<Value>) -> Result<(), AutomationError> {
        let hint = identity_hint(document).map_err(|error| self.fail(error))?;
        if hint != *self.anchor.borrow() {
            return Err(self.fail("Active-document identity hint changed; final state is unverified."));
        }
        Ok(())
    }
}

impl<S: LiveEditSession + ?Sized> LiveEditSession for GoalHost<'_, S> {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        if let Some(detail) = self.poison.borrow().as_ref() {
            return Err(AutomationError::App(detail.clone()));
        }
        let document = self.inner.inspect_active().map_err(|error| self.fail(error.to_string()))?;
        self.check_document(&document)?;
        Ok(document)
    }

    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
        self.inspect_active()?;
        let result = self.inner.command_run(command, params);
        self.calls.push(GoalCommandRecord { command: command.into(), output: result.as_ref().ok().cloned(),
            error: result.as_ref().err().map(ToString::to_string), observation: false });
        if result.as_ref().is_ok_and(deferred) {
            return Err(self.fail("Host returned a pending/dialog receipt instead of completed execution; already-started work may still commit. Further dispatch stopped, not cancelled."));
        }
        let document = self.inner.inspect_active().map_err(|error| self.fail(error.to_string()))?;
        if self.allow_creation && command == "file.new" && self.anchor.borrow().is_none() {
            *self.anchor.borrow_mut() = identity_hint(&document).map_err(|error| self.fail(error))?;
            self.allow_creation = false;
        } else {
            self.check_document(&document)?;
        }
        result
    }

    fn visual_digest(&mut self, max_pixels: u64, region: Option<[i64; 4]>) -> Option<Result<Value, String>> {
        if let Err(error) = self.inspect_active() { return Some(Err(error.to_string())); }
        let result = self.inner.visual_digest(max_pixels, region);
        self.calls.push(GoalCommandRecord { command: "visual_digest".into(),
            output: result.as_ref().and_then(|result| result.as_ref().ok().cloned()),
            error: result.as_ref().and_then(|result| result.as_ref().err().cloned()), observation: true });
        if let Some(result) = &result {
            if result.as_ref().is_ok_and(deferred) {
                let error = self.fail("Observation hook returned deferred work; final state unverified and further dispatch stopped.");
                return Some(Err(error.to_string()));
            }
        }
        if let Err(error) = self.inspect_active() { return Some(Err(error.to_string())); }
        result
    }
}

/// Measured before/after region evidence for one attempt. A sampled global
/// digest cannot prove region colour or unchanged-pixel claims, so the target
/// region (the union of the plan-targeted layers' bounds) and up to four
/// protected border regions (the canvas complement) get their own exact
/// digests at the evidence budget's stride - regions within 1M px sample every
/// pixel, making equal digests an exact unchanged-pixel proof there.
struct RegionWatch {
    target: Option<[i64; 4]>,
    protected: Vec<[i64; 4]>,
}

struct CapturedRegion {
    target: bool,
    mark: Option<DigestMark>,
}

/// Local measured evidence; `available` is false when the host offers no
/// digest capability, in which case nothing about the evaluation changes
/// (honest absence, never assumed change or no-change).
struct RegionMeasurement {
    target: Option<VisualEvidence>,
    protected: Vec<VisualEvidence>,
    available: bool,
}

impl RegionWatch {
    /// Target = union of the bounds of layers the plan addresses by `layer`/`id`
    /// param; protected = the canvas complement as up to four border rects.
    /// Regions over the evidence pixel budget are skipped (honest unknown,
    /// never a pretend measurement).
    fn new(scene: &VisualScene, plan: &EditingPlan) -> Self {
        let mut target: Option<[i64; 4]> = None;
        for step in &plan.steps {
            let Some(id) = step.params.get("layer").or_else(|| step.params.get("id")).and_then(Value::as_u64) else { continue };
            let Some(layer) = scene.layers.iter().find(|l| l.id == id) else { continue };
            let [x, y, w, h] = layer.bounds;
            if w <= 0 || h <= 0 { continue; }
            let (r, b) = (i64::from(x) + i64::from(w), i64::from(y) + i64::from(h));
            target = Some(match target {
                None => [i64::from(x), i64::from(y), i64::from(w), i64::from(h)],
                Some(t) => {
                    let x0 = t[0].min(i64::from(x));
                    let y0 = t[1].min(i64::from(y));
                    [x0, y0, (t[0] + t[2]).max(r) - x0, (t[1] + t[3]).max(b) - y0]
                }
            });
        }
        let fits = |w: i64, h: i64| w > 0 && h > 0 && (w as u64).checked_mul(h as u64).is_some_and(|a| a <= EVIDENCE_MAX_PIXELS);
        let target = target.filter(|t| fits(t[2], t[3]));
        let mut protected = Vec::new();
        if let Some([tx, ty, tw, th]) = target {
            let (cw, ch) = (i64::from(scene.width), i64::from(scene.height));
            for rect in [[0, 0, cw, ty], [0, ty + th, cw, ch - ty - th], [0, ty, tx, th], [tx + tw, ty, cw - tx - tw, th]] {
                if fits(rect[2], rect[3]) { protected.push(rect); }
            }
        }
        Self { target, protected }
    }

    /// One capture per region through the host's admitted digest hook.
    /// `None` when there is nothing to measure or the host has no capability.
    fn capture<S: LiveEditSession + ?Sized>(&self, session: &mut S) -> Option<Vec<CapturedRegion>> {
        if self.target.is_none() && self.protected.is_empty() { return None; }
        let mut out = Vec::new();
        for (is_target, rect) in self.target.iter().map(|r| (true, r)).chain(self.protected.iter().map(|r| (false, r))) {
            match ExecutionEngine::capture_digest(session, Some(*rect)) {
                Ok(Some(mark)) => out.push(CapturedRegion { target: is_target, mark: Some(mark) }),
                Ok(None) => return None,
                Err(_) => out.push(CapturedRegion { target: is_target, mark: None }),
            }
        }
        Some(out)
    }

    /// After-execution captures over the identical rects, compared on an
    /// identical grid only (the comparison itself classifies anything else
    /// as Unknown).
    fn measure<S: LiveEditSession + ?Sized>(&self, session: &mut S, before: Option<Vec<CapturedRegion>>) -> RegionMeasurement {
        let unavailable = || RegionMeasurement { target: None, protected: Vec::new(), available: false };
        let Some(before) = before else { return unavailable() };
        let Some(after) = self.capture(session) else { return unavailable() };
        let mut measurement = RegionMeasurement { target: None, protected: Vec::new(), available: true };
        for (b, a) in before.iter().zip(after.iter()) {
            let evidence = match (&b.mark, &a.mark) {
                (Some(b), Some(a)) => ExecutionEngine::compare_digests(b, a),
                _ => VisualEvidence { class: VisualEvidenceClass::Unknown,
                    detail: "a region capture failed or was refused; no evidence of change or of no-change".into() },
            };
            if b.target { measurement.target = Some(evidence); } else { measurement.protected.push(evidence); }
        }
        measurement
    }
}

impl RegionMeasurement {
    /// Apply measured criteria to the heuristic evaluation. A measured pixel
    /// change in a protected region blocks completion (correction or an honest
    /// unmet outcome follows); measured no-change in protected regions and
    /// change in the target are recorded as observations. No measurement
    /// upgrades a weak heuristic score into a completed goal on its own.
    fn apply(self, evaluation: &mut EvaluationReport) {
        if !self.available { return; }
        let changed = self.protected.iter().filter(|e| e.class == VisualEvidenceClass::Changed).count();
        if changed > 0 {
            evaluation.intent_satisfied = false;
            evaluation.observations.push(format!(
                "Measured pixel evidence: {changed} protected region(s) outside the target changed at the data level; unrelated content may have been affected."));
        } else if !self.protected.is_empty() && self.protected.iter().all(|e| e.class == VisualEvidenceClass::NoChangeObserved) {
            evaluation.observations.push(
                "Measured pixel evidence: no change observed in the protected regions outside the target.".into());
        }
        if let Some(target) = &self.target {
            match target.class {
                VisualEvidenceClass::Changed => evaluation.observations.push(
                    "Measured pixel evidence: the target region changed.".into()),
                VisualEvidenceClass::NoChangeObserved => evaluation.observations.push(
                    "Measured pixel evidence: no change observed in the target region; the edit may not have reached it.".into()),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod region_watch_tests {
    use super::*;

    fn scene(width: u32, height: u32, layers: &[(u64, [i32; 4])]) -> VisualScene {
        let layers = layers.iter().map(|(id, bounds)| {
            let mut layer = VisualElement { ..Default::default() };
            layer.id = *id;
            layer.bounds = *bounds;
            layer
        }).collect();
        VisualScene { width, height, layers, ..Default::default() }
    }

    fn plan(steps: &[(&str, u64)]) -> EditingPlan {
        EditingPlan { title: "t".into(), rationale: "r".into(), category: IntentCategory::ColorStyling, steps: steps.iter()
            .map(|(command, layer)| PlanStep { command: command.to_string(),
                params: json!({"layer": layer}), description: String::new() }).collect() }
    }

    #[test]
    fn region_watch_unions_target_bounds_and_measures_the_complement() {
        let scene = scene(1000, 800, &[(2, [100, 100, 300, 200]), (3, [150, 250, 100, 100])]);
        let watch = RegionWatch::new(&scene, &plan(&[("layer.fill.set", 2), ("layer.opacity.set", 3)]));
        assert_eq!(watch.target, Some([100, 100, 350, 250]));
        assert_eq!(watch.protected, vec![
            [0, 0, 1000, 100], [0, 350, 1000, 450], [0, 100, 100, 250], [450, 100, 550, 250]]);
    }

    #[test]
    fn region_watch_skips_oversized_and_unknown_targets() {
        let scene = scene(100, 100, &[(2, [10, 10, 20, 20])]);
        // A plan whose layer is not in the scene has no known target region.
        assert!(RegionWatch::new(&scene, &plan(&[("layer.fill.set", 99)])).target.is_none());
        // A region over the evidence pixel budget is skipped, never sampled into a false proof.
        let huge = scene(9000, 9000, &[(2, [0, 0, 5000, 5000])]);
        let watch = RegionWatch::new(&huge, &plan(&[("layer.fill.set", 2)]));
        assert!(watch.target.is_none() && watch.protected.is_empty());
    }
}
