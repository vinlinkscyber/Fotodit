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
            return outcome.stop(GoalStop::InvalidRequest, "Use a nonempty goal, 1..=8 attempts and 1..=128 planned commands.");
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
        if document.is_none() && intent.category != IntentCategory::GraphicDesign {
            return outcome.stop(GoalStop::NeedsInput, AiError::NoDocumentOpen.to_string());
        }
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
                return outcome.stop(GoalStop::PlanningFailed, "Repair must use the existing document, not create or replace it.");
            }
            if plan.steps.is_empty() {
                return outcome.stop(GoalStop::NeedsInput, "The planner supplied no edit steps; the goal is not verified.");
            }
            if plan.steps.len() > request.max_commands.saturating_sub(outcome.planned_commands) {
                return outcome.stop(GoalStop::BudgetExhausted, "The next plan exceeds the remaining planned-command budget; it was not dispatched.");
            }
            let fingerprint = plan_fingerprint(&plan);
            if !fingerprints.insert(fingerprint) {
                return outcome.stop(GoalStop::RepeatedPlan, "The same command/parameter sequence was proposed again; no repeated edits were dispatched.");
            }
            for step in &plan.steps {
                if let Err(error) = self.catalog.validate_command(&step.command, &step.params) {
                    return outcome.stop(GoalStop::PlanningFailed, error.to_string());
                }
            }
            outcome.planned_commands += plan.steps.len();
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
                        return outcome.stop(GoalStop::FinalStateUnverified, "Execution did not complete; no repair was attempted.");
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
                    let evaluation = self.eval_engine.evaluate(&intent, &scene, &final_scene, &execution);
                    let satisfied = evaluation.intent_satisfied;
                    attempt.evaluation = Some(evaluation);
                    outcome.attempts.push(attempt);
                    // Record only completed executions, with the real original goal.
                    context.record_interaction(request.original_prompt.clone(), plan);
                    if satisfied {
                        return outcome.stop(GoalStop::Completed, "Completed execution; the evaluator reported the original goal satisfied. This is evaluator evidence, not independent visual-quality proof.");
                    }
                }
            }
        }
        outcome.stop(GoalStop::BudgetExhausted, "Attempt limit reached after completed but unmet edits; no further repair was dispatched.")
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

    fn visual_digest(&mut self, max_pixels: u64) -> Option<Result<Value, String>> {
        if let Err(error) = self.inspect_active() { return Some(Err(error.to_string())); }
        let result = self.inner.visual_digest(max_pixels);
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
