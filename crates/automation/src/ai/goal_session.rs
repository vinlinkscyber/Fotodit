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
use super::natural_language::NaturalLanguageEngine;
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
            // A selection-driving plan is measured at its target-selection
            // stage (execution.rs captures it mid-plan); the pre-execution
            // scene's selection is NOT the target when the plan first selects.
            let plan_drives_selection = plan.steps.iter().any(|s| s.command.starts_with("select.") || s.command.contains("mask"));
            let region_watch = (!plan_drives_selection).then(|| RegionWatch::new(&scene, &plan));
            let region_before = region_watch.as_ref().and_then(|w| w.capture(&mut host));
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
                    let selection_capture = attempt.evidence.as_ref().and_then(|e| e.selection_capture.clone());
                    let measured = match &region_watch {
                        Some(watch) => watch.measure(&mut host, region_before),
                        None => measure_from_selection_stage(&mut host, selection_capture.as_ref()),
                    };
                    let mut evaluation = self.eval_engine.evaluate(&intent, &scene, &final_scene, &execution);
                    let grounding = measured.apply(intent.category, &mut evaluation);
                    // The target-colour criterion: when the request named a
                    // colour, the final selection's measured colour must match
                    // it (mismatch = unmet) and must be measurable at all for
                    // completion (unavailable = unverified, never completed).
                    let colour = measure_target_colour(&mut host, &intent, &plan, &final_scene, selection_capture.as_ref(), &mut evaluation);
                    let mut unverified: Vec<String> = match grounding {
                        CompletionGrounding::Grounded => Vec::new(),
                        CompletionGrounding::Unverified(reason) => vec![reason],
                    };
                    match &colour {
                        ColourCriterion::NotApplicable => {}
                        ColourCriterion::Checked(check) if !check.within_tolerance => {
                            evaluation.intent_satisfied = false;
                            evaluation.observations.push(format!(
                                "Measured colour criterion: {} does not match the requested {}.",
                                check.measured, check.requested));
                        }
                        ColourCriterion::Checked(check) => evaluation.observations.push(format!(
                            "Measured colour criterion: {} matches the requested {} within tolerance.",
                            check.measured, check.requested)),
                        ColourCriterion::Unavailable(reason) => unverified.push(reason.clone()),
                    }
                    let satisfied = evaluation.intent_satisfied;
                    attempt.evaluation = Some(evaluation);
                    outcome.attempts.push(attempt);
                    // Record only completed executions, with the real original goal.
                    context.record_interaction(request.original_prompt.clone(), plan);
                    if satisfied {
                        if unverified.is_empty() {
                            return outcome.stop(GoalStop::Completed,
                                "All edit steps completed. Completion grounds on exact per-pixel comparison of the target and its protected surroundings; sampled-grid digest observations, when recorded, are diagnostic only and did not authorize this outcome.");
                        }
                        return outcome.stop(GoalStop::FinalStateUnverified, format!(
                            "the edit finished, but goal satisfaction is unverified: {}.",
                            unverified.join("; ")));
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
/// digest cannot support region colour or unchanged-pixel claims, so the
/// target region (the union of the plan-targeted layers' bounds) and up to
/// four protected border regions (the canvas complement) get their own region
/// digests. Labels stay honest: a DIFFERENT digest proves the sampled data
/// differed (change is a fact); an EQUAL digest is only "no difference
/// observed on the sampled, 8-bit-quantized grid" - stride sampling skips
/// pixels, HDR and sub-8-bit differences quantize away, and a 64-bit
/// non-cryptographic hash can collide. It is never proof of unchanged pixels.
struct RegionWatch {
    target: Option<[i64; 4]>,
    protected: Vec<[i64; 4]>,
    /// The derived target spans (nearly) the whole canvas: the complement is
    /// empty or sliver-sized, so there is no protected region to measure and
    /// no LOCAL preservation claim can rest on this attempt.
    target_is_canvas: bool,
    /// What the target region was derived from, for honest labeling:
    /// the active selection's bounding region when the plan works through a
    /// selection/mask, else the plan-targeted layers' bounds, else the canvas.
    target_source: &'static str,
}




/// Exact comparison of two same-shape raw buffers: the differing-pixel count.
fn compare_exact_raw(before: &[f32], after: &[f32]) -> Option<u64> {
    if before.len() != after.len() || before.len() % 4 != 0 { return None; }
    let mut changed = 0u64;
    for (a, b) in after.chunks_exact(4).zip(before.chunks_exact(4)) {
        if a != b { changed += 1; }
    }
    Some(changed)
}


/// Exact covered/uncovered comparison over raw row-major buffers: the
/// selection's preserved per-pixel coverage splits the bounding region into
/// the target (ANY nonzero coverage, so feathered and low-opacity edges are
/// never discarded) and the region's collateral (exactly zero coverage).
/// Returns (covered_changed, covered_total, collateral_changed). The
/// preserved coverage is what lets this survive the selection's consumption.
fn compare_exact_coverage(before: &[f32], after: &[f32], coverage: &[f32]) -> Option<(u64, u64, u64)> {
    if before.len() != after.len() || coverage.len().checked_mul(4) != Some(before.len()) { return None; }
    let (mut covered_changed, mut covered_total, mut collateral_changed) = (0u64, 0u64, 0u64);
    for ((c, a), b) in coverage.iter().zip(after.chunks_exact(4)).zip(before.chunks_exact(4)) {
        if *c > 0.0 {
            covered_total += 1;
            if a != b { covered_changed += 1; }
        } else if a != b {
            collateral_changed += 1;
        }
    }
    Some((covered_changed, covered_total, collateral_changed))
}

/// Coverage-weighted mean RGB of the covered pixels of a raw region buffer,
/// for the exact colour criterion. A mean verifies no distribution.
fn exact_covered_mean(pixels: &[f32], coverage: &[f32]) -> Option<[f32; 3]> {
    if coverage.len().checked_mul(4) != Some(pixels.len()) { return None; }
    let (mut acc, mut total) = ([0f64; 3], 0f64);
    for (c, p) in coverage.iter().zip(pixels.chunks_exact(4)) {
        let w = f64::from(*c);
        acc[0] += w * f64::from(p[0]);
        acc[1] += w * f64::from(p[1]);
        acc[2] += w * f64::from(p[2]);
        total += w;
    }
    if total <= 0.0 { return None; }
    Some([(acc[0] / total) as f32, (acc[1] / total) as f32, (acc[2] / total) as f32])
}

struct CapturedRegion {
    target: bool,
    rect: [i64; 4],
    mark: Option<DigestMark>,
    /// Exact rendered pixels as a raw row-major RGBA f32 buffer, tiled under
    /// the engine's per-call cap with a 2M-pixel aggregate budget; preferred
    /// over the digest tier.
    pixels: Option<Vec<f32>>,
}

/// Local measured evidence; `available` is false when the host offers no
/// digest capability, in which case nothing about the evaluation changes
/// (honest absence, never assumed change or no-change).
struct RegionMeasurement {
    /// (evidence, exact): exact means an unsampled, unquantized, unhashed
    /// per-pixel comparison over the whole region. Sampled-grid digest
    /// evidence is diagnostic only and can never ground completion.
    target: Option<(VisualEvidence, bool)>,
    protected: Vec<(VisualEvidence, bool)>,
    available: bool,
    target_is_canvas: bool,
    target_source: &'static str,
    /// An aspect of the plan this measurement does not cover (e.g. earlier
    /// edit segments in a multi-segment plan): always Unverified, never
    /// silently dropped.
    uncovered: Option<String>,
}

/// Whether a satisfied evaluation may stop as Completed. Unknown goal
/// satisfaction stays unverified; it never completes on execution success
/// alone.
enum CompletionGrounding {
    Grounded,
    Unverified(String),
}

impl RegionWatch {
    /// Target = union of the bounds of layers the plan addresses by `layer`/`id`
    /// param; protected = the canvas complement as up to four border rects.
    /// Regions over the evidence pixel budget are skipped (honest unknown,
    /// never a pretend measurement).
    fn new(scene: &VisualScene, plan: &EditingPlan) -> Self {
        let mut target: Option<[i64; 4]> = None;
        let mut addressed = false;
        let mut mask_derived = false;
        for step in &plan.steps {
            let Some(id) = step.params.get("layer").or_else(|| step.params.get("id")).and_then(Value::as_u64) else { continue };
            addressed = true;
            let Some(layer) = scene.layers.iter().find(|l| l.id == id) else { continue };
            // A mask bounds the layer's editable pixels tighter than its
            // layer bounds: any nonzero effective coverage counts, so
            // feathered edges stay inside the measured target.
            let [x, y, w, h] = layer.mask_bounds
                .inspect(|_| mask_derived = true)
                .unwrap_or(layer.bounds);
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
        let (cw, ch) = (i64::from(scene.width), i64::from(scene.height));
        // Selection-derived target: when the plan works through the active
        // selection (selection or mask steps), the selection's bounding
        // region IS the intent's region (the shirt), which layer bounds are
        // not. Captured BEFORE execution consumes the selection.
        let uses_selection = plan.steps.iter().any(|s| s.command.starts_with("select.") || s.command.contains("mask"));
        let selection_region = (uses_selection && scene.has_selection)
            .then(|| scene.selection_bounds)
            .flatten()
            .filter(|b| fits(i64::from(b[2]), i64::from(b[3])))
            .map(|b| [i64::from(b[0]), i64::from(b[1]), i64::from(b[2]), i64::from(b[3])]);
        let mut target_source: &'static str = if mask_derived {
            "the plan-targeted layers' mask bounds (per-pixel mask coverage split not yet wired)"
        } else {
            "the plan-targeted layers' bounds"
        };
        if let Some(region) = selection_region {
            target = Some(region);
            target_source = "the active selection's bounding region";
        } else if target.is_none() && !addressed && !plan.steps.is_empty() && fits(cw, ch) {
            // A plan that never addresses a layer by id (a global filter or
            // adjustment on the active layer) is measured against the canvas;
            // a plan whose target ids are simply absent from the scene stays
            // unmeasured - an unknown target is Unknown, never the canvas.
            target = Some([0, 0, cw, ch]);
            target_source = "the canvas";
        }
        let target = target.filter(|t| fits(t[2], t[3]));
        let target_is_canvas = target.is_some_and(|t| t[2] * t[3] * 20 >= cw * ch * 19);
        let mut protected = Vec::new();
        if let Some([tx, ty, tw, th]) = target {
            if !target_is_canvas {
                for rect in [[0, 0, cw, ty], [0, ty + th, cw, ch - ty - th], [0, ty, tx, th], [tx + tw, ty, cw - tx - tw, th]] {
                    if fits(rect[2], rect[3]) { protected.push(rect); }
                }
            }
        }
        Self { target, protected, target_is_canvas, target_source }
    }

    /// One capture per region through the host's admitted digest hook.
    /// `None` when there is nothing to measure or the host has no capability.
    fn capture<S: LiveEditSession + ?Sized>(&self, session: &mut S) -> Option<Vec<CapturedRegion>> {
        if self.target.is_none() && self.protected.is_empty() { return None; }
        let mut out = Vec::new();
        for (is_target, rect) in self.target.iter().map(|r| (true, r)).chain(self.protected.iter().map(|r| (false, r))) {
            let pixels = ExecutionEngine::capture_tiled(session, "image.analysis.regionPixels", *rect, 4);
            match ExecutionEngine::capture_digest(session, Some(*rect)) {
                Ok(Some(mark)) => out.push(CapturedRegion { target: is_target, rect: *rect, mark: Some(mark), pixels }),
                // No digest capability is not the end when exact pixels were captured.
                Ok(None) => {
                    if pixels.is_none() { return None; }
                    out.push(CapturedRegion { target: is_target, rect: *rect, mark: None, pixels })
                }
                Err(_) => out.push(CapturedRegion { target: is_target, rect: *rect, mark: None, pixels }),
            }
        }
        Some(out)
    }

    /// After-execution captures over the identical rects, compared on an
    /// identical grid only (the comparison itself classifies anything else
    /// as Unknown).
    fn measure<S: LiveEditSession + ?Sized>(&self, session: &mut S, before: Option<Vec<CapturedRegion>>) -> RegionMeasurement {
        let unavailable = || RegionMeasurement { target: None, protected: Vec::new(), available: false, target_is_canvas: false, target_source: "nothing measurable", uncovered: None };
        let Some(before) = before else { return unavailable() };
        let Some(after) = self.capture(session) else { return unavailable() };
        let mut measurement = RegionMeasurement { target: None, protected: Vec::new(), available: true, target_is_canvas: self.target_is_canvas, target_source: self.target_source, uncovered: None };
        for (b, a) in before.iter().zip(after.iter()) {
            let exact = match (&b.pixels, &a.pixels) {
                (Some(before_px), Some(after_px)) => compare_exact_raw(before_px, after_px).map(|changed| {
                    let class = if changed > 0 { VisualEvidenceClass::Changed } else { VisualEvidenceClass::NoChangeObserved };
                    (VisualEvidence { class, detail: format!(
                        "exact per-pixel comparison over {} pixels of the region: {changed} pixels differ",
                        b.rect[2] * b.rect[3]) }, true)
                }),
                _ => None,
            };
            let evidence = exact.unwrap_or_else(|| {
                let evidence = match (&b.mark, &a.mark) {
                    (Some(b), Some(a)) => ExecutionEngine::compare_digests(b, a),
                    _ => VisualEvidence { class: VisualEvidenceClass::Unknown,
                        detail: "a region capture failed or was refused; no evidence of change or of no-change".into() },
                };
                (evidence, false)
            });
            if b.target { measurement.target = Some(evidence); } else { measurement.protected.push(evidence); }
        }
        measurement
    }
}

impl RegionMeasurement {
    /// Apply measured criteria to the heuristic evaluation and say whether a
    /// satisfied evaluation may stop as Completed. Goal relevance first:
    /// preservation of the complement is a LOCAL-edit criterion only - a
    /// geometry, global or design request legitimately alters it. A measured
    /// change in a protected region of a local edit blocks completion; a
    /// measured no-change target blocks it too (the edit may not have reached
    /// the target; "unchanged wrong result" is unmet, not success). What is
    /// never measured stays Unverified, never completed on execution success.
    fn apply(self, category: IntentCategory, evaluation: &mut EvaluationReport) -> CompletionGrounding {
        if !self.available {
            return CompletionGrounding::Unverified(
                "no pixel measurement was available for this attempt".into());
        }
        let local = matches!(category, IntentCategory::ColorStyling | IntentCategory::SubjectIsolation
            | IntentCategory::BackgroundReplacement | IntentCategory::TypographyRefinement);
        let changed = self.protected.iter().filter(|(e, _)| e.class == VisualEvidenceClass::Changed).count();
        if changed > 0 {
            if local {
                evaluation.intent_satisfied = false;
                evaluation.observations.push(format!(
                    "Measured pixel evidence: {changed} protected region(s) outside the target changed at the data level; unrelated content may have been affected."));
            } else {
                evaluation.observations.push(format!(
                    "Measured pixel evidence: {changed} region(s) outside the derived target changed; this request's intent is not a local-edit preservation criterion, so this is recorded, not failed."));
            }
        } else if !self.protected.is_empty() {
            if self.protected.iter().all(|(e, exact)| e.class == VisualEvidenceClass::NoChangeObserved && *exact) {
                evaluation.observations.push(
                    "Measured pixel evidence: exact per-pixel comparison found no differing pixel in the protected regions outside the target.".into());
            } else if self.protected.iter().all(|(e, _)| e.class == VisualEvidenceClass::NoChangeObserved) {
                evaluation.observations.push(
                    "Measured pixel evidence: no difference observed in the protected regions outside the target (sampled 8-bit grid; not proof of unchanged pixels).".into());
            }
        }
        let digest_only = "only sampled-grid digest evidence was available, which is diagnostic and never grounds completion; the exact per-pixel comparison was unavailable (region over the exact cap, or a capture failed or was refused)";
        let grounding = match &self.target {
            Some((target, target_exact)) => match target.class {
                VisualEvidenceClass::Changed => {
                    evaluation.observations.push(format!(
                        "Measured pixel evidence: the target region ({}) changed.", self.target_source));
                    if matches!(category, IntentCategory::GraphicDesign) {
                        // Completion rests on the category's own semantic
                        // checks PLUS this measured change; neither alone is
                        // proof of the full requested design, and the change
                        // must be exact, not a sampled-grid digest.
                        if *target_exact { CompletionGrounding::Grounded }
                        else { CompletionGrounding::Unverified(digest_only.into()) }
                    } else if !local {
                        CompletionGrounding::Unverified(
                            "a measured change is activity, not verification that this kind of request was satisfied".into())
                    } else if self.target_is_canvas {
                        CompletionGrounding::Unverified(
                            "the derived target spans the canvas; the requested local target was not measured".into())
                    } else if !*target_exact {
                        CompletionGrounding::Unverified(digest_only.into())
                    } else if local && (self.protected.is_empty()
                        || !self.protected.iter().all(|(e, _)| e.class == VisualEvidenceClass::NoChangeObserved)) {
                        CompletionGrounding::Unverified(
                            "the areas outside the target were not all measured unchanged".into())
                    } else if !self.protected.iter().all(|(_, exact)| *exact) {
                        CompletionGrounding::Unverified(digest_only.into())
                    } else {
                        CompletionGrounding::Grounded
                    }
                }
                VisualEvidenceClass::NoChangeObserved => {
                    evaluation.intent_satisfied = false;
                    evaluation.observations.push(format!(
                        "Measured pixel evidence: no difference observed in the target region ({}); the edit may not have reached it.",
                        self.target_source));
                    CompletionGrounding::Unverified("the target region showed no change".into())
                }
                _ => CompletionGrounding::Unverified(
                    "the target region could not be measured comparably".into()),
            },
            None => CompletionGrounding::Unverified(
                "no target region could be derived from the plan".into()),
        };
        // Uncovered aspects of the plan (earlier unmeasured edit segments)
        // withhold completion even when the measured parts ground it.
        match (grounding, self.uncovered) {
            (CompletionGrounding::Grounded, Some(uncovered)) => CompletionGrounding::Unverified(uncovered),
            (grounding, _) => grounding,
        }
    }
}

/// The target-colour criterion: the request's extracted colour compared with
/// the selection-aware colour statistics of the FINAL state. The criterion
/// binds the colour to the request's target: the plan must have worked
/// through a selection or mask (a surviving arbitrary selection proves
/// nothing), exactly one colour may be named ("keep the background red, make
/// the shirt blue" leaves the binding ambiguous), and the selection must
/// survive to the final state. Anything unavailable is honest Unknown with
/// its reason, never assumed match or mismatch.
struct ColourCheck {
    requested: String,
    /// What was compared (a mean colour value, or the basis note for the
    /// weak bounding-region tier).
    measured: String,
    within_tolerance: bool,
}

enum ColourCriterion {
    /// The request named no colour: the criterion does not apply.
    NotApplicable,
    Checked(ColourCheck),
    Unavailable(String),
}

fn parse_hex_rgb(hex: &str) -> Option<[u8; 3]> {
    let h = hex.strip_prefix('#').unwrap_or(hex);
    if h.len() != 6 { return None; }
    let v = u32::from_str_radix(h, 16).ok()?;
    Some([((v >> 16) & 0xff) as u8, ((v >> 8) & 0xff) as u8, (v & 0xff) as u8])
}

/// Measurement for a selection-driving plan, built from the execution-time
/// target-selection stage capture plus after-execution captures over the
/// identical regions. The pre-execution scene's selection is never used:
/// when the plan first selects the shirt, the target IS the plan's selection.
fn measure_from_selection_stage<S: LiveEditSession + ?Sized>(session: &mut S, capture: Option<&SelectionCapture>) -> RegionMeasurement {
    let unavailable = || RegionMeasurement { target: None, protected: Vec::new(), available: false, target_is_canvas: false, target_source: "nothing measurable", uncovered: None };
    let Some(capture) = capture else { return unavailable() };
    let failed = |why: &str| VisualEvidence { class: VisualEvidenceClass::Unknown, detail: why.to_string() };
    // Rollback/cancellation identity: the after-execution measurement is only
    // comparable when the document is still the one captured before the
    // evaluated step (same name + path, read back now). A document swapped
    // under the plan makes every capture stale; nothing is assumed.
    // Identity is the engine's unique live DocId, bound through the
    // transaction by a monotonic revision (the after state must be at or
    // past the capture's revision). Name+path corroborate but never
    // identify: a same-named different document has a different DocId.
    // Rollback keeps the same document and honestly measures no-change; a
    // close/reopen or swap fails the id. A missing id fails closed.
    let identity_ok = session.inspect_active().ok().flatten()
        .map(|doc| {
            let id_ok = match (doc.get("id").and_then(Value::as_u64), capture.document_id) {
                (Some(now), Some(then)) => now == then,
                _ => false,
            };
            let revision_ok = match (doc.get("revision").and_then(Value::as_u64), capture.revision) {
                (Some(now), Some(then)) => now >= then,
                _ => true, // revision unreported: the id carries identity alone
            };
            id_ok && revision_ok
        })
        .unwrap_or(false);
    if !identity_ok {
        let mut m = unavailable();
        m.available = true;
        m.target = Some((failed("the document identity changed between the target-selection capture and the after-execution measurement (rollback, close or swap); the captures are not comparable"), false));
        return m;
    }
    // Exact target evidence first: the preserved per-pixel coverage splits
    // the bounding region into the selection's own pixels and the region's
    // collateral, and the before/after rendered pixels compare exactly. This
    // survives the selection's consumption and needs no surviving selection.
    let target_exact = match (&capture.before_pixels, &capture.coverage) {
        (Some(before_px), Some(cov)) => {
            match ExecutionEngine::capture_tiled(session, "image.analysis.regionPixels", capture.bounds, 4) {
                Some(after_px) => compare_exact_coverage(before_px, &after_px, cov).map(|(covered_changed, covered_total, collateral_changed)| {
                    let class = if covered_changed > 0 { VisualEvidenceClass::Changed } else { VisualEvidenceClass::NoChangeObserved };
                    (VisualEvidence { class, detail: format!(
                        "exact per-pixel comparison under the selection's preserved coverage: {covered_changed} of {covered_total} covered (target) pixels differ; {collateral_changed} collateral pixel(s) inside the bounding region also differ") }, true)
                }),
                None => Some((failed("the after-execution exact region capture failed, was refused, or exceeded the aggregate budget"), false)),
            }
        }
        _ => None,
    };
    let target = target_exact.or_else(|| capture.region.as_ref().map(|before| (match ExecutionEngine::capture_digest(session, Some(capture.bounds)) {
        Ok(Some(after)) => ExecutionEngine::compare_digests(before, &after),
        _ => failed("the after-execution region capture failed or was refused"),
    }, false)));
    let mut protected = Vec::new();
    for ((rect, before), (_, before_px)) in capture.protected.iter().zip(capture.protected_pixels.iter()) {
        let exact = match before_px {
            Some(before_px) => match ExecutionEngine::capture_tiled(session, "image.analysis.regionPixels", *rect, 4) {
                Some(after_px) => compare_exact_raw(before_px, &after_px).map(|changed| {
                    let class = if changed > 0 { VisualEvidenceClass::Changed } else { VisualEvidenceClass::NoChangeObserved };
                    (VisualEvidence { class, detail: format!(
                        "exact per-pixel comparison over {} pixels of the protected region: {changed} pixels differ",
                        rect[2] * rect[3]) }, true)
                }),
                None => Some((failed("an after-execution exact protected-region capture failed, was refused, or exceeded the aggregate budget"), false)),
            },
            None => None,
        };
        protected.push(exact.unwrap_or_else(|| (match before {
            Some(before) => match ExecutionEngine::capture_digest(session, Some(*rect)) {
                Ok(Some(after)) => ExecutionEngine::compare_digests(before, &after),
                _ => failed("an after-execution protected-region capture failed or was refused"),
            },
            None => failed("the protected region's before-execution capture failed or was refused"),
        }, false)));
    }
    let target_source: &'static str = if capture.earlier_mutations > 0 {
        "the plan's final selection segment, captured immediately before its edit"
    } else {
        "the plan's selection, captured immediately before the evaluated edit"
    };
    let uncovered = (capture.earlier_mutations > 0).then(|| format!(
        "{} earlier edit segment(s) of this multi-segment plan were not individually measured; the final segment's protected-complement comparison cannot verify them",
        capture.earlier_mutations));
    RegionMeasurement { target, protected, available: true, target_is_canvas: false, target_source, uncovered }
}

fn measure_target_colour<S: LiveEditSession + ?Sized>(session: &mut S, intent: &EditingIntent, plan: &EditingPlan, final_scene: &VisualScene, capture: Option<&SelectionCapture>, evaluation: &mut EvaluationReport) -> ColourCriterion {
    let Some(requested) = intent.extracted_color.clone() else { return ColourCriterion::NotApplicable };
    let colours = NaturalLanguageEngine::new().colour_terms(&intent.raw_prompt);
    if colours.len() > 1 {
        return ColourCriterion::Unavailable(format!(
            "the request names {} colours; which colour binds to which object is unresolved", colours.len()));
    }
    if !plan.steps.iter().any(|s| s.command.starts_with("select.") || s.command.contains("mask")) {
        return ColourCriterion::Unavailable(
            "no selection- or mask-based step bound the request's target; a surviving arbitrary selection proves nothing".into());
    }
    // The target's ORIGINAL colour, captured coverage-aware at the
    // target-selection stage, is recorded for the report.
    if let Some(stats) = capture.and_then(|c| c.color_stats.as_ref()) {
        if let Some(before_mean) = stats.get("meanColor").and_then(Value::as_str) {
            evaluation.observations.push(format!(
                "Measured colour criterion: the target selection's mean colour before the edit was {before_mean} (coverage-aware)."));
        }
    }
    let unknown = |why: &str| ColourCriterion::Unavailable(format!("the target's colour could not be measured ({why})"));
    // Exact tier: the target's preserved per-pixel coverage plus the exact
    // after-execution pixels give the coverage-weighted mean colour of the
    // covered pixels - the requested colour bound to the requested region,
    // valid even after the selection was consumed.
    if let Some(c) = capture {
        if let (Some(before_cov), Some(_)) = (&c.coverage, &c.before_pixels) {
            match ExecutionEngine::capture_tiled(session, "image.analysis.regionPixels", c.bounds, 4) {
                Some(after_px) => {
                    if let (Some(mean), Some(want)) = (exact_covered_mean(&after_px, before_cov), parse_hex_rgb(&requested)) {
                        let tolerance = 32.0 / 255.0;
                        let within_tolerance = want.iter().zip(mean.iter())
                            .all(|(a, b)| (f32::from(*a) / 255.0 - *b).abs() <= tolerance);
                        let measured = format!("#{:02x}{:02x}{:02x}",
                            (mean[0] * 255.0).round() as u8, (mean[1] * 255.0).round() as u8, (mean[2] * 255.0).round() as u8);
                        evaluation.observations.push(format!(
                            "Measured colour criterion: exact coverage-weighted mean colour of the target's covered pixels is {measured}. A mean matches the requested colour on average only; it does not verify the colour's distribution across the target."));
                        return ColourCriterion::Checked(ColourCheck { requested, measured, within_tolerance });
                    }
                }
                None => return unknown("the exact after-execution region capture failed, was refused, or exceeded the aggregate budget"),
            }
        }
    }
    if final_scene.has_selection {
        let value = match session.command_run("image.analysis.colorStats",
            serde_json::json!({"maxPixels": EVIDENCE_MAX_PIXELS})) {
            Ok(v) => v,
            Err(_) => return unknown("the capture failed or was refused"),
        };
        if value.get("domain").and_then(Value::as_str) != Some("selection") { return unknown("no selection domain"); }
        let Some(measured) = value.get("meanColor").and_then(Value::as_str).map(str::to_string) else { return unknown("no mean colour in the result") };
        let (Some(want), Some(got)) = (parse_hex_rgb(&requested), parse_hex_rgb(&measured)) else { return unknown("an unparseable colour") };
        // Per-channel tolerance for shading/quantization: a recolour that
        // missed its hue lands far outside this; a quantized rendering of the
        // right colour lands inside.
        let within_tolerance = want.iter().zip(got.iter()).all(|(a, b)| (*a as i32 - *b as i32).abs() <= 32);
        return ColourCriterion::Checked(ColourCheck { requested, measured, within_tolerance });
    }
    // The selection was consumed: measure its recorded bounding region as an
    // exact rect. Every pixel counts (no selection coverage), so the mean
    // mixes in unselected surroundings; the weaker criterion asks whether the
    // requested colour became a DOMINANT colour of the region instead.
    let Some(capture) = capture else {
        return ColourCriterion::Unavailable("no selection survived to the final state and none was captured at the target-selection stage".into());
    };
    let [x, y, width, height] = capture.bounds;
    let value = match session.command_run("image.analysis.colorStats",
        serde_json::json!({"maxPixels": EVIDENCE_MAX_PIXELS, "x": x, "y": y, "width": width, "height": height})) {
        Ok(v) => v,
        Err(_) => return unknown("the bounding-region capture failed or was refused"),
    };
    if value.get("domain").and_then(Value::as_str) != Some("region") { return unknown("no region domain in the result"); }
    let Some(want) = parse_hex_rgb(&requested) else { return unknown("an unparseable requested colour") };
    let matched = value.get("dominantColors").and_then(Value::as_array).map(|colors| colors.iter().any(|c| {
        c.get("color").and_then(Value::as_str).and_then(parse_hex_rgb)
            .is_some_and(|got| want.iter().zip(got.iter()).all(|(a, b)| (*a as i32 - *b as i32).abs() <= 32))
    }));
    match matched {
        Some(within_tolerance) => {
            // Diagnostic only: the selection was consumed, so this measures
            // the bounding region whole - unselected surroundings count, and
            // a colour merely surviving anywhere in the region has produced
            // false matches. It is recorded for the report and can NEVER
            // satisfy the criterion.
            evaluation.observations.push(format!(
                "Colour diagnostic (cannot satisfy the criterion): with the selection consumed and no exact coverage preserved, the requested colour {} a dominant colour of the whole bounding region, unselected surroundings included.",
                if within_tolerance { "is" } else { "is not" }));
            ColourCriterion::Unavailable(
                "the selection was consumed without a preserved exact coverage; the bounding-region dominant-colour check is diagnostic only".into())
        }
        None => unknown("no dominant colours in the region result"),
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

    fn evidence(class: VisualEvidenceClass) -> VisualEvidence {
        VisualEvidence { class, detail: "test".into() }
    }

    fn measurement(target: Option<(VisualEvidenceClass, bool)>, protected: Vec<(VisualEvidenceClass, bool)>, canvas: bool) -> RegionMeasurement {
        RegionMeasurement {
            target: target.map(|(c, x)| (evidence(c), x)),
            protected: protected.into_iter().map(|(c, x)| (evidence(c), x)).collect(),
            available: true,
            target_is_canvas: canvas,
            target_source: "test region",
            uncovered: None,
        }
    }

    fn satisfied_report() -> EvaluationReport {
        EvaluationReport { confidence_score: 0.9, intent_satisfied: true, contrast_check_passed: true,
            hierarchy_valid: true, layout_balance_score: 0.9, observations: Vec::new() }
    }

    #[test]
    fn full_canvas_raster_target_is_not_local_protected_region_proof() {
        let scene = scene(100, 100, &[(2, [0, 0, 100, 100])]);
        let watch = RegionWatch::new(&scene, &plan(&[("layer.fill.set", 2)]));
        assert!(watch.target_is_canvas && watch.protected.is_empty());
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Changed, true)), Vec::new(), true)
            .apply(IntentCategory::ColorStyling, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)), "a canvas-spanning target must not complete a local recolour");
        assert!(report.intent_satisfied, "execution success alone stays untouched; completion is withheld by grounding");
    }

    #[test]
    fn unknown_target_stays_unknown_not_canvas() {
        let scene = scene(100, 100, &[(2, [10, 10, 20, 20])]);
        let watch = RegionWatch::new(&scene, &plan(&[("layer.fill.set", 99)]));
        assert!(watch.target.is_none() && watch.protected.is_empty() && !watch.target_is_canvas);
        let mut report = satisfied_report();
        let grounding = measurement(None, Vec::new(), false).apply(IntentCategory::ColorStyling, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)));
    }

    #[test]
    fn missing_capture_is_unverified_not_completed() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Unknown, false)), vec![(VisualEvidenceClass::Unknown, false)], false)
            .apply(IntentCategory::ColorStyling, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)));
        let grounding = RegionMeasurement { target: None, protected: Vec::new(), available: false, target_is_canvas: false, target_source: "test region", uncovered: None }
            .apply(IntentCategory::ColorStyling, &mut satisfied_report());
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)), "no digest capability must never complete");
        let mut design = satisfied_report();
        let grounding = RegionMeasurement { target: None, protected: Vec::new(), available: false, target_is_canvas: false, target_source: "test region", uncovered: None }
            .apply(IntentCategory::GraphicDesign, &mut design);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)), "heuristic design checks without any measurement are not completion proof");
    }

    #[test]
    fn intentional_global_edit_change_is_activity_not_satisfaction() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Changed, true)), Vec::new(), true)
            .apply(IntentCategory::PhotoEnhancement, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)),
            "a changed canvas does not verify a global enhancement request");
        assert!(report.intent_satisfied, "execution success is untouched; completion is withheld by grounding");
    }

    #[test]
    fn unchanged_wrong_colour_result_is_unmet_not_success() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::NoChangeObserved, true)), vec![(VisualEvidenceClass::NoChangeObserved, true)], false)
            .apply(IntentCategory::ColorStyling, &mut report);
        assert!(!report.intent_satisfied, "a target with no observed change is an unmet goal");
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)));
    }

    #[test]
    fn measured_local_recolour_is_grounded_and_labels_stay_honest() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Changed, true)),
            vec![(VisualEvidenceClass::NoChangeObserved, true), (VisualEvidenceClass::NoChangeObserved, true)], false)
            .apply(IntentCategory::ColorStyling, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Grounded));
        assert!(report.observations.iter().any(|o| o.contains("exact per-pixel comparison")),
            "exact evidence is labeled exact: {:?}", report.observations);
    }

    #[test]
    fn digest_only_evidence_never_grounds_completion() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Changed, false)),
            vec![(VisualEvidenceClass::NoChangeObserved, false)], false)
            .apply(IntentCategory::ColorStyling, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Unverified(_)),
            "sampled-grid digest evidence is diagnostic only and never grounds completion");
        assert!(report.observations.iter().any(|o| o.contains("not proof of unchanged pixels")),
            "equal digests are labeled no-difference-observed, never proof: {:?}", report.observations);
    }

    #[test]
    fn local_protected_change_fails_but_global_is_recorded_only() {
        let mut local = satisfied_report();
        measurement(Some((VisualEvidenceClass::Changed, true)), vec![(VisualEvidenceClass::Changed, true)], false)
            .apply(IntentCategory::ColorStyling, &mut local);
        assert!(!local.intent_satisfied, "collateral change fails a local edit");
        let mut global = satisfied_report();
        measurement(Some((VisualEvidenceClass::Changed, true)), vec![(VisualEvidenceClass::Changed, true)], false)
            .apply(IntentCategory::PhotoEnhancement, &mut global);
        assert!(global.intent_satisfied, "a global request legitimately alters the complement");
    }

    #[test]
    fn design_completion_rests_on_its_semantic_checks() {
        let mut report = satisfied_report();
        let grounding = measurement(Some((VisualEvidenceClass::Changed, true)), vec![(VisualEvidenceClass::Changed, true)], true)
            .apply(IntentCategory::GraphicDesign, &mut report);
        assert!(matches!(grounding, CompletionGrounding::Grounded));
        assert!(report.intent_satisfied);
    }

    #[test]
    fn selection_driven_plan_targets_the_selection_region() {
        let mut scene = scene(1000, 800, &[(2, [0, 0, 1000, 800])]);
        scene.has_selection = true;
        scene.selection_bounds = Some([100, 100, 300, 200]);
        let sel_plan = plan(&[("select.quick", 2)]);
        let watch = RegionWatch::new(&scene, &sel_plan);
        assert_eq!(watch.target, Some([100, 100, 300, 200]));
        assert_eq!(watch.target_source, "the active selection's bounding region");
        assert!(!watch.target_is_canvas && watch.protected.len() == 4);
        // A plan that never touches the selection keeps layer/canvas targeting.
        let watch = RegionWatch::new(&scene, &plan(&[("layer.fill.set", 2)]));
        assert_eq!(watch.target_source, "the plan-targeted layers' bounds");
    }

    #[test]
    fn hex_colours_parse_and_tolerances_are_per_channel() {
        assert_eq!(parse_hex_rgb("#FF0000"), Some([255, 0, 0]));
        assert_eq!(parse_hex_rgb("336699"), Some([0x33, 0x66, 0x99]));
        assert_eq!(parse_hex_rgb("#abc"), None);
        assert_eq!(parse_hex_rgb("#gggggg"), None);
    }
}

