//! Subsystem E: Native Transactional Execution
//!
//! Executes structured plans against headless sessions, records command traces, and performs
//! transactional rollback (compound undo) if any step in an editing plan fails.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{AiError, EditingPlan, LiveEditSession};
use crate::headless::Headless;

/// Observed-composite evidence class for one executed step. This records what a
/// before/after pixel comparison OBSERVED; it is never a semantic verdict on
/// whether the user's intended outcome was achieved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualEvidenceClass {
    /// The document dimensions changed, or the sampled quantized composite
    /// DATA differs on an identical grid. A data difference is not proof of a
    /// visible appearance change (RGB differences under full transparency
    /// change the data, not the appearance) and says nothing about intent or
    /// quality.
    Changed,
    /// No difference was observed on the sampled, 8-bit-clamped grid. Not
    /// proof of no change: stride sampling skips pixels, HDR and sub-8-bit
    /// differences are clamped or quantized away, and a 64-bit
    /// non-cryptographic hash can collide. An idempotent or no-op visual
    /// command legitimately lands here.
    NoChangeObserved,
    /// A non-visual command class: success never required a pixel change, so no
    /// comparison was made. This is a legitimate outcome, not a gap.
    NotApplicable,
    /// No evidence: the command is unclassified, the host has no digest
    /// capability, a capture failed or was refused, or the plan's capture budget
    /// was exhausted. Honest absence, never assumed change or no-change.
    #[default]
    Unknown,
}

/// Per-step visual result evidence: an observation record that never feeds
/// `success`, `is_success`, rollback or any safety decision.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VisualEvidence {
    pub class: VisualEvidenceClass,
    /// Human-readable basis for the class, including sampling parameters or the
    /// reason evidence is unavailable.
    pub detail: String,
}

/// One step's observed-composite evidence, paired with its plan step index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepVisualEvidence {
    pub step_index: usize,
    pub visual: VisualEvidence,
}

/// Observed-composite evidence collected while executing a plan, one entry
/// per executed step. Each entry is an observation taken AT THAT STEP'S TIME:
/// after a failure and its rollback the final document is the post-rollback
/// state, so a successful pre-rollback step's evidence does not describe the
/// final canvas (on the error path the whole bundle is dropped, so callers
/// never see pre-rollback observations of a canvas that rollback then
/// changed). Evidence is observational only: it never feeds `is_success`,
/// rollback, dispatch or any safety decision. It is returned only through the
/// additive `execute_plan*_with_evidence` API so that `StepExecution` and all
/// existing callers stay byte-compatible.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionEvidence {
    pub steps: Vec<StepVisualEvidence>,
}

/// Detailed trace of an executed plan step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepExecution {
    pub step_index: usize,
    pub command: String,
    pub params: Value,
    pub success: bool,
    pub output: Option<Value>,
    pub error: Option<String>,
}

/// Execution summary of an editing plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub total_steps: usize,
    pub executed_steps: usize,
    pub step_executions: Vec<StepExecution>,
    pub is_success: bool,
    pub error: Option<String>,
    pub undos_performed: usize,
}

/// Commands that create, open or close documents, changing which document is
/// active (grounded in the engine's job table: crates/engine/src/jobs.rs:687 and
/// jobs.rs `OPEN_JOB`, plus cmd.rs "file.close"). Transactional rollback cannot
/// attribute history entries across such a boundary - a same-name unsaved
/// document would defeat name/path comparison - so plans containing them are
/// refused up front. The one exception is a leading `file.new` in a
/// `creates_document` plan: the created document is plan-owned, its post-creation
/// state becomes the rollback baseline, and a failure report says it remains open.
const IDENTITY_CHANGING: &[&str] = &["file.new", "file.open", "file.close", "file.closeAll"];

// ------------------------------------------------------------------ visual result evidence

/// Per-capture composite render budget for evidence digests, matching the
/// conservative automatic-perception budget already used for color statistics
/// (`computed_dominant_colors`). This caps RENDERED pixels per capture (about
/// 1M are actually sampled at the resulting stride); the digest bounds the
/// composite buffer only (16 bytes per rendered pixel, at most 64,000,000
/// bytes here); renderer-internal intermediates and wall-clock time are
/// governed by the renderer itself.
const EVIDENCE_MAX_PIXELS: u64 = 4_000_000;

/// Total digest captures allowed per plan execution. Captures are synchronous
/// composite renders on the host's live path; the aggregate per run is bounded
/// at 8 x 4M = 32M rendered px, plus the perception path's 2 color-statistics
/// renders (at most 8M rendered px): about 40M rendered px, i.e. roughly
/// 640,000,000 decimal bytes (~610 MiB) of CUMULATIVE composite output per run
/// - not peak memory (one bounded composite buffer at a time, plus renderer
/// internals, which this does not bound) and no wall-clock or latency claim.
/// k visual steps chain k+1 captures; once the budget is exhausted further
/// visual steps get honest `Unknown` evidence instead of unbounded render
/// cost.
const EVIDENCE_MAX_CAPTURES: usize = 8;

/// Commands whose success is EXPECTED to change the rendered composite, each
/// verified against the engine registry (registered, `journal: true`).
/// Heuristic for evidence capture only: a command listed here that observably
/// does not change pixels (idempotent or no-op) is still a successful step,
/// classified `NoChangeObserved`. The `image.adjustments.*` and
/// `layer.newAdjustmentLayer.*` prefixes are grounded in the ADJ table
/// generator in cmd.rs, which registers every such id with `journal: true`.
const VISUAL_COMMANDS: &[&str] = &[
    "type.create",
    "type.edit",
    "type.setStyle",
    "layer.newFillLayer.solidColor",
    "layer.newFillLayer.gradient",
    "layer.duplicate",
    "layer.removeBackground",
    "filter.sharpen.unsharpMask",
    "filter.blur.gaussianBlur",
    "edit.undo",
    "edit.redo",
];

/// Read-only query commands (registered, `journal: false`): success never
/// involves a pixel change. `image.adjustments.colorLookup.list` is the one
/// non-journaling id under its prefix.
const NONVISUAL_COMMANDS: &[&str] = &["image.analysis.colorStats", "image.analysis.pixelDigest", "image.adjustments.colorLookup.list"];

/// Evidence classification for a plan step's command.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EvidenceKind {
    Visual,
    Nonvisual,
    Unclassified,
}

/// One digest capture: schema-validated geometry + sampling parameters + the
/// digest itself. The host's JSON is unauthenticated input - a custom host's
/// output inherits no engine consistency - so every field of the fixed schema
/// must be present, well-formed and self-consistent before the capture is
/// trusted.
#[derive(Clone, Debug, PartialEq)]
struct DigestMark {
    /// The digest as a NUMBER (parsed from the host's hex), so textually
    /// different but numerically identical digests (e.g. case differences)
    /// can never register as a change.
    digest: u64,
    width: u64,
    height: u64,
    stride: u64,
    sampled: u64,
}

/// Observed identity hint and history state of the active document, read through
/// the host's own inspection JSON (`fotodit_engine::inspect::document`). Every
/// field must be present and well-formed; anything missing or malformed makes
/// the mark unreadable, and an unreadable mark closes rollback rather than
/// assuming.
///
/// PRECONDITION, not a guarantee: name+path is an identity HINT. A document
/// replaced by a same-name/same-path document whose history labels also collide
/// is unobservable through this interface. This code certifies behavior only
/// under a synchronous host that keeps the active document stable for the whole
/// operation (the LiveEditSession contract) and resolves commands through the
/// engine registry; a hostile or non-conforming host is not certified.
#[derive(Clone, Debug, PartialEq)]
struct DocMark {
    name: String,
    path: Option<String>,
    /// History panel entry labels (undo stack + current state): entries().len()
    /// is past_len + 1, and each undo removes exactly one entry
    /// (crates/ops/src/lib.rs History::entries/undo).
    entries: Vec<String>,
}

/// Execution Engine responsible for atomic command dispatch and rollback.
pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Evidence classification of a plan step's command. Prefix membership is
    /// verified against the LIVE REGISTRY (a registered spec with
    /// `journal: true`), so an invented id under a visual prefix - say an
    /// unregistered `image.adjustments.*` - classifies Unclassified:
    /// unknown-safe, no admission expansion. Anything not grounded as visual
    /// or read-only is Unclassified: unobserved, never assumed.
    fn evidence_class(command: &str) -> EvidenceKind {
        if NONVISUAL_COMMANDS.contains(&command) {
            return EvidenceKind::Nonvisual;
        }
        let candidate = VISUAL_COMMANDS.contains(&command)
            || command.starts_with("image.adjustments.")
            || command.starts_with("layer.newAdjustmentLayer.");
        if candidate
            && fotodit_engine::command_specs()
                .iter()
                .any(|s| s.id == command && s.journal)
        {
            return EvidenceKind::Visual;
        }
        EvidenceKind::Unclassified
    }

    /// One digest capture through the host's OPTIONAL observation hook.
    /// `Ok(None)`: the host offers no digest capability. `Err`: the host
    /// supports it but this capture failed or was refused, or its output
    /// failed the fixed-schema validation below. Never fails the step it
    /// observes. The hook contract forbids mutation and requires the active
    /// document to be preserved (see `LiveEditSession::visual_digest`).
    fn capture_digest<S: LiveEditSession + ?Sized>(session: &mut S) -> Result<Option<DigestMark>, String> {
        let Some(result) = session.visual_digest(EVIDENCE_MAX_PIXELS) else {
            return Ok(None);
        };
        let value = result?;
        let invalid = || "the host's digest output failed schema validation".to_string();
        if value.get("domain").and_then(Value::as_str) != Some("canvas")
            || value.get("schema").and_then(Value::as_u64) != Some(1)
            || value.get("hash").and_then(Value::as_str) != Some("fnv1a64")
            || value.get("quantization").and_then(Value::as_str) != Some("8bit/channel")
        {
            return Err(invalid());
        }
        let digest_hex = value.get("digest").and_then(Value::as_str).ok_or_else(invalid)?;
        if digest_hex.len() != 16 || !digest_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid());
        }
        let digest = u64::from_str_radix(digest_hex, 16).map_err(|_| invalid())?;
        let width = value.get("width").and_then(Value::as_u64).filter(|v| *v >= 1).ok_or_else(invalid)?;
        let height = value.get("height").and_then(Value::as_u64).filter(|v| *v >= 1).ok_or_else(invalid)?;
        let stride = value.get("stride").and_then(Value::as_u64).filter(|v| *v >= 1).ok_or_else(invalid)?;
        let sampled = value.get("sampledPixels").and_then(Value::as_u64).ok_or_else(invalid)?;
        // Declared geometry must not overflow and must stay within the capture
        // budget we requested; actual resource use remains a host precondition
        // (the hook contract), not something this JSON can prove.
        let area = width.checked_mul(height).ok_or_else(invalid)?;
        if area > EVIDENCE_MAX_PIXELS {
            return Err(invalid());
        }
        let cols = (width - 1) / stride + 1;
        let rows = (height - 1) / stride + 1;
        if cols.checked_mul(rows) != Some(sampled) {
            return Err(invalid());
        }
        Ok(Some(DigestMark { digest, width, height, stride, sampled }))
    }

    /// Compare two captures. A dimension change is an observable change of
    /// its own (geometry evidence; digests are never compared). A stride
    /// change at identical dimensions is NOT a pixel change: the grids
    /// differ, so the digests are incomparable and the honest class is
    /// Unknown. Digests are only compared over an identical grid.
    fn compare_digests(before: &DigestMark, after: &DigestMark) -> VisualEvidence {
        if (before.width, before.height) != (after.width, after.height) {
            return VisualEvidence {
                class: VisualEvidenceClass::Changed,
                detail: format!(
                    "the document dimensions changed ({}x{} -> {}x{}); a geometry change is an observable change of its own and digests are not compared",
                    before.width, before.height, after.width, after.height
                ),
            };
        }
        if before.stride != after.stride {
            return VisualEvidence {
                class: VisualEvidenceClass::Unknown,
                detail: format!(
                    "the sampling grid changed (stride {} -> {}) at identical dimensions; the digests are not comparable, so there is no evidence of change or of no-change",
                    before.stride, after.stride
                ),
            };
        }
        if before.digest != after.digest {
            VisualEvidence {
                class: VisualEvidenceClass::Changed,
                detail: format!(
                    "the sampled quantized composite data differs ({}x{}, stride {}, {} px sampled); a data difference is not proof of a visible appearance change, intent or quality",
                    after.width, after.height, after.stride, after.sampled
                ),
            }
        } else {
            VisualEvidence {
                class: VisualEvidenceClass::NoChangeObserved,
                detail: format!(
                    "no difference observed on the sampled, 8-bit-clamped grid ({}x{}, stride {}, {} px sampled); sampling, quantization and hash collisions can hide changes",
                    after.width, after.height, after.stride, after.sampled
                ),
            }
        }
    }

    /// Execute an editing plan step-by-step with automatic transactional rollback on failure.
    pub fn execute_plan(headless: &mut Headless, plan: &EditingPlan) -> Result<ExecutionResult, AiError> {
        Self::execute_plan_in(headless, plan)
    }

    /// `execute_plan` plus the observed-composite evidence bundle (additive;
    /// the evidence never changes execution, see [`ExecutionEvidence`]).
    pub fn execute_plan_with_evidence(
        headless: &mut Headless,
        plan: &EditingPlan,
    ) -> Result<(ExecutionResult, ExecutionEvidence), AiError> {
        Self::execute_plan_in_with_evidence(headless, plan)
    }

    /// Execute on the host's admitted command path, including rollback attempts.
    ///
    /// Rollback is fail-closed and prefix-verified, never assumption-based:
    /// - every successful step is verified against the engine's command registry
    ///   (`CommandSpec::journal`): a journaling command must append exactly one
    ///   History entry with the prior entry list kept as an exact prefix (a
    ///   replaced or reordered list is foreign history, not ours); a
    ///   non-journaling command must leave the entry list unchanged
    ///   (non-journaling commands may still change state outside history - view
    ///   state, session state, saved state - which this verification does not
    ///   cover);
    /// - the document identity hint (name + path) must stay stable across the
    ///   plan;
    /// - any step that cannot be verified - unknown command, unreadable or
    ///   malformed inspection, identity change, unexpected history delta
    ///   (coalescing, capacity trim, visibility edits when those are not
    ///   undoable, foreign replacement) - closes rollback entirely: no blind
    ///   undo is ever issued, because reaching verified entries would require
    ///   undoing through unverified ones;
    /// - a failed step is reconciled by inspection with the same prefix rule:
    ///   it may have journaled an entry before erroring (mutate-then-Err), and
    ///   that entry is rolled back first, accounted to the failed step rather
    ///   than to earlier successes;
    /// - every undo is followed by an inspection that must show the identity
    ///   hint and the exact expected retained entry list (pre-plan baseline
    ///   plus the not-yet-undone plan entries); an undo is counted only after
    ///   that exact postcondition passes, so a no-op undo is reported as zero
    ///   verified undos.
    ///
    /// The guarantee is scoped to the verified history boundary: matching entry
    /// lists after rollback prove the undo stack returned to its pre-plan
    /// entries, not that pixels, metadata, selection, view state or other
    /// non-journal state equal the pre-plan document. A creation plan is never
    /// reported complete: undoing its edits leaves the created document open,
    /// which differs from the original state. It is not a universal atomic
    /// transaction for non-history or cross-document commands.
    pub fn execute_plan_in<S: LiveEditSession + ?Sized>(
        session: &mut S,
        plan: &EditingPlan,
    ) -> Result<ExecutionResult, AiError> {
        // Strict opt-in: existing callers keep the old behavior exactly -
        // no evidence captures, so no additional renders.
        Self::execute_plan_in_impl(session, plan, false).map(|(result, _)| result)
    }

    /// `execute_plan_in` plus the observed-composite evidence bundle. The
    /// bundle is observational only and never feeds `is_success`, rollback or
    /// dispatch; on the error path it is dropped, so callers never see
    /// pre-rollback observations of a canvas that rollback then changed.
    /// Evidence captures (bounded by `EVIDENCE_MAX_CAPTURES` x
    /// `EVIDENCE_MAX_PIXELS`) happen ONLY through this entry point.
    pub fn execute_plan_in_with_evidence<S: LiveEditSession + ?Sized>(
        session: &mut S,
        plan: &EditingPlan,
    ) -> Result<(ExecutionResult, ExecutionEvidence), AiError> {
        Self::execute_plan_in_impl(session, plan, true)
    }

    fn execute_plan_in_impl<S: LiveEditSession + ?Sized>(
        session: &mut S,
        plan: &EditingPlan,
        collect_evidence: bool,
    ) -> Result<(ExecutionResult, ExecutionEvidence), AiError> {
        // Identity-changing commands cannot be rolled back soundly; refuse plans
        // that contain them before anything executes (a leading file.new of a
        // creates_document plan excepted, documented above).
        for (idx, step) in plan.steps.iter().enumerate() {
            if IDENTITY_CHANGING.contains(&step.command.as_str())
                && !(idx == 0 && step.command == "file.new" && plan.creates_document)
            {
                return Err(AiError::ExecutionError {
                    step: idx,
                    command: step.command.clone(),
                    error: format!(
                        "the plan step '{}' changes which document is active, so the plan cannot be executed transactionally",
                        step.command
                    ),
                });
            }
        }
        // Early, user-readable state checks, not a safety or authorization boundary.
        // Every command still goes through the host's authoritative admission path.
        if plan.category == super::IntentCategory::PhotoEnhancement {
            let document = session.inspect_active()?.ok_or(AiError::NoDocumentOpen)?;
            super::enhancement_preflight::validate(&document, plan)?;
        }
        let mut executed = Vec::new();
        let mut evidence: Vec<StepVisualEvidence> = Vec::new();
        let total_steps = plan.steps.len();
        let mut baseline: Option<DocMark> = None;
        // Entries verified as appended by this plan's steps, oldest first.
        let mut plan_labels: Vec<String> = Vec::new();
        // Real ids of layers this plan's steps created, by step index: the
        // resolution source for intra-plan `{"$step": N}` references.
        let mut created_ids: Vec<Option<u64>> = vec![None; total_steps];
        // The first reason rollback became unsafe; once set, no undo is issued.
        let mut closed: Option<String> = None;
        // A creation plan leaves the created document open even when all its
        // journaled edits are undone: incomplete relative to the original state.
        let mut created_document_open = false;
        let mut mark_prev = Self::document_mark(session);
        // Visual-evidence state: the latest captured digest, chained so the next
        // visual step reuses it as its baseline (k+1 captures for k visual
        // steps). An unclassified step breaks the chain: the next visual step
        // re-baselines instead of attributing an unobserved step's changes.
        let mut last_digest: Option<DigestMark> = None;
        let mut evidence_captures = 0usize;

        for (idx, step) in plan.steps.iter().enumerate() {
            // The leading file.new of a creation plan: the created document is
            // plan-owned, so its post-creation state is the rollback baseline.
            if idx == 0 && step.command == "file.new" && plan.creates_document {
                match session.command_run(&step.command, step.params.clone()) {
                    Ok(output) => {
                        let after = Self::document_mark(session);
                        if after.is_none() {
                            closed = Some("the created document could not be inspected".to_string());
                        }
                        created_document_open = after.is_some();
                        baseline = after.clone();
                        mark_prev = after;
                        executed.push(StepExecution {
                            step_index: idx,
                            command: step.command.clone(),
                            params: step.params.clone(),
                            success: true,
                            output: Some(output),
                            error: None,
                        });
                        if collect_evidence {
                            evidence.push(StepVisualEvidence {
                                step_index: idx,
                                visual: VisualEvidence {
                                    class: VisualEvidenceClass::NotApplicable,
                                    detail: "document creation is observed through inspection; the first visual step captures its own baseline".to_string(),
                                },
                            });
                        }
                        continue;
                    }
                    Err(err) => {
                        let err_msg = err.to_string();
                        executed.push(StepExecution {
                            step_index: idx,
                            command: step.command.clone(),
                            params: step.params.clone(),
                            success: false,
                            output: None,
                            error: Some(err_msg.clone()),
                        });
                        if collect_evidence {
                            evidence.push(StepVisualEvidence {
                                step_index: idx,
                                visual: VisualEvidence {
                                    class: VisualEvidenceClass::NotApplicable,
                                    detail: "document creation is observed through inspection; the first visual step captures its own baseline".to_string(),
                                },
                            });
                        }
                        // Reconcile mutate-then-Err on creation too. Only an
                        // inspection showing the identical pre-step mark verifies
                        // that nothing was applied; no readable mark at all means
                        // the effect is unknown, not absent.
                        let note = match (Self::document_mark(session), &mark_prev) {
                            (Some(after), Some(before)) if after == *before => {
                                "nothing was applied; no rollback needed".to_string()
                            }
                            (None, None) => {
                                "the creation step's effect could not be verified; no rollback performed".to_string()
                            }
                            _ => "the failed creation step may have left a document open; no rollback performed".to_string(),
                        };
                        return Err(AiError::ExecutionError {
                            step: idx,
                            command: step.command.clone(),
                            error: format!("{err_msg} ({note})"),
                        });
                    }
                }
            }

            if baseline.is_none() {
                baseline = mark_prev.clone();
                if baseline.is_none() {
                    closed = Some("the document could not be inspected before the plan started".to_string());
                }
            }
            let spec = fotodit_engine::command_specs().iter().find(|s| s.id == step.command);

            // Visual result evidence (observation only): baseline for a visual
            // step before it runs. Never affects success, rollback or dispatch.
            // With collect_evidence = false (the legacy entry points) none of
            // this runs: no classification, no captures, no renders.
            let evidence_kind = if collect_evidence { Self::evidence_class(&step.command) } else { EvidenceKind::Nonvisual };
            let mut before_digest: Option<DigestMark> = None;
            let mut before_note: Option<String> = None;
            match evidence_kind {
                EvidenceKind::Visual => {
                    if let Some(mark) = &last_digest {
                        before_digest = Some(mark.clone());
                    } else if evidence_captures >= EVIDENCE_MAX_CAPTURES {
                        before_note = Some(format!("the plan's visual-evidence capture budget ({EVIDENCE_MAX_CAPTURES}) is exhausted; skipped to bound render cost"));
                    } else {
                        evidence_captures += 1;
                        match Self::capture_digest(session) {
                            Ok(Some(mark)) => {
                                before_digest = Some(mark.clone());
                                last_digest = Some(mark);
                            }
                            Ok(None) => before_note = Some("the host offers no pixel digest capability".to_string()),
                            Err(e) => before_note = Some(format!("the baseline digest capture failed: {e}")),
                        }
                    }
                }
                EvidenceKind::Nonvisual => {}
                // An unobserved step breaks the comparison chain: the next
                // visual step re-baselines instead of attributing changes.
                EvidenceKind::Unclassified => last_digest = None,
            }

            // Resolve intra-plan references (`{"$step": N}`) to the real ids
            // of layers created by earlier steps of this plan. Contract
            // validation admits only backward references to layer-creating
            // steps, but a reference can still fail to resolve at runtime when
            // a creation step's output is absent or unexpected. That failure is
            // routed through the same recorded-failure, reconciliation and
            // rollback path as a refused command: it is reported with the
            // partial-change accounting, and earlier applied steps are undone
            // by the standard machinery below. The recorded step keeps the
            // plan's original symbolic params.
            let resolution: Result<Value, crate::AutomationError> = (|| {
                let mut resolved_params = step.params.clone();
                if let Some(map) = resolved_params.as_object_mut() {
                    for key in ["layer", "target"] {
                        let referenced = map.get(key)
                            .and_then(Value::as_object)
                            .filter(|o| o.len() == 1)
                            .and_then(|o| o.get("$step"))
                            .and_then(Value::as_u64);
                        if let Some(n) = referenced {
                            let resolved = n.checked_sub(1)
                                .and_then(|i| created_ids.get(i as usize))
                                .copied()
                                .flatten();
                            match resolved {
                                Some(id) => { map.insert(key.to_string(), json!(id)); }
                                None => return Err(crate::AutomationError::BadRequest(format!(
                                    "the step references a layer created by step {n}, but that step created no layer; the step was not dispatched"
                                ))),
                            }
                        }
                    }
                }
                Ok(resolved_params)
            })();

            match resolution.and_then(|params| session.command_run(&step.command, params)) {
                Ok(output) => {
                    if let Some(id) = output.get("layer").and_then(Value::as_u64) {
                        created_ids[idx] = Some(id);
                    }
                    let after = Self::document_mark(session);
                    if closed.is_none() {
                        match (spec, &mark_prev, &after) {
                            (Some(spec), Some(prev), Some(after))
                                if prev.name == after.name && prev.path == after.path =>
                            {
                                // Prefix rule: a journaling step must keep the prior
                                // entries as an exact prefix and append exactly one;
                                // a non-journaling step must not touch the list.
                                if spec.journal {
                                    if after.entries.len() == prev.entries.len() + 1
                                        && after.entries[..prev.entries.len()] == prev.entries[..]
                                    {
                                        plan_labels.push(after.entries[prev.entries.len()].clone());
                                    } else {
                                        closed = Some(format!(
                                            "step {} ('{}') did not append exactly one history entry ({} -> {} entries)",
                                            idx, step.command, prev.entries.len(), after.entries.len()
                                        ));
                                    }
                                } else if after.entries != prev.entries {
                                    closed = Some(format!(
                                        "step {} ('{}') is non-journaling but changed the history entries",
                                        idx, step.command
                                    ));
                                }
                            }
                            (Some(_), Some(_), Some(_)) => {
                                closed = Some(format!("step {} ('{}') changed the active document", idx, step.command));
                            }
                            (Some(_), _, _) => {
                                closed = Some(format!("step {} ('{}') could not be inspected", idx, step.command));
                            }
                            (None, _, _) => {
                                closed = Some(format!("step {} ('{}') is not a registered engine command, so its effect is unverifiable", idx, step.command));
                            }
                        }
                    }
                    mark_prev = after;
                    executed.push(StepExecution {
                        step_index: idx,
                        command: step.command.clone(),
                        params: step.params.clone(),
                        success: true,
                        output: Some(output),
                        error: None,
                    });
                    if collect_evidence {
                        let visual = match evidence_kind {
                            EvidenceKind::Nonvisual => VisualEvidence {
                                class: VisualEvidenceClass::NotApplicable,
                                detail: "a non-visual command class; success does not require a pixel change".to_string(),
                            },
                            EvidenceKind::Unclassified => VisualEvidence {
                                class: VisualEvidenceClass::Unknown,
                                detail: "the command is not in the visual-evidence classification table; its visual effect is unobserved".to_string(),
                            },
                            EvidenceKind::Visual => {
                                if evidence_captures >= EVIDENCE_MAX_CAPTURES {
                                    VisualEvidence {
                                        class: VisualEvidenceClass::Unknown,
                                        detail: format!("the plan's visual-evidence capture budget ({EVIDENCE_MAX_CAPTURES}) is exhausted; skipped to bound render cost"),
                                    }
                                } else {
                                    evidence_captures += 1;
                                    let after_outcome = Self::capture_digest(session);
                                    match (before_digest, after_outcome) {
                                        (Some(before), Ok(Some(mark))) => {
                                            last_digest = Some(mark.clone());
                                            Self::compare_digests(&before, &mark)
                                        }
                                        // A fresh capture still seeds the chain for the next
                                        // visual step; only the comparison is missing.
                                        (None, Ok(Some(mark))) => {
                                            last_digest = Some(mark);
                                            VisualEvidence {
                                                class: VisualEvidenceClass::Unknown,
                                                detail: match before_note {
                                                    Some(note) => format!("no baseline to compare against: {note}"),
                                                    None => "no baseline to compare against".to_string(),
                                                },
                                            }
                                        }
                                        // The first failure is the honest reason; a later
                                        // failure must not mask it.
                                        (_, Ok(None)) => {
                                            last_digest = None;
                                            VisualEvidence {
                                                class: VisualEvidenceClass::Unknown,
                                                detail: before_note.unwrap_or_else(|| "the host offers no pixel digest capability".to_string()),
                                            }
                                        }
                                        (_, Err(e)) => {
                                            last_digest = None;
                                            VisualEvidence {
                                                class: VisualEvidenceClass::Unknown,
                                                detail: before_note.unwrap_or_else(|| format!("the post-step digest capture failed: {e}")),
                                            }
                                        }
                                    }
                                }
                            }
                        };
                        evidence.push(StepVisualEvidence { step_index: idx, visual });
                    }
                }
                Err(err) => {
                    let err_msg = err.to_string();
                    executed.push(StepExecution {
                        step_index: idx,
                        command: step.command.clone(),
                        params: step.params.clone(),
                        success: false,
                        output: None,
                        error: Some(err_msg.clone()),
                    });
                    if collect_evidence {
                        evidence.push(StepVisualEvidence {
                            step_index: idx,
                            visual: VisualEvidence {
                                class: VisualEvidenceClass::Unknown,
                                detail: "the step failed; its effect is reconciled by the rollback inspection, not by pixel evidence, and the evidence bundle is dropped on this error path".to_string(),
                            },
                        });
                    }

                    // Reconcile the failed step itself with the same prefix rule:
                    // an entry it verifiably appended before erroring is rolled
                    // back first and accounted to this step, not earlier successes.
                    if closed.is_none() {
                        let after_failure = Self::document_mark(session);
                        match (&mark_prev, &after_failure) {
                            (Some(prev), Some(after))
                                if prev.name == after.name && prev.path == after.path =>
                            {
                                if after.entries == prev.entries {
                                    // Verified: the failure journaled nothing.
                                } else if after.entries.len() == prev.entries.len() + 1
                                    && after.entries[..prev.entries.len()] == prev.entries[..]
                                {
                                    plan_labels.push(after.entries[prev.entries.len()].clone());
                                } else {
                                    closed = Some(format!(
                                        "the failed step left history in an unverifiable state ({} -> {} entries)",
                                        prev.entries.len(), after.entries.len()
                                    ));
                                }
                            }
                            _ => closed = Some("the failed step's effect could not be verified".to_string()),
                        }
                    }

                    let budget = plan_labels.len();
                    let mut rollback_count = 0usize;
                    let mut undo_attempts = 0usize;
                    let mut stop_reason: Option<String> = closed.clone();
                    if stop_reason.is_none() {
                        if let Some(baseline) = baseline.clone() {
                            while rollback_count < budget {
                                undo_attempts += 1;
                                if session.command_run("edit.undo", json!({})).is_err() {
                                    stop_reason = Some("an undo command was refused by the host".to_string());
                                    break;
                                }
                                // Count only after the exact postcondition: identity
                                // hint plus the expected retained entry list
                                // (baseline plus the not-yet-undone plan entries).
                                let mut expected = baseline.entries.clone();
                                expected.extend_from_slice(&plan_labels[..budget - rollback_count - 1]);
                                match Self::document_mark(session) {
                                    Some(m)
                                        if m.name == baseline.name
                                            && m.path == baseline.path
                                            && m.entries == expected =>
                                    {
                                        rollback_count += 1;
                                    }
                                    other => {
                                        stop_reason = Some(format!(
                                            "undo verification failed (expected {} retained entries, observed {}; identity hint preserved: {})",
                                            expected.len(),
                                            other.as_ref().map_or("unknown".to_string(), |m| m.entries.len().to_string()),
                                            other.as_ref().is_some_and(|m| m.name == baseline.name && m.path == baseline.path)
                                        ));
                                        break;
                                    }
                                }
                            }
                            if stop_reason.is_none() && created_document_open {
                                stop_reason = Some("the plan-created document remains open (creation is not a history entry)".to_string());
                            }
                        } else {
                            stop_reason = Some("no verified pre-plan history baseline exists".to_string());
                        }
                    }

                    let note = match stop_reason {
                        None => format!("rolled back {rollback_count} entr(y/ies); the history boundary is verified at its pre-plan entries (state outside history is not covered)"),
                        Some(reason) => {
                            let mut note = format!(
                                "rollback incomplete: rolled back {rollback_count} of {budget} applied entr(y/ies); {reason}"
                            );
                            if undo_attempts > rollback_count {
                                note.push_str(&format!("; {} undo attempt(s) had no verifiable effect", undo_attempts - rollback_count));
                            }
                            note
                        }
                    };
                    return Err(AiError::ExecutionError {
                        step: idx,
                        command: step.command.clone(),
                        error: format!("{err_msg} ({note})"),
                    });
                }
            }
        }

        Ok((
            ExecutionResult {
                total_steps,
                executed_steps: executed.len(),
                step_executions: executed,
                is_success: true,
                error: None,
                undos_performed: 0,
            },
            ExecutionEvidence { steps: evidence },
        ))
    }

    /// Active document identity hint and full history entry list via the host's
    /// inspection path. `None` unless every required field is present and
    /// well-formed: name must be a string, path a string or null, and history an
    /// array of strings. Callers treat `None` as unverifiable, never as
    /// permission to undo.
    fn document_mark<S: LiveEditSession + ?Sized>(session: &S) -> Option<DocMark> {
        let doc = session.inspect_active().ok().flatten()?;
        let name = doc.get("name").and_then(Value::as_str)?.to_string();
        let path = match doc.get("path") {
            Some(Value::String(p)) => Some(p.clone()),
            Some(Value::Null) => None,
            _ => return None,
        };
        let entries = doc
            .get("history")
            .and_then(Value::as_array)?
            .iter()
            .map(|e| e.as_str().map(str::to_string))
            .collect::<Option<Vec<String>>>()?;
        Some(DocMark { name, path, entries })
    }
}
