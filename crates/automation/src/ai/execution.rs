//! Subsystem E: Native Transactional Execution
//!
//! Executes structured plans against headless sessions, records command traces, and performs
//! transactional rollback (compound undo) if any step in an editing plan fails.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::provider_bridge::is_layer_creating;
use super::{AiError, EditingPlan, LiveEditSession, PlanStep};
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
    /// The plan's target selection, captured right after its LAST selection-
    /// or mask-driving step ran: the actual target-selection stage, before a
    /// following edit could consume the selection. A selection that existed
    /// before the plan is never mistaken for the plan's target. None when the
    /// plan drives no selection/mask step, the selection is empty at that
    /// point, or the host lacks the capability.
    #[serde(default)]
    pub selection_capture: Option<SelectionCapture>,
}

/// What was true at the plan's target-selection stage: the selection's
/// content bounds, a digest over that bounding region (includes unselected
/// pixels within it), the canvas-complement border digests (the protected
/// regions' BEFORE state), and coverage-aware colour statistics of the
/// selection itself. All observational; none of it feeds success, rollback
/// or dispatch.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelectionCapture {
    /// The step the capture was taken IMMEDIATELY BEFORE: the operation
    /// whose effect is being evaluated.
    pub step_index: usize,
    /// Document identity at capture: the engine's unique live DocId plus
    /// the revision, bound through the transaction. Name+path (and history
    /// labels, which a same-named different document can share) are kept as
    /// corroborating context only, never as identity.
    pub document_id: Option<u64>,
    pub document_name: String,
    pub document_path: Option<String>,
    /// Mutation steps before the captured segment (multi-segment plans):
    /// their effects are measured only via the protected-complement
    /// comparison, never attributed to this capture's target.
    pub earlier_mutations: usize,
    /// Selection content bounds [x, y, width, height] at capture time.
    pub bounds: [i64; 4],
    pub revision: Option<u64>,
    /// Exact per-pixel coverage of the selection over `bounds` (row-major
    /// f32, tiled under the engine's per-call cap, 2M-px aggregate): the
    /// target's exact coverage, preserved locally so it survives the
    /// selection's consumption.
    pub coverage: Option<Vec<f32>>,
    /// Exact rendered pixels of `bounds` before the edit (row-major RGBA
    /// f32, same tiling and budget).
    pub before_pixels: Option<Vec<f32>>,
    pub region: Option<DigestMark>,
    /// Up to four border rects (top, bottom, left, right) with their digests.
    pub protected: Vec<([i64; 4], Option<DigestMark>)>,
    /// Exact rendered pixels of each border rect before the edit.
    pub protected_pixels: Vec<([i64; 4], Option<Vec<f32>>)>,
    /// Selection-domain colour statistics (engine JSON shape), when captured.
    pub color_stats: Option<Value>,
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
pub(crate) const EVIDENCE_MAX_PIXELS: u64 = 4_000_000;

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
const NONVISUAL_COMMANDS: &[&str] = &[
    "image.analysis.colorStats", "image.analysis.pixelDigest",
    "image.analysis.regionPixels", "image.analysis.selectionCoverage",
    "image.adjustments.colorLookup.list"];

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DigestMark {
    /// The digest as a NUMBER (parsed from the host's hex), so textually
    /// different but numerically identical digests (e.g. case differences)
    /// can never register as a change.
    pub digest: u64,
    /// Canvas origin of the captured region; (0, 0) for a whole-canvas
    /// capture. Digests are only ever compared over an identical rect.
    pub x: i64,
    pub y: i64,
    pub width: u64,
    pub height: u64,
    pub stride: u64,
    pub sampled: u64,
    /// Engine document revision the capture was taken at, when the host
    /// reports it: binds the evidence to a document state lineage.
    pub revision: Option<u64>,
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
    pub(crate) fn capture_digest<S: LiveEditSession + ?Sized>(session: &mut S, region: Option<[i64; 4]>) -> Result<Option<DigestMark>, String> {
        let Some(result) = session.visual_digest(EVIDENCE_MAX_PIXELS, region) else {
            return Ok(None);
        };
        let value = result?;
        let invalid = || "the host's digest output failed schema validation".to_string();
        if value.get("domain").and_then(Value::as_str) != Some(if region.is_some() { "region" } else { "canvas" })
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
        let revision = value.get("revision").and_then(Value::as_u64);
        let (x, y) = if region.is_some() {
            (value.get("x").and_then(Value::as_i64).ok_or_else(invalid)?,
             value.get("y").and_then(Value::as_i64).ok_or_else(invalid)?)
        } else { (0, 0) };
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
        Ok(Some(DigestMark { digest, x, y, width, height, stride, sampled, revision }))
    }

    /// Parse an engine regionPixels result into (x, y, width, height, pixels).
    pub(crate) fn parse_exact_region(value: &Value) -> Option<(i64, i64, u64, u64, Vec<f32>)> {
        if value.get("format").and_then(Value::as_str) != Some("rgba-f32-rowmajor") { return None; }
        let x = value.get("x").and_then(Value::as_i64)?;
        let y = value.get("y").and_then(Value::as_i64)?;
        let width = value.get("width").and_then(Value::as_u64).filter(|v| *v >= 1)?;
        let height = value.get("height").and_then(Value::as_u64).filter(|v| *v >= 1)?;
        let pixels: Vec<f32> = value.get("pixels")?.as_array()?.iter().map(|v| v.as_f64().map(|f| f as f32)).collect::<Option<_>>()?;
        if pixels.len() as u64 != width.checked_mul(height)?.checked_mul(4)? { return None; }
        Some((x, y, width, height, pixels))
    }

    /// Parse an engine selectionCoverage result into (x, y, width, height, coverage).
    pub(crate) fn parse_coverage(value: &Value) -> Option<(i64, i64, u64, u64, Vec<f32>)> {
        if value.get("format").and_then(Value::as_str) != Some("coverage-f32-rowmajor") { return None; }
        let x = value.get("x").and_then(Value::as_i64)?;
        let y = value.get("y").and_then(Value::as_i64)?;
        let width = value.get("width").and_then(Value::as_u64).filter(|v| *v >= 1)?;
        let height = value.get("height").and_then(Value::as_u64).filter(|v| *v >= 1)?;
        let coverage: Vec<f32> = value.get("coverage")?.as_array()?.iter().map(|v| v.as_f64().map(|f| f as f32)).collect::<Option<_>>()?;
        if coverage.len() as u64 != width.checked_mul(height)? { return None; }
        Some((x, y, width, height, coverage))
    }

    /// Tiled exact capture of `command` ("image.analysis.regionPixels" or
    /// "image.analysis.selectionCoverage") over a region into one row-major
    /// buffer: 512x512 tiles (the engine's 262144-px per-call cap) scattered
    /// into place so no full-region JSON value ever exists, hard aggregate
    /// budget 2M px (32MB of RGBA f32 / 8MB of coverage at 16/4 bytes per
    /// pixel), checked arithmetic throughout. None on any shortfall.
    pub(crate) fn capture_tiled<S: LiveEditSession + ?Sized>(session: &mut S, command: &str, rect: [i64; 4], channels: u64) -> Option<Vec<f32>> {
        const TILE: u64 = 512;
        const MAX_AGGREGATE_PIXELS: u64 = 2_000_000;
        let (x0, y0) = (rect[0], rect[1]);
        let (w, h) = (u64::try_from(rect[2]).ok()?, u64::try_from(rect[3]).ok()?);
        let total = w.checked_mul(h)?;
        if total == 0 || total > MAX_AGGREGATE_PIXELS { return None; }
        let len = usize::try_from(total.checked_mul(channels)?).ok()?;
        let mut out = Vec::new();
        out.try_reserve_exact(len).ok()?;
        out.resize(len, 0.0);
        let mut ty = 0u64;
        while ty < h {
            let th = TILE.min(h - ty);
            let mut tx = 0u64;
            while tx < w {
                let tw = TILE.min(w - tx);
                let value = session.command_run(command,
                    json!({"x": x0 + tx as i64, "y": y0 + ty as i64, "width": tw, "height": th})).ok()?;
                let (rx, ry, pw, ph, px) = if channels == 4 {
                    Self::parse_exact_region(&value)?
                } else {
                    Self::parse_coverage(&value)?
                };
                // The engine intersects with the canvas: accept only the
                // exact tile requested, else the scatter offsets would be
                // wrong - a shortfall is honest None, never misplaced data.
                if pw != tw || ph != th || rx != x0 + tx as i64 || ry != y0 + ty as i64 { return None; }
                for row in 0..ph {
                    let src = usize::try_from(row.checked_mul(tw.checked_mul(channels)?)?).ok()?;
                    let dst = usize::try_from(((ty + row).checked_mul(w)?.checked_add(tx)?).checked_mul(channels)?).ok()?;
                    let n = usize::try_from(tw.checked_mul(channels)?).ok()?;
                    out[dst..dst + n].copy_from_slice(&px[src..src + n]);
                }
                tx += tw;
            }
            ty += th;
        }
        Some(out)
    }

    /// The target-selection stage capture, taken IMMEDIATELY BEFORE the
    /// operation whose effect is being evaluated: document identity, the
    /// selection's exact per-pixel coverage and the region's exact rendered
    /// pixels (both bounded by the engine's exact cap, so the target's exact
    /// coverage survives the selection's consumption and before/after pixels
    /// compare exactly - no image bytes ever leave the process), the
    /// bounding-region digest, the protected canvas-complement border rects
    /// (digests plus exact pixels), and coverage-aware colour statistics of
    /// the selection itself. None when the selection is empty or unreadable
    /// at this point; unavailable exact reads stay None (honest unknown).
    fn capture_selection_stage<S: LiveEditSession + ?Sized>(session: &mut S, step_index: usize, earlier_mutations: usize) -> Option<SelectionCapture> {
        let document = session.inspect_active().ok()??;
        let revision = document.get("revision").and_then(Value::as_u64);
        let name = document.get("name").and_then(Value::as_str)?.to_string();
        let path = match document.get("path") {
            Some(Value::String(p)) => Some(p.clone()),
            _ => None,
        };
        let (cw, ch) = (document.get("width").and_then(Value::as_i64)?, document.get("height").and_then(Value::as_i64)?);
        let document_id = document.get("id").and_then(Value::as_u64);
        let a = document.get("selectionBounds")?.as_array()?;
        if a.len() != 4 { return None; }
        let mut bounds = [0i64; 4];
        for (i, v) in a.iter().enumerate() { bounds[i] = v.as_i64()?; }
        if bounds[2] <= 0 || bounds[3] <= 0 { return None; }
        let [tx, ty, tw, th] = bounds;
        let coverage = Self::capture_tiled(session, "image.analysis.selectionCoverage", bounds, 1);
        let before_pixels = Self::capture_tiled(session, "image.analysis.regionPixels", bounds, 4);
        let region = Self::capture_digest(session, Some(bounds)).ok().flatten();
        let mut protected = Vec::new();
        let mut protected_pixels = Vec::new();
        for rect in [[0, 0, cw, ty], [0, ty + th, cw, ch - ty - th], [0, ty, tx, th], [tx + tw, ty, cw - tx - tw, th]] {
            if rect[2] > 0 && rect[3] > 0 {
                let mark = Self::capture_digest(session, Some(rect)).ok().flatten();
                let px = Self::capture_tiled(session, "image.analysis.regionPixels", rect, 4);
                protected.push((rect, mark));
                protected_pixels.push((rect, px));
            }
        }
        let color_stats = session.command_run("image.analysis.colorStats", json!({"maxPixels": EVIDENCE_MAX_PIXELS})).ok()
            .filter(|v| v.get("domain").and_then(Value::as_str) == Some("selection"));
        Some(SelectionCapture {
            step_index,
            document_id,
            document_name: name,
            document_path: path,
            earlier_mutations,
            bounds,
            revision,
            coverage,
            before_pixels,
            region,
            protected,
            protected_pixels,
            color_stats,
        })
    }

    /// Compare two captures. A dimension change is an observable change of
    /// its own (geometry evidence; digests are never compared). A stride
    /// change at identical dimensions is NOT a pixel change: the grids
    /// differ, so the digests are incomparable and the honest class is
    /// Unknown. Digests are only compared over an identical grid.
    pub(crate) fn compare_digests(before: &DigestMark, after: &DigestMark) -> VisualEvidence {
        if (before.x, before.y) != (after.x, after.y) {
            return VisualEvidence {
                class: VisualEvidenceClass::Unknown,
                detail: "the capture origins differ; the digests cover different regions and are not comparable".into(),
            };
        }
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
        let revision_note = match (before.revision, after.revision) {
            (Some(b), Some(a)) if b == a => format!("; SAME document revision {b} at both captures: no journaled change landed between them"),
            (Some(b), Some(a)) => format!("; document revisions {b} -> {a}"),
            _ => "; capture revisions unreported".to_string(),
        };
        if before.digest != after.digest {
            VisualEvidence {
                class: VisualEvidenceClass::Changed,
                detail: format!(
                    "the sampled quantized composite data differs ({}x{}, stride {}, {} px sampled{}); a data difference is not proof of a visible appearance change, intent or quality",
                    after.width, after.height, after.stride, after.sampled, revision_note
                ),
            }
        } else {
            VisualEvidence {
                class: VisualEvidenceClass::NoChangeObserved,
                detail: format!(
                    "no difference observed on the sampled, 8-bit-clamped grid ({}x{}, stride {}, {} px sampled{}); sampling, quantization and hash collisions can hide changes",
                    after.width, after.height, after.stride, after.sampled, revision_note
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
        // The target-selection stage: IMMEDIATELY BEFORE the first mutation
        // following the plan's last selection/mask step THAT PRECEDES A
        // MUTATION. Sequences like "select, edit, deselect" must not anchor
        // on the trailing deselect (no edit follows it), and the capture must
        // never land after an edit it claims to precede. A plan that edits in
        // several selection segments has its LAST segment's target measured;
        // earlier segments are covered only by the protected-complement
        // comparison, which the measurement report states explicitly.
        let is_selection_step = |s: &PlanStep| s.command.starts_with("select.");
        // Effect metadata, not string guessing: a command mutates when the
        // engine's own registry says it journals; unknown commands are
        // treated as mutations (fail closed). Mask commands journal - they
        // are edits and get a pre-mutation baseline like any other edit.
        let is_mutation = |s: &PlanStep| !is_selection_step(s)
            && fotodit_engine::command_specs().iter()
                .find(|spec| spec.id == s.command).map_or(true, |spec| spec.journal);
        let capture_before_step = collect_evidence.then(|| {
            // Last selection step that has a mutation somewhere after it;
            // capture immediately before that mutation.
            plan.steps.iter().enumerate().rev()
                .find(|(_, s)| is_selection_step(s))
                .and_then(|(i, _)| plan.steps.iter().enumerate()
                    .skip(i + 1)
                    .find(|(_, s)| is_mutation(s))
                    .map(|(j, _)| j))
        }).flatten();
        let mut selection_capture: Option<SelectionCapture> = None;

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
                        match Self::capture_digest(session, None) {
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

            if capture_before_step == Some(idx) && selection_capture.is_none() {
                // Mutations (engine journal metadata) before the last
                // selection step preceding idx belong to earlier selection
                // segments.
                let last_sel = plan.steps[..idx].iter().rposition(|s| s.command.starts_with("select."));
                let earlier_mutations = plan.steps[..last_sel.unwrap_or(0)].iter()
                    .filter(|s| is_mutation(s)).count();
                selection_capture = Self::capture_selection_stage(session, idx, earlier_mutations);
            }
            match resolution.and_then(|params| session.command_run(&step.command, params)) {
                Ok(output) => {
                    let _ = &output;
                    // Only real creation steps produce referenceable ids; other
                    // commands echo their target id in the same "layer" output
                    // key (type.setStyle, layer.removeBackground, ...). The set
                    // is shared with contract validation so the two cannot drift.
                    if is_layer_creating(&step.command) {
                        if let Some(id) = output.get("layer").and_then(Value::as_u64) {
                            created_ids[idx] = Some(id);
                        }
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
                                    let after_outcome = Self::capture_digest(session, None);
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
            ExecutionEvidence { steps: evidence, selection_capture },
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
