//! Synchronous editing boundary shared by the AI pipeline and its host.
//!
//! Implementations retain their own document, command and filesystem policy. This trait
//! grants no access and does not make a bare engine session a GUI-safe dispatcher.

use serde_json::Value;

use crate::{AutomationError, Headless};

/// A borrowed, synchronous editing destination for a single AI operation.
///
/// The host must keep the active document stable for the entire operation, including
/// rollback commands. Return a completed command result, never a pending job receipt.
/// Apply authorization, modal/document ownership and job admission before dispatch,
/// including for `edit.undo`. Existing stricter authorization hooks must be preserved.
///
/// No `Send`, `Sync` or `'static` bound is required: a host can borrow its real editor
/// through a local adapter instead of creating a second, invisible editing session.
/// For planning separated from execution, the host must bind and recheck document ID,
/// active-document generation and revision. This synchronous interface supplies neither
/// that prepared-plan protocol nor permission to use a remote provider.
///
/// There is intentionally no production implementation for `fotodit_engine::Session`.
/// Desktop adapters must use the host's full admitted command path, not raw `execute`.
pub trait LiveEditSession {
    /// Inspect the active document using `fotodit_engine::inspect::document`'s JSON shape.
    /// `None` means only that no document is active; other failures remain errors.
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError>;

    /// Run an admitted command against the same live destination and wait for completion.
    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError>;

    /// OPTIONAL observation hook: a position-sensitive digest of the rendered
    /// active-document composite (the engine's read-only, non-journaling
    /// `image.analysis.pixelDigest` command) for before/after result evidence.
    ///
    /// Contract:
    /// - the default implementation returns `None`: a host without pixel access
    ///   honestly has no visual evidence, and callers classify that as unknown -
    ///   evidence is never invented;
    /// - implementations MUST route through the host's own admitted command path,
    ///   preserving admission, existing authorization hooks and job policy exactly
    ///   as for editing commands. Non-mutation, no journaling, synchronous
    ///   completion, and a stable active document across the capture are
    ///   PRECONDITIONS the implementation must honor: the caller's inspection is
    ///   only an identity hint (name + path) and cannot detect every transient
    ///   or same-name substitution, so it is not a verification of this
    ///   contract;
    /// - an authorization refusal is reported as `Some(Err(_))`: it is never
    ///   bypassed through a side channel, and it never fails the editing step it
    ///   observes - it surfaces as honest unknown evidence;
    /// - the returned JSON is unauthenticated host input: callers validate the
    ///   full fixed schema (domain, schema version, hash, quantization, digest
    ///   syntax, stride/dims/sample-count consistency) before trusting it, so a
    ///   custom host's output inherits no engine consistency;
    /// - `Some(Err(_))` means the host supports the hook but this capture failed
    ///   or was refused; the error string is surfaced in the evidence detail so a
    ///   failure is distinguishable from honest absence; `None` means the host
    ///   offers no digest capability at all;
    /// - `max_pixels` caps the composite render work of one capture;
    ///   implementations must pass it through and must not exceed it. The digest
    ///   bounds the composite buffer only; renderer-internal intermediates and
    ///   wall-clock time remain governed by the renderer.
    fn visual_digest(&mut self, max_pixels: u64) -> Option<Result<Value, String>> {
        let _ = max_pixels;
        None
    }
}

impl LiveEditSession for Headless {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        Ok(self.session.active().map(fotodit_engine::inspect::document))
    }

    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
        // Preserve Headless admission, filesystem policy, job synchronization and cache cleanup.
        Headless::command_run(self, command, params)
    }

    fn visual_digest(&mut self, max_pixels: u64) -> Option<Result<Value, String>> {
        // The same admitted path as every other command (see above): admission,
        // job synchronization and cache cleanup are preserved. `image.analysis.pixelDigest`
        // is read-only and non-journaling.
        Some(Headless::command_run(self, "image.analysis.pixelDigest", serde_json::json!({"maxPixels": max_pixels})).map_err(|e| e.to_string()))
    }
}
