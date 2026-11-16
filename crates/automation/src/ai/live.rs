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
}

impl LiveEditSession for Headless {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        Ok(self.session.active().map(fotodit_engine::inspect::document))
    }

    fn command_run(&mut self, command: &str, params: Value) -> Result<Value, AutomationError> {
        // Preserve Headless admission, filesystem policy, job synchronization and cache cleanup.
        Headless::command_run(self, command, params)
    }
}
