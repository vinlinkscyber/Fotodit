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
    /// Execute an editing plan step-by-step with automatic transactional rollback on failure.
    pub fn execute_plan(headless: &mut Headless, plan: &EditingPlan) -> Result<ExecutionResult, AiError> {
        Self::execute_plan_in(headless, plan)
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
        let total_steps = plan.steps.len();
        let mut baseline: Option<DocMark> = None;
        // Entries verified as appended by this plan's steps, oldest first.
        let mut plan_labels: Vec<String> = Vec::new();
        // The first reason rollback became unsafe; once set, no undo is issued.
        let mut closed: Option<String> = None;
        // A creation plan leaves the created document open even when all its
        // journaled edits are undone: incomplete relative to the original state.
        let mut created_document_open = false;
        let mut mark_prev = Self::document_mark(session);

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

            match session.command_run(&step.command, step.params.clone()) {
                Ok(output) => {
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

        Ok(ExecutionResult {
            total_steps,
            executed_steps: executed.len(),
            step_executions: executed,
            is_success: true,
            error: None,
            undos_performed: 0,
        })
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
