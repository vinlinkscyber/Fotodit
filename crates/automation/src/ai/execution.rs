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

/// Execution Engine responsible for atomic command dispatch and rollback.
pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Execute an editing plan step-by-step with automatic transactional rollback on failure.
    pub fn execute_plan(headless: &mut Headless, plan: &EditingPlan) -> Result<ExecutionResult, AiError> {
        Self::execute_plan_in(headless, plan)
    }

    /// Execute on the host's admitted command path, including rollback attempts.
    /// Preserves the existing one-undo-per-successful-step behavior; it is not a
    /// universal atomic transaction for non-history or cross-document commands.
    pub fn execute_plan_in<S: LiveEditSession + ?Sized>(
        session: &mut S,
        plan: &EditingPlan,
    ) -> Result<ExecutionResult, AiError> {
        let mut executed = Vec::new();
        let total_steps = plan.steps.len();

        for (idx, step) in plan.steps.iter().enumerate() {
            match session.command_run(&step.command, step.params.clone()) {
                Ok(output) => {
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

                    // Transactional rollback: undo previous successful steps
                    let mut rollback_count = 0;
                    for _ in 0..idx {
                        if session.command_run("edit.undo", json!({})).is_ok() {
                            rollback_count += 1;
                        }
                    }

                    return Err(AiError::ExecutionError {
                        step: idx,
                        command: step.command.clone(),
                        error: format!("{err_msg} (rolled back {rollback_count} previous step(s))"),
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
}
