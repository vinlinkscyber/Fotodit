//! Executed NL-pipeline proof: runs `process_instruction` end to end on real headless
//! documents and prints the prompt, plan, per-step execution results, and evaluation
//! output as JSON, so the run itself (not only assertions) is reviewable evidence.

use fotodit_automation::{AiContext, AiEngine, Headless};
use serde_json::json;

#[test]
fn executed_design_generation_prints_full_run() {
    let ai = AiEngine::new();
    let mut headless = Headless::trusted_local();
    let mut context = AiContext::new();

    let prompt = "Create an elegant wedding invitation";
    let result = ai
        .process_instruction(&mut headless, prompt, &mut context)
        .expect("design generation run should succeed");

    println!("=== EXECUTED RUN 1: design generation on empty canvas ===");
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    println!("=== END RUN 1 ===");

    assert!(result.execution.is_success);
    assert!(result.evaluation.intent_satisfied);
}

#[test]
fn executed_photo_enhancement_prints_full_run() {
    let ai = AiEngine::new();
    let mut headless = Headless::trusted_local();
    headless
        .command_run("file.new", json!({"width": 800, "height": 600, "background": "white"}))
        .unwrap();
    let mut context = AiContext::new();

    let prompt = "Make this photograph look professional and enhance clarity";
    let result = ai
        .process_instruction(&mut headless, prompt, &mut context)
        .expect("photo enhancement run should succeed");

    println!("=== EXECUTED RUN 2: photo enhancement on open document ===");
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    println!("=== END RUN 2 ===");

    assert!(result.execution.is_success);
    assert!(result.evaluation.intent_satisfied);
}
