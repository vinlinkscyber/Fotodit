//! Ten-prompt parse_intent diagnostic (external review set, 2026-11-16). Prints the raw
//! parser return for each case against the specified synthetic fixture. Diagnostic only:
//! it asserts nothing about parser quality; the printed output is the evidence.

use fotodit_automation::ai::{AiPlanner, CommandCatalog, DeterministicRulePlanner, NaturalLanguageEngine, VisualElement, VisualScene};
use fotodit_automation::AiContext;
use std::time::Instant;

fn layer(id: u64, name: &str) -> VisualElement {
    VisualElement {
        id,
        name: name.to_string(),
        kind: "Pixel".to_string(),
        text_content: None,
        font_size_pt: None,
        bounds: [0, 0, 1200, 800],
        is_visible: true,
        has_mask: false,
        confidence: 1.0,
    }
}

/// Common fixture: 1200x800, L1 Background / L2 Portrait / L3 Logo, all visible, L2 selected.
fn fixture() -> VisualScene {
    VisualScene {
        width: 1200,
        height: 800,
        active_layer_id: Some(2),
        layers: vec![layer(1, "Background"), layer(2, "Portrait"), layer(3, "Logo")],
        ..Default::default()
    }
}

fn run(engine: &NaturalLanguageEngine, case: &str, prompt: &str, scene: &VisualScene, context: &AiContext) {
    let start = Instant::now();
    let result = engine.parse_intent(prompt, scene, context);
    let elapsed = start.elapsed();
    println!("=== CASE {case} ===");
    println!("prompt: {prompt}");
    match &result {
        Ok(intent) => println!("parse_intent Ok ({} us):\n{}", elapsed.as_micros(), serde_json::to_string_pretty(intent).unwrap()),
        Err(e) => println!("parse_intent Err ({} us): {e}", elapsed.as_micros()),
    }
    println!("=== END CASE {case} ===");
}

#[test]
fn ten_prompt_parse_diagnostic_prints_raw_outputs() {
    let engine = NaturalLanguageEngine::new();

    println!("REVISION: branch chore/optimization-efficiency (see CI run commit)");
    println!("FIXTURE: 1200x800; L1 Background, L2 Portrait (selected), L3 Logo; all visible; no selection/mask; empty context unless stated");

    let cases = [
        ("01", "Set the opacity of the Portrait layer to 60%."),
        ("02", "plz make Portrait 40 percent transparant"),
        ("03", "Do not hide the Portrait layer. Set only its opacity to 60%."),
        ("04", "Rename the Portrait layer to Subject, then set that same layer's opacity to 60%."),
        ("05", "Make it pop, but keep it natural."),
        ("06", "Make the sky bluer without changing the person."),
        ("07", "Generate a new image of a red bicycle on a white background. Leave my open document unchanged."),
        ("09", "Set the opacity of the Watermark layer to 20%."),
        ("10", "Undo the last change."),
    ];
    for (case, prompt) in cases {
        run(&engine, case, prompt, &fixture(), &AiContext::new());
    }

    // Case 08: one explicitly successful prior interaction (case 01 prompt, parsed and planned
    // for real, then recorded), per the diagnostic setup.
    let mut context = AiContext::new();
    let scene = fixture();
    let prior_prompt = "Set the opacity of the Portrait layer to 60%.";
    match engine.parse_intent(prior_prompt, &scene, &context) {
        Ok(prior_intent) => {
            let planner = DeterministicRulePlanner::new(CommandCatalog::new());
            match planner.plan(&prior_intent, &scene, &context) {
                Ok(plan) => context.record_interaction(prior_prompt.to_string(), plan),
                Err(e) => {
                    println!("case 08 setup: case 01 planning failed ({e}); recording prompt history only");
                    context.prompt_history.push(prior_prompt.to_string());
                }
            }
        }
        Err(e) => {
            println!("case 08 setup: case 01 parsing failed ({e}); recording prompt history only");
            context.prompt_history.push(prior_prompt.to_string());
        }
    }
    println!("=== CASE 08 SETUP ===");
    println!("prior interaction recorded: {prior_prompt}");
    println!("context after setup: {}", serde_json::to_string_pretty(&context).unwrap());
    run(&engine, "08", "Do the same to Logo, but use 30% instead.", &scene, &context);
}
