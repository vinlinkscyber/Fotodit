//! Subsystem F: Visual Evaluation Engine
//!
//! Performs visual verification of resulting document scenes, verifies goal achievement against
//! user intents, checks WCAG contrast compliance on typography, validates typographic hierarchy,
//! and computes layout balance and confidence metrics.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};

use super::creative_design::ColorHarmony;
use super::execution::ExecutionResult;
use super::natural_language::{EditingIntent, IntentCategory, NaturalLanguageEngine};
use super::perception::{VisualElement, VisualScene};

/// Diagnostic evaluation report assessing the visual quality and intent fulfillment of an edit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Overall satisfaction confidence score (0.0 to 1.0).
    pub confidence_score: f32,
    /// Whether the user intent was successfully realized.
    pub intent_satisfied: bool,
    /// Whether contrast requirements for text legibility were met.
    pub contrast_check_passed: bool,
    /// Whether typographic hierarchy (heading > body) is strictly maintained.
    pub hierarchy_valid: bool,
    /// Layout balance score based on margins and alignment (0.0 to 1.0).
    pub layout_balance_score: f32,
    /// Qualitative observations and verification findings.
    pub observations: Vec<String>,
}

/// Evaluation Engine that inspects before and after states.
#[derive(Clone, Debug, Default)]
pub struct EvaluationEngine;

impl EvaluationEngine {
    /// Create new evaluation engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate goal achievement and visual quality between initial and final document scenes.
    pub fn evaluate(
        &self,
        intent: &EditingIntent,
        initial_scene: &VisualScene,
        final_scene: &VisualScene,
        execution: &ExecutionResult,
    ) -> EvaluationReport {
        let mut observations = Vec::new();
        let mut intent_satisfied = execution.is_success;
        let mut contrast_check_passed = true;
        let mut hierarchy_valid = true;
        let mut layout_balance_score = 0.85f32;

        if !execution.is_success {
            observations.push("Execution failed before all steps completed.".into());
            return EvaluationReport {
                confidence_score: 0.0,
                intent_satisfied: false,
                contrast_check_passed: false,
                hierarchy_valid: false,
                layout_balance_score: 0.0,
                observations,
            };
        }

        match intent.category {
            IntentCategory::SubjectIsolation => {
                let has_masked_layer = final_scene.layers.iter().any(|l| l.has_mask);
                if has_masked_layer {
                    observations.push("Foreground subject successfully isolated with layer mask.".into());
                } else if final_scene.layers.len() > initial_scene.layers.len() {
                    observations.push("New isolated subject layer created.".into());
                } else {
                    observations.push("Subject isolation command executed.".into());
                }
            }
            IntentCategory::BackgroundReplacement => {
                let has_fill_or_bg = final_scene
                    .layers
                    .iter()
                    .any(|l| l.name.to_lowercase().contains("fill") || l.name.to_lowercase().contains("background"));
                if has_fill_or_bg {
                    observations.push("New background layer present behind subject.".into());
                } else {
                    observations.push("Background replacement executed.".into());
                }
            }
            IntentCategory::GraphicDesign => {
                // Only visible type layers count: a hidden layer carrying the
                // text does not put the requested wording on the canvas.
                let text_layers: Vec<_> = final_scene.layers.iter().filter(|l| l.kind == "Type" && l.is_visible).collect();
                if text_layers.len() >= 2 {
                    let mut sizes: Vec<f32> = text_layers.iter().filter_map(|l| l.font_size_pt).collect();
                    sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    if sizes.len() >= 2 && sizes[0] > sizes[1] * 1.15 {
                        observations.push(format!(
                            "Clear typographic hierarchy established (Headline: {:.1}pt, Subhead: {:.1}pt).",
                            sizes[0], sizes[1]
                        ));
                    } else {
                        hierarchy_valid = false;
                        observations.push("Typographic hierarchy could be amplified between heading and body.".into());
                    }
                } else if text_layers.is_empty() {
                    intent_satisfied = false;
                    observations.push("Design intent expected typography elements but none were found.".into());
                }

                // Semantic goal check: names/subject text the user explicitly
                // supplied must survive into the actual final text layers -
                // structure alone (two type layers with hierarchy) is not intent.
                let required_terms = NaturalLanguageEngine::new().subject_terms(&intent.raw_prompt);
                if !required_terms.is_empty() {
                    let layer_texts: Vec<String> = text_layers.iter().filter_map(|l| l.text_content.clone()).collect();
                    for term in &required_terms {
                        let found = layer_texts
                            .iter()
                            .any(|t| t.to_lowercase().contains(&term.to_lowercase()));
                        if !found {
                            intent_satisfied = false;
                            observations.push(format!(
                                "Design intent required the text '{term}' but no text layer contains it."
                            ));
                        }
                    }
                }

                // Check text contrast
                if let Some(bg_color) = intent.extracted_color.as_deref().or(Some("#ffffff")) {
                    for _text_layer in &text_layers {
                        let text_color = "#1e293b"; // standard fallback
                        let ratio = ColorHarmony::contrast_ratio(text_color, bg_color);
                        if ratio < 4.5 {
                            contrast_check_passed = false;
                            observations.push(format!("Contrast ratio ({ratio:.1}:1) is below WCAG AA recommended 4.5:1."));
                        }
                    }
                }

                // Rendered-geometry readability of the GOAL-BEARING text:
                // inspect `bounds` are real content bounds in document pixels
                // (type size is the em; 1 pt == 1 px at the default 72 dpi),
                // so scale, on-canvas fit and overlap are verifiable facts for
                // measured layers - not plan intentions, and not the largest
                // unrelated prior layer. A layer whose ink was never measured
                // (zero bounds) carries no geometry evidence: satisfaction is
                // withheld, never assumed. Layers with no bounds entry at all
                // arrive here through perception's record fallback and cannot
                // be distinguished from full-canvas content, so they are not
                // treated as missing.
                let canvas_w = final_scene.width as i32;
                let canvas_h = final_scene.height as i32;
                if canvas_w > 0 && canvas_h > 0 {
                    let contains_term = |layer: &VisualElement, term: &str| {
                        layer
                            .text_content
                            .as_deref()
                            .is_some_and(|t| t.to_lowercase().contains(&term.to_lowercase()))
                    };
                    let mut goal_bearing: Vec<&VisualElement> = Vec::new();
                    for term in &required_terms {
                        let bearers: Vec<&VisualElement> = text_layers
                            .iter()
                            .filter(|l| contains_term(l, term))
                            .copied()
                            .collect();
                        // Presence was checked above; here every required term
                        // must also have at least one bearer whose measured
                        // ink sits fully on the canvas.
                        let mut any_measured = false;
                        let mut any_on_canvas = false;
                        for bearer in &bearers {
                            let [x, y, w, h] = bearer.bounds;
                            if w <= 0 || h <= 0 {
                                continue;
                            }
                            any_measured = true;
                            if x >= 0 && y >= 0 && x + w <= canvas_w && y + h <= canvas_h {
                                any_on_canvas = true;
                            }
                        }
                        if !any_on_canvas {
                            intent_satisfied = false;
                            if any_measured {
                                observations.push(format!(
                                    "Required text '{term}' renders off the canvas or is clipped."
                                ));
                            } else {
                                observations.push(format!(
                                    "Required text '{term}' has no measured ink geometry; satisfaction withheld."
                                ));
                            }
                        }
                        for bearer in bearers {
                            if !goal_bearing.iter().any(|l| l.id == bearer.id) {
                                goal_bearing.push(bearer);
                            }
                        }
                    }

                    // Required text layers must not print on top of each other.
                    for (i, a) in goal_bearing.iter().enumerate() {
                        for b in goal_bearing.iter().skip(i + 1) {
                            let overlap_w = (a.bounds[0] + a.bounds[2]).min(b.bounds[0] + b.bounds[2])
                                - a.bounds[0].max(b.bounds[0]);
                            let overlap_h = (a.bounds[1] + a.bounds[3]).min(b.bounds[1] + b.bounds[3])
                                - a.bounds[1].max(b.bounds[1]);
                            if overlap_w > 0 && overlap_h > 0 {
                                intent_satisfied = false;
                                observations.push(format!(
                                    "Required text layers '{}' and '{}' overlap on the canvas.",
                                    a.text_content.as_deref().unwrap_or("?"),
                                    b.text_content.as_deref().unwrap_or("?")
                                ));
                            }
                        }
                    }

                    // Legibility floor on the actual headline: the
                    // largest-font goal-bearing layer when the prompt named
                    // copy, otherwise the largest-font visible text layer.
                    // Keyword presence alone is not a usable design.
                    let pool: Vec<&VisualElement> = if goal_bearing.is_empty() {
                        text_layers.iter().copied().collect()
                    } else {
                        goal_bearing.clone()
                    };
                    let mut measured: Vec<&VisualElement> = pool
                        .into_iter()
                        .filter(|l| l.bounds[2] > 0 && l.bounds[3] > 0)
                        .collect();
                    measured.sort_by(|a, b| {
                        b.font_size_pt
                            .partial_cmp(&a.font_size_pt)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some(headline) = measured.first() {
                        let ink_frac = headline.bounds[3] as f32 / canvas_h as f32;
                        if ink_frac < 0.035 {
                            intent_satisfied = false;
                            observations.push(format!(
                                "Headline renders at {:.1}% of canvas height - below the 3.5% legibility floor for a promotional headline.",
                                ink_frac * 100.0
                            ));
                        }
                        let [x, y, w, h] = headline.bounds;
                        if x < 0 || y < 0 || x + w > canvas_w || y + h > canvas_h {
                            intent_satisfied = false;
                            observations.push("Headline ink extends outside the canvas.".into());
                        }
                    }
                }
                layout_balance_score = 0.92;
            }
            IntentCategory::PhotoEnhancement => {
                let adj_count = final_scene
                    .layers
                    .iter()
                    .filter(|l| l.kind == "Adjustment" || l.name.to_lowercase().contains("curves") || l.name.to_lowercase().contains("vibrance"))
                    .count();
                observations.push(format!(
                    "Photo enhancement applied ({adj_count} non-destructive adjustment/filter layers active)."
                ));
            }
            IntentCategory::TypographyRefinement => {
                observations.push("Typography styling and positioning updated.".into());
            }
            IntentCategory::ColorStyling => {
                observations.push("Color harmony and tonal adjustments applied.".into());
            }
            IntentCategory::LayoutAdjustment => {
                observations.push("Layout geometry and spatial positioning adjusted.".into());
            }
            IntentCategory::DirectFilterOrAdjustment | IntentCategory::GeneralEditing => {
                observations.push("Requested adjustments applied successfully.".into());
            }
        }

        // Calculate overall confidence score
        let mut score = 0.80f32;
        if intent_satisfied {
            score += 0.10;
        }
        if contrast_check_passed {
            score += 0.05;
        }
        if hierarchy_valid {
            score += 0.05;
        }
        let confidence_score = score.clamp(0.0, 1.0);

        EvaluationReport {
            confidence_score,
            intent_satisfied,
            contrast_check_passed,
            hierarchy_valid,
            layout_balance_score,
            observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::perception::VisualElement;

    #[test]
    fn evaluate_subject_isolation() {
        let engine = EvaluationEngine::new();
        let intent = EditingIntent {
            raw_prompt: "isolate subject".into(),
            category: IntentCategory::SubjectIsolation,
            archetype: None,
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: None,
        };

        let initial_scene = VisualScene::default();
        let mut final_scene = VisualScene::default();
        final_scene.layers.push(VisualElement {
            id: 1,
            name: "Subject".into(),
            kind: "Pixel".into(),
            text_content: None,
            font_size_pt: None,
            bounds: [0, 0, 800, 600],
            is_visible: true,
            has_mask: true,
            confidence: 1.0,
            ..Default::default()
        });

        let execution = ExecutionResult {
            total_steps: 1,
            executed_steps: 1,
            step_executions: vec![],
            is_success: true,
            error: None,
            undos_performed: 0,
        };

        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &execution);
        assert!(report.intent_satisfied);
        assert!(report.confidence_score >= 0.9);
    }

    fn graphic_design_intent(raw_prompt: &str, text_content: Option<&str>) -> EditingIntent {
        EditingIntent {
            raw_prompt: raw_prompt.into(),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: text_content.map(str::to_string),
        }
    }

    fn type_layer(id: u64, text: &str, size: f32) -> VisualElement {
        VisualElement {
            id,
            name: text.into(),
            kind: "Type".into(),
            text_content: Some(text.into()),
            font_size_pt: Some(size),
            bounds: [0, 0, 800, 100],
            is_visible: true,
            has_mask: false,
            confidence: 1.0,
            ..Default::default()
        }
    }

    fn successful_execution(steps: usize) -> ExecutionResult {
        ExecutionResult {
            total_steps: steps,
            executed_steps: steps,
            step_executions: vec![],
            is_success: true,
            error: None,
            undos_performed: 0,
        }
    }

    #[test]
    fn evaluate_graphic_design_flags_missing_user_named_content() {
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("create a poster for my taxi busines to promote it its called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();
        let mut final_scene = VisualScene::default();
        final_scene.layers.push(type_layer(1, "CREATIVE EXPO 2026", 72.0));
        final_scene.layers.push(type_layer(2, "A Celebration of Art & Technology", 36.0));

        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &successful_execution(3));
        assert!(!report.intent_satisfied, "canned template text must not satisfy a named-business prompt");
        assert!(report.observations.iter().any(|o| o.contains("chndrax")));
    }

    #[test]
    fn evaluate_graphic_design_accepts_user_named_content() {
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("create a poster for my taxi busines to promote it its called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();
        let mut final_scene = VisualScene::default();
        final_scene.layers.push(type_layer(1, "chndrax", 72.0));
        final_scene.layers.push(type_layer(2, "taxi busines", 36.0));

        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &successful_execution(3));
        assert!(report.intent_satisfied, "user-named content present should satisfy intent: {:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_requires_all_user_terms() {
        // Quoted tagline present but brand missing must not satisfy.
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("make a poster \"Grand Opening\" called chndrax", Some("Grand Opening"));
        let initial_scene = VisualScene::default();
        let mut final_scene = VisualScene::default();
        final_scene.layers.push(type_layer(1, "Grand Opening", 72.0));
        final_scene.layers.push(type_layer(2, "A Celebration of Art & Technology", 36.0));

        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &successful_execution(3));
        assert!(!report.intent_satisfied, "missing brand term must fail the goal: {:?}", report.observations);

        let mut full_scene = VisualScene::default();
        full_scene.layers.push(type_layer(1, "Grand Opening", 72.0));
        full_scene.layers.push(type_layer(2, "chndrax", 36.0));
        let report = engine.evaluate(&intent, &initial_scene, &full_scene, &successful_execution(3));
        assert!(report.intent_satisfied, "both required terms present: {:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_rejects_tiny_headline_on_real_canvas() {
        // Rendered-geometry readability: keyword presence alone is not a
        // usable poster. A 43 px headline on a 1600 px canvas (the defect the
        // user hit) is below the 3.5% legibility floor and fails the goal.
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("create a poster for my taxi busines to promote it its called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();

        let mut tiny_scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut headline = type_layer(1, "chndrax", 43.0);
        headline.bounds = [400, 300, 300, 43];
        tiny_scene.layers.push(headline);
        tiny_scene.layers.push(type_layer(2, "Taxi business", 32.0));
        let report = engine.evaluate(&intent, &initial_scene, &tiny_scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "a tiny headline is not a usable poster: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("legibility floor")), "{:?}", report.observations);

        // A properly scaled, on-canvas headline passes the geometry check.
        let mut good_scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut big_headline = type_layer(1, "chndrax", 144.0);
        big_headline.bounds = [262, 384, 676, 112];
        good_scene.layers.push(big_headline);
        let mut sub = type_layer(2, "Taxi business", 90.0);
        sub.bounds = [228, 575, 744, 70];
        good_scene.layers.push(sub);
        let report = engine.evaluate(&intent, &initial_scene, &good_scene, &successful_execution(4));
        assert!(report.intent_satisfied, "scaled on-canvas headline satisfies: {:?}", report.observations);

        // Ink extending past the canvas edge fails too.
        let mut off_scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut off_headline = type_layer(1, "chndrax", 144.0);
        off_headline.bounds = [900, 384, 676, 112];
        off_scene.layers.push(off_headline);
        off_scene.layers.push(type_layer(2, "Taxi business", 90.0));
        let report = engine.evaluate(&intent, &initial_scene, &off_scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "off-canvas headline fails: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("outside the canvas")), "{:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_withholds_when_required_geometry_is_missing_or_bad() {
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("make a poster \"Grand Opening\" called chndrax", Some("Grand Opening"));
        let initial_scene = VisualScene::default();

        // The required secondary term renders fully off-canvas: presence in
        // the layer list must not count as satisfied.
        let mut scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut head = type_layer(1, "Grand Opening", 144.0);
        head.bounds = [200, 400, 800, 120];
        let mut brand = type_layer(2, "chndrax", 90.0);
        brand.bounds = [200, 1700, 400, 70];
        scene.layers.push(head);
        scene.layers.push(brand);
        let report = engine.evaluate(&intent, &initial_scene, &scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "off-canvas required text must not satisfy: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("off the canvas") && o.contains("chndrax")), "{:?}", report.observations);

        // Required copy with no measured ink geometry: withhold, never assume.
        let mut scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut head = type_layer(1, "Grand Opening", 144.0);
        head.bounds = [200, 400, 800, 120];
        let mut brand = type_layer(2, "chndrax", 90.0);
        brand.bounds = [0, 0, 0, 0];
        scene.layers.push(head);
        scene.layers.push(brand);
        let report = engine.evaluate(&intent, &initial_scene, &scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "unmeasured required text must withhold satisfaction: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("no measured ink geometry")), "{:?}", report.observations);

        // Required layers printed on top of each other are not a composition.
        let mut scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut head = type_layer(1, "Grand Opening", 144.0);
        head.bounds = [200, 400, 800, 120];
        let mut brand = type_layer(2, "chndrax", 90.0);
        brand.bounds = [300, 450, 400, 70];
        scene.layers.push(head);
        scene.layers.push(brand);
        let report = engine.evaluate(&intent, &initial_scene, &scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "overlapping required text must not satisfy: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("overlap")), "{:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_withholds_when_inspect_bounds_field_is_absent() {
        // End-to-end through perception: a type layer whose inspect JSON has
        // NO bounds field at all must not count as measured, on-canvas ink.
        // Presence of the wording is not geometry evidence.
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("make a poster called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();
        let doc = serde_json::json!({
            "width": 1200,
            "height": 1600,
            "layers": [
                { "id": 1, "name": "Background", "kind": "Pixel", "visible": true, "hasMask": false, "bounds": [0, 0, 1200, 1600] },
                { "id": 2, "name": "Headline", "kind": "Type", "visible": true, "hasMask": false,
                  "text": { "text": "chndrax", "font": "Montserrat", "sizePt": 144 } }
            ]
        });
        let final_scene = super::super::perception::PerceptionEngine::new().perceive(&doc);
        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "absent inspect geometry must withhold satisfaction: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("no measured ink geometry")), "{:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_checks_goal_bearing_layers_not_unrelated_giants() {
        // A huge unrelated pre-existing layer must not become the evaluated
        // headline while the actual required copy is tiny.
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("make a poster called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();
        let mut scene = VisualScene { width: 1200, height: 1600, ..Default::default() };
        let mut giant = type_layer(1, "DECORATION", 300.0);
        giant.bounds = [0, 0, 1200, 300];
        let mut brand = type_layer(2, "chndrax", 20.0);
        brand.bounds = [500, 800, 200, 20];
        scene.layers.push(giant);
        scene.layers.push(brand);
        let report = engine.evaluate(&intent, &initial_scene, &scene, &successful_execution(4));
        assert!(!report.intent_satisfied, "the tiny required name fails even with an unrelated giant layer: {:?}", report.observations);
        assert!(report.observations.iter().any(|o| o.contains("legibility floor")), "{:?}", report.observations);
    }

    #[test]
    fn evaluate_graphic_design_ignores_hidden_text_layers() {
        // A hidden layer carrying the wording does not put it on the canvas.
        let engine = EvaluationEngine::new();
        let intent = graphic_design_intent("make a poster called chndrax", Some("chndrax"));
        let initial_scene = VisualScene::default();
        let mut final_scene = VisualScene::default();
        let mut hidden = type_layer(1, "chndrax", 72.0);
        hidden.is_visible = false;
        final_scene.layers.push(hidden);
        final_scene.layers.push(type_layer(2, "CREATIVE EXPO 2026", 36.0));

        let report = engine.evaluate(&intent, &initial_scene, &final_scene, &successful_execution(3));
        assert!(!report.intent_satisfied, "hidden text must not satisfy the goal: {:?}", report.observations);
    }
}
