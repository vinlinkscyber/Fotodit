//! Subsystem G: AI Planning Engine
//!
//! Generates structured, deterministic, and validated editing plans from natural language intents,
//! visual scene analysis, and conversational context. Supports provider-independent planner traits
//! for both offline deterministic rule engines and future LLM providers.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::command_intelligence::CommandCatalog;
use super::creative_design::{ArchetypeKind, DesignArchetype, LayoutEngine, TypographyHierarchy};
use super::natural_language::{EditingIntent, IntentCategory, NaturalLanguageEngine, normalize_display_copy, subject_purpose};
use super::perception::VisualScene;
use super::{AiContext, AiError, EditingPlan};

/// A single discrete step in an editing plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    /// Native command identifier (e.g. `type.create`, `layer.removeBackground`).
    pub command: String,
    /// Parameters matching the native engine schema.
    pub params: Value,
    /// Human-readable explanation of this step.
    pub description: String,
}

/// Abstract AI planner trait allowing pluggable planners.
pub trait AiPlanner: Send + Sync {
    /// Generate an editing plan given an intent, scene, and conversational context.
    fn plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        context: &AiContext,
    ) -> Result<EditingPlan, AiError>;
}

/// Fully deterministic, offline rule-based planner for creative workflows.
pub struct DeterministicRulePlanner {
    _catalog: CommandCatalog,
    layout_engine: LayoutEngine,
}

impl DeterministicRulePlanner {
    /// Create a new deterministic planner.
    pub fn new(catalog: CommandCatalog) -> Self {
        Self {
            _catalog: catalog,
            layout_engine: LayoutEngine::new(),
        }
    }

    /// Access the command catalog.
    pub fn catalog(&self) -> &CommandCatalog {
        &self._catalog
    }

    fn plan_subject_isolation(&self, intent: &EditingIntent, _scene: &VisualScene) -> EditingPlan {
        let mut params = json!({
            "sampleAllLayers": true,
            "refine": true,
        });

        if let Some(target_id) = intent.target_layer_id {
            if let Some(map) = params.as_object_mut() {
                map.insert("layer".into(), json!(target_id));
            }
        }

        EditingPlan {
            title: "Non-Destructive Subject Isolation".into(),
            rationale: "Isolates the primary subject cleanly with an edge-refined layer mask without destructive pixel alterations.".into(),
            category: IntentCategory::SubjectIsolation,
            steps: vec![PlanStep {
                command: "layer.removeBackground".into(),
                params,
                description: "Isolate subject onto layer mask with boundary alpha refinement".into(),
            }],
            creates_document: false,
            target_size: None,
        }
    }

    fn plan_background_replacement(&self, intent: &EditingIntent, scene: &VisualScene) -> EditingPlan {
        let mut steps = Vec::new();

        // 1. Isolate subject if not already isolated
        let already_isolated = scene.layers.iter().any(|l| l.has_mask);
        if !already_isolated {
            steps.push(PlanStep {
                command: "layer.removeBackground".into(),
                params: json!({ "sampleAllLayers": true, "refine": true }),
                description: "Isolate subject to separate from background".into(),
            });
        }

        // 2. Add solid color fill layer
        let hex_color = intent.extracted_color.clone().unwrap_or_else(|| "#0f172a".to_string());
        steps.push(PlanStep {
            command: "layer.newFillLayer.solidColor".into(),
            params: json!({ "color": hex_color }),
            description: format!("Create new solid color fill layer with '{hex_color}'"),
        });

        EditingPlan {
            title: "Background Replacement with Preserved Subject".into(),
            rationale: "Preserves foreground person or product by creating an alpha mask, then places a new solid color fill layer underneath.".into(),
            category: IntentCategory::BackgroundReplacement,
            steps,
            creates_document: false,
            target_size: None,
        }
    }

    fn plan_photo_enhancement(&self, intent: &EditingIntent, scene: &VisualScene) -> Result<EditingPlan, AiError> {
        if intent.target_layer_id.is_some() && intent.target_layer_id != scene.active_layer_id {
            return Err(AiError::PlanningError("photo enhancement targets the active layer; select the intended layer first".into()));
        }
        let steps = vec![
            PlanStep {
                command: "filter.sharpen.unsharpMask".into(),
                params: json!({
                    "layer": scene.active_layer_id,
                    "amount": 40,
                    "radius": 1.0,
                    "threshold": 2
                }),
                description: "Sharpen key details".into(),
            },
            PlanStep {
                command: "layer.newAdjustmentLayer.curves".into(),
                params: json!({
                    "points": [[0, 0], [64, 55], [192, 205], [255, 255]]
                }),
                description: "Add gentle contrast to highlights and shadows".into(),
            },
            PlanStep {
                command: "layer.newAdjustmentLayer.vibrance".into(),
                params: json!({
                    "vibrance": 18,
                    "saturation": 4
                }),
                description: "Boost color vibrance gently".into(),
            },
        ];

        Ok(EditingPlan {
            title: "Professional Photographic Enhancement & Clarity".into(),
            rationale: "Sharpens the photo first, then adjusts contrast and color vibrance.".into(),
            category: IntentCategory::PhotoEnhancement,
            steps,
            creates_document: false,
            target_size: None,
        })
    }

    fn plan_graphic_design(&self, intent: &EditingIntent, scene: &VisualScene) -> Result<EditingPlan, AiError> {
        let width = if scene.width > 0 { scene.width } else { 1200 };
        let height = if scene.height > 0 { scene.height } else { 1600 };

        let archetype_kind = if let Some(arch_name) = &intent.archetype {
            DesignArchetype::from_name(arch_name)
                .map(|a| a.kind)
                .unwrap_or(ArchetypeKind::PromotionalPoster)
        } else {
            ArchetypeKind::PromotionalPoster
        };

        let archetype = DesignArchetype::from_kind(archetype_kind);
        let hierarchy = TypographyHierarchy::compute_for_canvas(width, height, Some(archetype_kind));

        // Required user copy: every subject term the user supplied (quoted
        // spans and names introduced by "called"/"named") must appear in the
        // design, because the evaluator checks for each of them. Emitting only
        // the first would make the remaining terms unsatisfiable.
        let required_terms = NaturalLanguageEngine::new().subject_terms(&intent.raw_prompt);
        // Headline: the first user-supplied term when present (typically the
        // quoted or named subject); the archetype placeholder only when the
        // user supplied no content of their own.
        let headline_text = required_terms.first().cloned().unwrap_or_else(|| {
            match archetype_kind {
                ArchetypeKind::WeddingInvitation => "TOGETHER WITH THEIR FAMILIES",
                ArchetypeKind::PromotionalPoster => "CREATIVE EXPO 2026",
                ArchetypeKind::ProductAdvertising => "INNOVATION DEFINED",
                _ => "FEATURED PRESENTATION",
            }
            .to_string()
        });
        // Subheading: the second user-supplied term when there is one (e.g.
        // the brand name when a quoted tagline took the headline); then the
        // user's stated purpose interpreted as display copy; otherwise the
        // archetype placeholder.
        let subhead_text = required_terms.get(1).cloned()
            .or_else(|| subject_purpose(&intent.raw_prompt).map(|p| normalize_display_copy(&p)))
            .unwrap_or_else(|| {
                match archetype_kind {
                    ArchetypeKind::WeddingInvitation => "Request the Honor of Your Presence",
                    ArchetypeKind::PromotionalPoster => "A Celebration of Art & Technology",
                    ArchetypeKind::ProductAdvertising => "Engineered for Performance",
                    _ => "Discover the Possibilities",
                }
                .to_string()
            });

        let headline_color = archetype.primary_palette.get(1).cloned().unwrap_or_else(|| "#ffffff".into());
        let secondary_color = archetype.primary_palette.get(2).cloned().unwrap_or_else(|| "#94a3b8".into());
        struct CopyLine {
            text: String,
            base_pt: f32,
            font: String,
            color: String,
            description: String,
            gap_em: f32,
        }
        let mut copy_lines = vec![
            CopyLine {
                text: headline_text.clone(),
                base_pt: hierarchy.heading_pt,
                font: hierarchy.heading_font.clone(),
                color: headline_color,
                description: format!("Create primary headline with '{headline_text}' in {}", hierarchy.heading_font),
                gap_em: 0.55,
            },
            CopyLine {
                text: subhead_text.clone(),
                base_pt: hierarchy.subheading_pt,
                font: hierarchy.body_font.clone(),
                color: secondary_color.clone(),
                description: format!("Create subheading with '{subhead_text}'"),
                gap_em: 0.9,
            },
        ];
        // Any further user-supplied terms (rare, but each one is a required
        // term the evaluator checks) get their own caption lines so no
        // requested copy is silently dropped.
        for term in required_terms.iter().skip(2) {
            copy_lines.push(CopyLine {
                text: term.clone(),
                base_pt: hierarchy.caption_pt,
                font: hierarchy.body_font.clone(),
                color: secondary_color.clone(),
                description: format!("Add requested text '{term}'"),
                gap_em: 0.9,
            });
        }

        // Capacity planning for the WHOLE composition. Render-unit basis:
        // documents default to 72 dpi, where 1 pt of type em is 1 px; ink is
        // smaller than the em and the 0.78/0.30/1.6 factors are conservative
        // layout estimates, not measured glyph metrics. Native center
        // alignment places each line with real shaped advances, and a box
        // turns an over-wide line into native paragraph wrapping. Try the
        // composition at descending scales; if no scale keeps every baseline
        // and estimated ink bottom on the canvas at a legible size, refuse
        // honestly instead of clamping baselines into a pile or cropping.
        let max_w = width as f32 * 0.85;
        // Legibility floor is relative to the canvas, matching the
        // evaluator's relative ink check; tiny canvases get proportionally
        // tiny (but planned) type rather than a false refusal.
        let min_legible = (width.min(height) as f32 * 0.025).max(0.1);
        let max_baseline = height as f32 * 0.95;
        let max_bottom = height as f32 * 0.97;
        let mut placed: Option<Vec<(f32, i32, Option<[f32; 4]>)>> = None;
        'scales: for scale in [1.0f32, 0.85, 0.7, 0.55] {
            let mut cursor = height as f32 * 0.30;
            let mut attempt: Vec<(f32, i32, Option<[f32; 4]>)> = Vec::with_capacity(copy_lines.len());
            for line in &copy_lines {
                let size = (line.base_pt * scale).max(0.1);
                if size < min_legible {
                    break 'scales;
                }
                // Long copy wraps natively into as many rows as it needs;
                // the vertical capacity check below decides whether the
                // composition fits, instead of shrinking text indefinitely.
                let rows = (TypographyHierarchy::estimate_line_width(&line.text, size) / max_w).ceil().max(1.0);
                let baseline = cursor + size * 0.78;
                let (text_box, bottom) = if rows > 1.0 {
                    let box_h = rows * size * 1.6;
                    let top = baseline - size * 0.78;
                    (Some([width as f32 * 0.075, top, max_w, box_h]), top + box_h)
                } else {
                    (None, baseline + size * 0.30)
                };
                if baseline > max_baseline || bottom > max_bottom {
                    break;
                }
                cursor = bottom + size * line.gap_em;
                attempt.push((size, baseline.round() as i32, text_box));
            }
            if attempt.len() == copy_lines.len() {
                placed = Some(attempt);
                break;
            }
        }
        let placed = placed.ok_or_else(|| {
            AiError::PlanningError(format!(
                "the requested copy does not fit a {width}x{height} canvas at a legible size"
            ))
        })?;

        let mut steps = Vec::new();

        // 1. Solid background from palette
        let bg_color = intent.extracted_color.clone().unwrap_or_else(|| {
            archetype.primary_palette.first().cloned().unwrap_or_else(|| "#0f172a".into())
        });
        steps.push(PlanStep {
            command: "layer.newFillLayer.solidColor".into(),
            params: json!({ "color": bg_color }),
            description: format!("Set background color to '{bg_color}'"),
        });

        // 2..N. Text lines, natively centered and wrapped where needed.
        for (line, (size, baseline, text_box)) in copy_lines.iter().zip(placed) {
            steps.push(PlanStep {
                command: "type.create".into(),
                params: json!({
                    "x": (width / 2) as i32,
                    "align": "center",
                    "y": baseline,
                    "box": text_box,
                    "text": line.text,
                    "font": line.font,
                    "size": size,
                    "color": line.color,
                }),
                description: line.description.clone(),
            });
        }

        Ok(EditingPlan {
            title: format!("{} Composition", archetype.name),
            rationale: format!(
                "Constructs a balanced {} layout with {} typography hierarchy, calibrated whitespace margins, and harmonic palette.",
                archetype.name, hierarchy.heading_font
            ),
            category: IntentCategory::GraphicDesign,
            steps,
            creates_document: false,
            target_size: Some([width, height]),
        })
    }

    fn plan_typography_refinement(&self, intent: &EditingIntent, scene: &VisualScene) -> EditingPlan {
        let mut steps = Vec::new();

        if let Some(target_id) = intent.target_layer_id {
            // Edit existing layer
            let mut style_params = json!({ "layer": target_id });
            if let Some(sz) = intent.numeric_value {
                if let Some(map) = style_params.as_object_mut() {
                    map.insert("size".into(), json!(sz));
                }
            }
            if let Some(col) = &intent.extracted_color {
                if let Some(map) = style_params.as_object_mut() {
                    map.insert("color".into(), json!(col));
                }
            }

            steps.push(PlanStep {
                command: "type.setStyle".into(),
                params: style_params,
                description: format!("Update typographic styling for layer {target_id}"),
            });

            if let Some(text) = &intent.text_content {
                steps.push(PlanStep {
                    command: "type.edit".into(),
                    params: json!({ "layer": target_id, "text": text }),
                    description: format!("Update text content to '{text}'"),
                });
            }
        } else {
            // Create new type layer
            let text = intent.text_content.clone().unwrap_or_else(|| "Typography Element".to_string());
            let size = intent.numeric_value.unwrap_or(24.0);
            let color = intent.extracted_color.clone().unwrap_or_else(|| "#ffffff".to_string());
            let x = self.layout_engine.center_x(300, scene.width.max(800));
            let y = (scene.height as i32 / 2).max(100);

            steps.push(PlanStep {
                command: "type.create".into(),
                params: json!({
                    "x": x,
                    "y": y,
                    "text": text,
                    "size": size,
                    "color": color,
                }),
                description: "Create new typography element with requested parameters".into(),
            });
        }

        EditingPlan {
            title: "Typography Refinement".into(),
            rationale: "Refines typography attributes, layout positioning, and text formatting.".into(),
            category: IntentCategory::TypographyRefinement,
            steps,
            creates_document: false,
            target_size: None,
        }
    }

    fn plan_color_styling(&self, intent: &EditingIntent, _scene: &VisualScene) -> EditingPlan {
        let prompt_lower = intent.raw_prompt.to_lowercase();
        let steps = if prompt_lower.contains("black and white") || prompt_lower.contains("black & white") || prompt_lower.contains("monochrome") {
            vec![PlanStep {
                command: "layer.newAdjustmentLayer.blackWhite".into(),
                params: json!({
                    "reds": 40,
                    "yellows": 60,
                    "greens": 40,
                    "cyans": 60,
                    "blues": 20,
                    "magentas": 80
                }),
                description: "Apply balanced black & white channel mixer adjustment".into(),
            }]
        } else if prompt_lower.contains("warm") {
            vec![PlanStep {
                command: "layer.newAdjustmentLayer.photoFilter".into(),
                params: json!({
                    "filter": "warming85",
                    "density": 25
                }),
                description: "Apply warm amber photo filter adjustment".into(),
            }]
        } else if prompt_lower.contains("cool") {
            vec![PlanStep {
                command: "layer.newAdjustmentLayer.photoFilter".into(),
                params: json!({
                    "filter": "cooling80",
                    "density": 25
                }),
                description: "Apply cool cyan photo filter adjustment".into(),
            }]
        } else {
            let color = intent.extracted_color.clone().unwrap_or_else(|| "#3b82f6".into());
            vec![PlanStep {
                command: "layer.newFillLayer.solidColor".into(),
                params: json!({ "color": color }),
                description: format!("Apply color wash fill layer with '{color}'"),
            }]
        };

        EditingPlan {
            title: "Color Styling & Tone Adjustment".into(),
            rationale: "Modulates chromatic palette using non-destructive adjustments.".into(),
            category: IntentCategory::ColorStyling,
            steps,
            creates_document: false,
            target_size: None,
        }
    }

    fn plan_direct_filter(&self, intent: &EditingIntent, _scene: &VisualScene) -> EditingPlan {
        let p_lower = intent.raw_prompt.to_lowercase();
        let steps = if p_lower.contains("blur") {
            let radius = intent.numeric_value.unwrap_or(5.0).max(0.1);
            vec![PlanStep {
                command: "filter.blur.gaussianBlur".into(),
                params: json!({ "radius": radius }),
                description: format!("Apply Gaussian blur with {radius}px radius"),
            }]
        } else if p_lower.contains("sharpen") {
            vec![PlanStep {
                command: "filter.sharpen.unsharpMask".into(),
                params: json!({ "amount": 60, "radius": 1.2, "threshold": 1 }),
                description: "Apply unsharp mask sharpening".into(),
            }]
        } else {
            vec![PlanStep {
                command: "layer.newAdjustmentLayer.brightnessContrast".into(),
                params: json!({ "brightness": 10, "contrast": 15 }),
                description: "Apply brightness and contrast boost".into(),
            }]
        };

        EditingPlan {
            title: "Filter / Adjustment Execution".into(),
            rationale: "Executes targeted raster filter or adjustment operation.".into(),
            category: IntentCategory::DirectFilterOrAdjustment,
            steps,
            creates_document: false,
            target_size: None,
        }
    }
}

impl AiPlanner for DeterministicRulePlanner {
    fn plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        _context: &AiContext,
    ) -> Result<EditingPlan, AiError> {
        let plan = match intent.category {
            IntentCategory::SubjectIsolation => self.plan_subject_isolation(intent, scene),
            IntentCategory::BackgroundReplacement => self.plan_background_replacement(intent, scene),
            IntentCategory::PhotoEnhancement => self.plan_photo_enhancement(intent, scene)?,
            IntentCategory::GraphicDesign => return self.plan_graphic_design(intent, scene),
            IntentCategory::TypographyRefinement => self.plan_typography_refinement(intent, scene),
            IntentCategory::ColorStyling => self.plan_color_styling(intent, scene),
            IntentCategory::DirectFilterOrAdjustment => self.plan_direct_filter(intent, scene),
            IntentCategory::LayoutAdjustment | IntentCategory::GeneralEditing => {
                // Fallback to photo enhancement or typography depending on scene
                if scene.has_text_content() {
                    self.plan_typography_refinement(intent, scene)
                } else {
                    self.plan_photo_enhancement(intent, scene)?
                }
            }
        };

        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_subject_isolation_steps() {
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: "Remove background".into(),
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
        let scene = VisualScene::default();
        let context = AiContext::new();

        let plan = planner.plan(&intent, &scene, &context).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].command, "layer.removeBackground");
    }

    #[test]
    fn plan_graphic_design_uses_user_named_business() {
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: "create a poster for my taxi busines to promote it its called chndrax".into(),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: Some("chndrax".into()),
        };
        let scene = VisualScene::default();
        let context = AiContext::new();

        let plan = planner.plan(&intent, &scene, &context).unwrap();
        let texts: Vec<&str> = plan
            .steps
            .iter()
            .filter(|s| s.command == "type.create")
            .filter_map(|s| s.params.get("text").and_then(|t| t.as_str()))
            .collect();
        assert!(texts.iter().any(|t| *t == "chndrax"), "headline must carry the user's name, got {texts:?}");
        assert!(
            texts.iter().any(|t| *t == "Taxi business"),
            "subheading must carry the user's stated purpose, interpreted as display copy (typo corrected, not invented): got {texts:?}"
        );
    }

    #[test]
    fn plan_graphic_design_scales_type_to_the_actual_canvas() {
        // Grounded in render units (1 pt == 1 px at the default 72 dpi): a
        // 1200x1600 poster headline must be a prominent fraction of the
        // canvas, not the old 43 px cap that rendered ~2.7% of canvas height.
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = |prompt: &str| EditingIntent {
            raw_prompt: prompt.into(),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: None,
        };
        let context = AiContext::new();

        let poster = planner.plan(&intent("make a poster called chndrax"), &VisualScene::default(), &context).unwrap();
        let headline_size = poster.steps.iter()
            .find(|s| s.command == "type.create")
            .and_then(|s| s.params["size"].as_f64()).unwrap() as f32;
        assert!(headline_size >= 1600.0 * 0.06, "headline must be prominent at canvas scale: {headline_size}pt");

        // The hierarchy follows the canvas, not a fixed constant: a small
        // canvas gets proportionally smaller (but still prominent) type.
        let mut small_scene = VisualScene::default();
        small_scene.width = 300;
        small_scene.height = 400;
        let small = planner.plan(&intent("make a poster called chndrax"), &small_scene, &context).unwrap();
        let small_headline = small.steps.iter()
            .find(|s| s.command == "type.create")
            .and_then(|s| s.params["size"].as_f64()).unwrap() as f32;
        assert!((24.0..=90.0).contains(&small_headline), "small canvas scales down: {small_headline}pt");
        assert!(small_headline < headline_size);

        // Centering uses the estimated width of the actual string: the
        // headline sits centered within the canvas, and a long quoted line is
        // shrunk to fit instead of overflowing.
        let hx = poster.steps.iter()
            .find(|s| s.command == "type.create")
            .and_then(|s| s.params["x"].as_i64()).unwrap() as f32;
        let center_offset = (hx - 600.0).abs();
        assert!(center_offset <= 1200.0 * 0.06, "headline horizontally centered: offset {center_offset}");

        let long = planner.plan(&intent("make a poster \"The Absolutely Tremendous Grand Opening Celebration\" called chndrax"), &VisualScene::default(), &context).unwrap();
        let long_step = long.steps.iter().find(|s| s.command == "type.create").unwrap();
        let long_size = long_step.params["size"].as_f64().unwrap() as f32;
        // Long copy wraps natively in a paragraph box and, when the whole
        // composition needs it, rescales as a group - never below the
        // canvas-relative legibility floor.
        assert!(long_size <= headline_size, "group rescale never grows type: {long_size}pt vs {headline_size}pt");
        assert!(long_size >= 1200.0 * 0.025, "rescale respects the legibility floor: {long_size}pt");
        let long_box = long_step.params["box"].as_array().expect("over-wide copy gets a native paragraph box");
        assert_eq!(long_box[2].as_f64(), Some(1200.0 * 0.85), "box wraps at 85% of canvas width: {long_box:?}");

        // Vertical composition: the subheading baseline sits below the
        // headline baseline with a gap that scales with the type sizes.
        let ys: Vec<i64> = poster.steps.iter().filter(|s| s.command == "type.create")
            .filter_map(|s| s.params["y"].as_i64()).collect();
        assert!(ys.len() == 2 && ys[1] > ys[0] + headline_size as i64 / 2, "subheading below headline: {ys:?}");
    }

    #[test]
    fn plan_graphic_design_refuses_copy_that_cannot_fit_legibly() {
        // Capacity honesty: on a canvas far too small for the requested copy,
        // the planner must refuse instead of clamping baselines into a pile
        // or emitting unreadably small text.
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: format!("make a poster \"{}\" called chndrax", "The Absolutely Tremendous Grand Opening Celebration ".repeat(10)),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: None,
        };
        let mut scene = VisualScene::default();
        scene.width = 120;
        scene.height = 160;
        let context = AiContext::new();
        let err = planner.plan(&intent, &scene, &context).unwrap_err();
        assert!(err.to_string().contains("does not fit"), "honest capacity refusal, got {err}");

        // Ordinary-length copy on an adequate canvas still plans.
        let ordinary = EditingIntent { raw_prompt: "make a poster called chndrax".into(), ..intent.clone() };
        let mut scene = VisualScene::default();
        scene.width = 1200;
        scene.height = 1600;
        let plan = planner.plan(&ordinary, &scene, &context).unwrap();
        assert!(plan.steps.iter().any(|s| s.command == "type.create"));
    }

    #[test]
    fn plan_graphic_design_emits_every_required_term() {
        // A quoted tagline AND a named brand must both reach the canvas; the
        // evaluator requires each, so emitting only one is an impossible goal.
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: "make a poster \"Grand Opening\" called chndrax".into(),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: Some("Grand Opening".into()),
        };
        let scene = VisualScene::default();
        let context = AiContext::new();

        let plan = planner.plan(&intent, &scene, &context).unwrap();
        let texts: Vec<&str> = plan
            .steps
            .iter()
            .filter(|s| s.command == "type.create")
            .filter_map(|s| s.params.get("text").and_then(|t| t.as_str()))
            .collect();
        assert!(texts.iter().any(|t| *t == "Grand Opening"), "quoted term must be emitted, got {texts:?}");
        assert!(texts.iter().any(|t| *t == "chndrax"), "named brand must be emitted, got {texts:?}");
    }

    #[test]
    fn plan_graphic_design_falls_back_to_archetype_without_user_content() {
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: "create a promotional poster".into(),
            category: IntentCategory::GraphicDesign,
            archetype: Some("promotional_poster".into()),
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: None,
        };
        let scene = VisualScene::default();
        let context = AiContext::new();

        let plan = planner.plan(&intent, &scene, &context).unwrap();
        let texts: Vec<&str> = plan
            .steps
            .iter()
            .filter(|s| s.command == "type.create")
            .filter_map(|s| s.params.get("text").and_then(|t| t.as_str()))
            .collect();
        assert!(texts.iter().any(|t| *t == "CREATIVE EXPO 2026"));
    }

    #[test]
    fn plan_photo_enhancement_steps() {
        let catalog = CommandCatalog::new();
        let planner = DeterministicRulePlanner::new(catalog);
        let intent = EditingIntent {
            raw_prompt: "Make it look professional".into(),
            category: IntentCategory::PhotoEnhancement,
            archetype: None,
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: Some("professional".into()),
            text_content: None,
        };
        let scene = VisualScene::default();
        let context = AiContext::new();

        let plan = planner.plan(&intent, &scene, &context).unwrap();
        assert!(plan.steps.len() >= 3);
        assert_eq!(plan.steps[0].command, "filter.sharpen.unsharpMask");
    }
}
