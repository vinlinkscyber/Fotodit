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
use super::natural_language::{EditingIntent, IntentCategory};
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

    fn plan_photo_enhancement(&self, _intent: &EditingIntent, _scene: &VisualScene) -> EditingPlan {
        let steps = vec![
            PlanStep {
                command: "layer.newAdjustmentLayer.curves".into(),
                params: json!({
                    "points": [[0, 0], [64, 55], [192, 205], [255, 255]]
                }),
                description: "Subtle S-curve tonal contrast enhancement for highlights and shadows".into(),
            },
            PlanStep {
                command: "layer.newAdjustmentLayer.vibrance".into(),
                params: json!({
                    "vibrance": 18,
                    "saturation": 4
                }),
                description: "Enhance color vibrance safely protecting skin tones".into(),
            },
            PlanStep {
                command: "filter.unsharpMask".into(),
                params: json!({
                    "amount": 40,
                    "radius": 1.0,
                    "threshold": 2
                }),
                description: "Unsharp mask clarity sharpening on focal features".into(),
            },
        ];

        EditingPlan {
            title: "Professional Photographic Enhancement & Clarity".into(),
            rationale: "Applies studio-grade tonal grading via non-destructive Curves, Vibrance adjustment, and gentle unsharp masking.".into(),
            category: IntentCategory::PhotoEnhancement,
            steps,
            creates_document: false,
            target_size: None,
        }
    }

    fn plan_graphic_design(&self, intent: &EditingIntent, scene: &VisualScene) -> EditingPlan {
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
        let hierarchy = TypographyHierarchy::compute(height, Some(archetype_kind));

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

        // 2. Headline text
        let headline_text = match archetype_kind {
            ArchetypeKind::WeddingInvitation => "TOGETHER WITH THEIR FAMILIES",
            ArchetypeKind::PromotionalPoster => "CREATIVE EXPO 2026",
            ArchetypeKind::ProductAdvertising => "INNOVATION DEFINED",
            _ => "FEATURED PRESENTATION",
        };

        let headline_x = self.layout_engine.center_x(400, width);
        let headline_y = ((height as f32) * 0.22).round() as i32;
        let headline_color = archetype.primary_palette.get(1).cloned().unwrap_or_else(|| "#ffffff".into());

        steps.push(PlanStep {
            command: "type.create".into(),
            params: json!({
                "x": headline_x,
                "y": headline_y,
                "text": headline_text,
                "font": hierarchy.heading_font,
                "sizePt": hierarchy.heading_pt,
                "color": headline_color,
            }),
            description: format!("Create primary headline with '{headline_text}' in {}", hierarchy.heading_font),
        });

        // 3. Subheading text
        let subhead_text = match archetype_kind {
            ArchetypeKind::WeddingInvitation => "Request the Honor of Your Presence",
            ArchetypeKind::PromotionalPoster => "A Celebration of Art & Technology",
            ArchetypeKind::ProductAdvertising => "Engineered for Performance",
            _ => "Discover the Possibilities",
        };

        let subhead_x = self.layout_engine.center_x(450, width);
        let subhead_y = ((height as f32) * 0.35).round() as i32;
        let subhead_color = archetype.primary_palette.get(2).cloned().unwrap_or_else(|| "#94a3b8".into());

        steps.push(PlanStep {
            command: "type.create".into(),
            params: json!({
                "x": subhead_x,
                "y": subhead_y,
                "text": subhead_text,
                "font": hierarchy.body_font,
                "sizePt": hierarchy.subheading_pt,
                "color": subhead_color,
            }),
            description: format!("Create subheading with '{subhead_text}'"),
        });

        EditingPlan {
            title: format!("{} Composition", archetype.name),
            rationale: format!(
                "Constructs a balanced {} layout with {} typography hierarchy, calibrated whitespace margins, and harmonic palette.",
                archetype.name, hierarchy.heading_font
            ),
            category: IntentCategory::GraphicDesign,
            steps,
            creates_document: false,
            target_size: Some([width, height]),
        }
    }

    fn plan_typography_refinement(&self, intent: &EditingIntent, scene: &VisualScene) -> EditingPlan {
        let mut steps = Vec::new();

        if let Some(target_id) = intent.target_layer_id {
            // Edit existing layer
            let mut style_params = json!({ "layer": target_id });
            if let Some(sz) = intent.numeric_value {
                if let Some(map) = style_params.as_object_mut() {
                    map.insert("sizePt".into(), json!(sz));
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
                    "sizePt": size,
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
                command: "filter.gaussianBlur".into(),
                params: json!({ "radius": radius }),
                description: format!("Apply Gaussian blur with {radius}px radius"),
            }]
        } else if p_lower.contains("sharpen") {
            vec![PlanStep {
                command: "filter.unsharpMask".into(),
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
            IntentCategory::PhotoEnhancement => self.plan_photo_enhancement(intent, scene),
            IntentCategory::GraphicDesign => self.plan_graphic_design(intent, scene),
            IntentCategory::TypographyRefinement => self.plan_typography_refinement(intent, scene),
            IntentCategory::ColorStyling => self.plan_color_styling(intent, scene),
            IntentCategory::DirectFilterOrAdjustment => self.plan_direct_filter(intent, scene),
            IntentCategory::LayoutAdjustment | IntentCategory::GeneralEditing => {
                // Fallback to photo enhancement or typography depending on scene
                if scene.has_text_content() {
                    self.plan_typography_refinement(intent, scene)
                } else {
                    self.plan_photo_enhancement(intent, scene)
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
        assert_eq!(plan.steps[0].command, "layer.newAdjustmentLayer.curves");
    }
}
