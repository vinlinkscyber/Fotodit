//! Subsystem B: Visual Perception Engine
//!
//! Inspects document hierarchy, analyzes spatial composition and visual hierarchy, detects
//! foreground subjects and background layers, extracts typography inventory and coordinates,
//! and evaluates margins, alignment and dominant colors.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Aspect ratio classification of the document canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectRatio {
    Square,
    Portrait,
    Landscape,
    WideBanner,
}

/// One visual element in the document scene.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VisualElement {
    pub id: u64,
    pub name: String,
    pub kind: String,
    pub text_content: Option<String>,
    pub font_size_pt: Option<f32>,
    /// `[x, y, width, height]`
    pub bounds: [i32; 4],
    pub is_visible: bool,
    pub has_mask: bool,
    /// 1.0: every field here is parsed directly from engine `inspect::document`
    /// JSON, not inferred. This is record fidelity only - it says nothing about
    /// semantic detection certainty. Heuristic interpretation lives on finder
    /// results (`SubjectCandidate`), not on the element record itself.
    #[serde(default = "full_confidence")]
    pub confidence: f32,
}

fn full_confidence() -> f32 {
    1.0
}

/// How a foreground-subject candidate was identified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectBasis {
    /// The layer carries a layer mask, which suggests an isolated subject.
    HasMask,
    /// The pixel layer sits fully inside the canvas without touching its edges.
    IsolatedBounds,
    /// Guessed from the layer name (anything not named "background").
    NameHeuristic,
}

/// A foreground-subject candidate with explicit basis and confidence. The
/// confidence values are uncalibrated heuristic weights chosen by convention,
/// not measured probabilities, and no pixel-level subject detector runs in
/// this engine yet. Consumers must not treat candidates as detected subjects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubjectCandidate {
    pub layer_id: u64,
    pub confidence: f32,
    pub basis: SubjectBasis,
}

/// Typography inventory item extracted from the document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypographyElement {
    pub layer_id: u64,
    pub text: String,
    pub font_family: String,
    pub size_pt: f32,
    pub bounds: [i32; 4],
    pub is_heading: bool,
}

/// Document visual structure and spatial layout summary.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentStructure {
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: String,
    pub total_layers: usize,
    pub has_background: bool,
    pub has_mask: bool,
    pub has_text: bool,
}

/// Synthesized visual scene understanding.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VisualScene {
    /// Engine document revision at perception time (`inspect::document`
    /// `revision`); `None` when the host did not supply it. Findings derived
    /// from this scene are valid for this revision only.
    #[serde(default)]
    pub revision: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: String,
    pub resolution_dpi: f64,
    pub active_layer_id: Option<u64>,
    pub layers: Vec<VisualElement>,
    pub typography: Vec<TypographyElement>,
    pub dominant_colors: Vec<String>,
    /// `[top, right, bottom, left]` estimated empty margins
    pub whitespace_margins: [i32; 4],
    pub has_selection: bool,
    pub selection_bounds: Option<[i32; 4]>,
}

impl VisualScene {
    /// Find the primary heading in the typography inventory.
    pub fn find_primary_heading(&self) -> Option<&VisualElement> {
        let text_layers: Vec<&VisualElement> = self.layers.iter().filter(|l| l.kind == "Type" && l.is_visible).collect();
        // Return text layer with largest font size or positioned nearest to top
        text_layers
            .iter()
            .max_by(|a, b| {
                let size_a = a.font_size_pt.unwrap_or(12.0);
                let size_b = b.font_size_pt.unwrap_or(12.0);
                size_a.partial_cmp(&size_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Find the bottom or background layer.
    pub fn find_background_layer(&self) -> Option<&VisualElement> {
        self.layers.iter().find(|l| l.name.to_lowercase() == "background" || (l.kind == "Pixel" && l.bounds[0] == 0 && l.bounds[1] == 0))
    }

    /// Find foreground subjects (pixel layers with masks or non-background
    /// names). Heuristic: see [`Self::foreground_candidates`] for the explicit
    /// basis and confidence of each candidate.
    pub fn find_foreground_subjects(&self) -> Vec<&VisualElement> {
        self.layers
            .iter()
            .filter(|l| (l.kind == "Pixel" || l.kind == "Raster") && (l.has_mask || l.name.to_lowercase() != "background"))
            .collect()
    }

    /// Foreground-subject candidates with explicit basis and confidence.
    ///
    /// No pixel-level subject detection runs here; every candidate is inferred
    /// from document structure (masks, bounds, names). `confidence` is an
    /// uncalibrated weight of the heuristic that fired, not a measured
    /// probability of a real subject.
    pub fn foreground_candidates(&self) -> Vec<SubjectCandidate> {
        let mut out = Vec::new();
        for l in &self.layers {
            if l.kind != "Pixel" && l.kind != "Raster" {
                continue;
            }
            let basis = if l.has_mask {
                SubjectBasis::HasMask
            } else if l.bounds[0] > 0
                && l.bounds[1] > 0
                && l.bounds[0] + l.bounds[2] < self.width as i32
                && l.bounds[1] + l.bounds[3] < self.height as i32
            {
                SubjectBasis::IsolatedBounds
            } else if l.name.to_lowercase() != "background" {
                SubjectBasis::NameHeuristic
            } else {
                continue;
            };
            let confidence = match basis {
                SubjectBasis::HasMask => 0.7,
                SubjectBasis::IsolatedBounds => 0.5,
                SubjectBasis::NameHeuristic => 0.25,
            };
            out.push(SubjectCandidate { layer_id: l.id, confidence, basis });
        }
        out
    }

    #[allow(dead_code)]
    fn find_foreground_subjects_legacy(&self) -> Vec<&VisualElement> {
        self.layers
            .iter()
            .filter(|l| (l.kind == "Pixel" || l.kind == "Raster") && (l.has_mask || l.name.to_lowercase() != "background"))
            .collect()
    }

    /// Check if the canvas has any text.
    pub fn has_text_content(&self) -> bool {
        !self.typography.is_empty()
    }
}

/// Visual Perception Engine.
#[derive(Clone, Debug, Default)]
pub struct PerceptionEngine;

impl PerceptionEngine {
    /// Create new perception engine.
    pub fn new() -> Self {
        Self
    }

    /// Analyze document inspect JSON into a rich `VisualScene`.
    pub fn perceive(&self, doc_json: &Value) -> VisualScene {
        let width = doc_json.get("width").and_then(Value::as_u64).unwrap_or(800) as u32;
        let height = doc_json.get("height").and_then(Value::as_u64).unwrap_or(600) as u32;
        let resolution_dpi = doc_json.get("resolution").and_then(Value::as_f64).unwrap_or(72.0);
        let active_layer_id = doc_json.get("activeLayer").and_then(Value::as_u64);
        let revision = doc_json.get("revision").and_then(Value::as_u64);

        let aspect_ratio = if width == height {
            "Square"
        } else if width > height * 2 {
            "WideBanner"
        } else if width > height {
            "Landscape"
        } else {
            "Portrait"
        }
        .to_string();

        let mut visual_elements = Vec::new();
        let mut typography = Vec::new();

        if let Some(layers_arr) = doc_json.get("layers").and_then(Value::as_array) {
            for layer_val in layers_arr {
                let id = layer_val.get("id").and_then(Value::as_u64).unwrap_or(0);
                let name = layer_val.get("name").and_then(Value::as_str).unwrap_or("Layer").to_string();
                let kind = layer_val.get("kind").and_then(Value::as_str).unwrap_or("Pixel").to_string();
                let is_visible = layer_val.get("visible").and_then(Value::as_bool).unwrap_or(true);
                let has_mask = layer_val.get("hasMask").and_then(Value::as_bool).unwrap_or(false);

                let bounds = if let Some(b) = layer_val.get("bounds").and_then(Value::as_array) {
                    let x0 = b.first().and_then(Value::as_i64).unwrap_or(0) as i32;
                    let y0 = b.get(1).and_then(Value::as_i64).unwrap_or(0) as i32;
                    let w = b.get(2).and_then(Value::as_i64).unwrap_or(0) as i32;
                    let h = b.get(3).and_then(Value::as_i64).unwrap_or(0) as i32;
                    [x0, y0, w, h]
                } else {
                    // Absent or null inspect geometry is UNKNOWN, not
                    // canvas-sized content: record zero bounds so consumers
                    // (the goal evaluator) withhold geometry-based
                    // satisfaction instead of counting unmeasured ink as a
                    // full-canvas pass.
                    [0, 0, 0, 0]
                };

                let mut text_content = None;
                let mut font_size_pt = None;

                if let Some(text_obj) = layer_val.get("text") {
                    let text_str = text_obj.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                    let font = text_obj.get("font").and_then(Value::as_str).unwrap_or("Inter").to_string();
                    let size = text_obj.get("sizePt").and_then(Value::as_f64).unwrap_or(12.0) as f32;

                    typography.push(TypographyElement {
                        layer_id: id,
                        text: text_str.clone(),
                        font_family: font,
                        size_pt: size,
                        bounds,
                        is_heading: size >= 20.0,
                    });

                    text_content = Some(text_str);
                    font_size_pt = Some(size);
                }

                visual_elements.push(VisualElement {
                    id,
                    name,
                    kind,
                    text_content,
                    font_size_pt,
                    bounds,
                    is_visible,
                    has_mask,
                    confidence: 1.0,
                });
            }
        }

        // Calculate whitespace / margins
        let min_x = visual_elements.iter().map(|e| e.bounds[0]).min().unwrap_or(0).max(0);
        let min_y = visual_elements.iter().map(|e| e.bounds[1]).min().unwrap_or(0).max(0);
        let max_x = visual_elements.iter().map(|e| e.bounds[0] + e.bounds[2]).max().unwrap_or(width as i32);
        let max_y = visual_elements.iter().map(|e| e.bounds[1] + e.bounds[3]).max().unwrap_or(height as i32);

        let margin_right = ((width as i32) - max_x).max(0);
        let margin_bottom = ((height as i32) - max_y).max(0);
        let whitespace_margins = [min_y, margin_right, margin_bottom, min_x];

        let has_selection = doc_json.get("hasSelection").and_then(Value::as_bool).unwrap_or(false);
        let selection_bounds = doc_json.get("selectionBounds").and_then(Value::as_array).and_then(|arr| {
            if arr.len() == 4 {
                Some([
                    arr[0].as_i64()? as i32,
                    arr[1].as_i64()? as i32,
                    arr[2].as_i64()? as i32,
                    arr[3].as_i64()? as i32,
                ])
            } else {
                None
            }
        });

        // No pixel data is available in `inspect::document` JSON, so dominant
        // colors cannot be computed here; leave empty instead of inventing
        // values. The honest source is an engine-side histogram/color query
        // (tracked as a proposed `crates/engine` addition).
        let dominant_colors = Vec::new();

        VisualScene {
            revision,
            width,
            height,
            aspect_ratio,
            resolution_dpi,
            active_layer_id,
            layers: visual_elements,
            typography,
            dominant_colors,
            whitespace_margins,
            has_selection,
            selection_bounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn perception_records_absent_bounds_as_unknown_zero_geometry() {
        // A layer whose inspect JSON carries no bounds (or null) has no
        // measured ink: the scene must record zero geometry, never the
        // canvas-sized fallback that used to let unknown geometry count as
        // measured, fully-on-canvas content.
        let engine = PerceptionEngine::new();
        let doc = json!({
            "width": 1200,
            "height": 800,
            "layers": [
                { "id": 1, "name": "Text A", "kind": "Type", "visible": true, "hasMask": false,
                  "text": { "text": "chndrax", "font": "Inter", "sizePt": 120 } },
                { "id": 2, "name": "Text B", "kind": "Type", "visible": true, "hasMask": false, "bounds": null,
                  "text": { "text": "taxi", "font": "Inter", "sizePt": 60 } },
                { "id": 3, "name": "Text C", "kind": "Type", "visible": true, "hasMask": false, "bounds": [10, 20, 300, 100],
                  "text": { "text": "measured", "font": "Inter", "sizePt": 60 } }
            ]
        });
        let scene = engine.perceive(&doc);
        let by_id = |id: u64| scene.layers.iter().find(|l| l.id == id).unwrap().bounds;
        assert_eq!(by_id(1), [0, 0, 0, 0], "absent bounds are unknown, not canvas-sized");
        assert_eq!(by_id(2), [0, 0, 0, 0], "null bounds are unknown, not canvas-sized");
        assert_eq!(by_id(3), [10, 20, 300, 100], "measured bounds pass through");
    }

    #[test]
    fn perceive_fotodit_doc() {
        let engine = PerceptionEngine::new();
        let doc_json = json!({
            "width": 1200,
            "height": 800,
            "resolution": 300.0,
            "revision": 7,
            "activeLayer": 2,
            "layers": [
                {
                    "id": 1,
                    "name": "Background",
                    "kind": "Pixel",
                    "bounds": [0, 0, 1200, 800],
                    "visible": true,
                    "hasMask": false
                },
                {
                    "id": 2,
                    "name": "Headline",
                    "kind": "Type",
                    "text": {
                        "text": "Grand Gala Evening",
                        "font": "Cinzel",
                        "sizePt": 36.0
                    },
                    "bounds": [100, 150, 600, 60],
                    "visible": true,
                    "hasMask": false
                }
            ]
        });

        let scene = engine.perceive(&doc_json);
        assert_eq!(scene.width, 1200);
        assert_eq!(scene.revision, Some(7));
        assert!(scene.layers.iter().all(|l| l.confidence == 1.0));
        // No pixel data in the inspect JSON: dominant colors stay empty
        // instead of carrying invented values.
        assert!(scene.dominant_colors.is_empty());
        assert_eq!(scene.height, 800);
        assert_eq!(scene.aspect_ratio, "Landscape");
        assert_eq!(scene.layers.len(), 2);
        assert_eq!(scene.typography.len(), 1);

        let heading = scene.find_primary_heading().unwrap();
        assert_eq!(heading.name, "Headline");
        assert_eq!(heading.font_size_pt, Some(36.0));
    }

    #[test]
    fn foreground_candidates_carry_basis_and_confidence() {
        let engine = PerceptionEngine::new();
        let doc_json = json!({
            "width": 1200,
            "height": 800,
            "layers": [
                { "id": 1, "name": "Background", "kind": "Pixel", "bounds": [0, 0, 1200, 800], "visible": true, "hasMask": false },
                { "id": 2, "name": "Cutout", "kind": "Pixel", "bounds": [200, 100, 400, 500], "visible": true, "hasMask": true },
                { "id": 3, "name": "Logo", "kind": "Pixel", "bounds": [900, 600, 200, 100], "visible": true, "hasMask": false },
                { "id": 4, "name": "Overlay", "kind": "Pixel", "bounds": [0, 0, 1200, 800], "visible": true, "hasMask": false }
            ]
        });
        let scene = engine.perceive(&doc_json);
        let c = scene.foreground_candidates();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].layer_id, 2);
        assert_eq!(c[0].basis, SubjectBasis::HasMask);
        assert!(c[0].confidence > c[1].confidence);
        assert_eq!(c[1].layer_id, 3);
        assert_eq!(c[1].basis, SubjectBasis::IsolatedBounds);
        assert_eq!(c[2].layer_id, 4);
        assert_eq!(c[2].basis, SubjectBasis::NameHeuristic);
    }
}
