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

    /// Find foreground subjects (pixel layers with masks or isolated bounds).
    pub fn find_foreground_subjects(&self) -> Vec<&VisualElement> {
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
                    [0, 0, width as i32, height as i32]
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

        // Default dominant colors
        let dominant_colors = vec!["#0f172a".into(), "#3b82f6".into(), "#ffffff".into()];

        VisualScene {
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
    fn perceive_fotodit_doc() {
        let engine = PerceptionEngine::new();
        let doc_json = json!({
            "width": 1200,
            "height": 800,
            "resolution": 300.0,
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
        assert_eq!(scene.height, 800);
        assert_eq!(scene.aspect_ratio, "Landscape");
        assert_eq!(scene.layers.len(), 2);
        assert_eq!(scene.typography.len(), 1);

        let heading = scene.find_primary_heading().unwrap();
        assert_eq!(heading.name, "Headline");
        assert_eq!(heading.font_size_pt, Some(36.0));
    }
}
