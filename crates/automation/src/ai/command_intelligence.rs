//! Subsystem C: Command Intelligence & Capability Index
//!
//! Indexes native Fotodit engine commands, validates parameters against exact engine requirements,
//! provides searchable capability discovery, and guarantees that commands and parameters are never
//! hallucinated or malformed.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::AiError;

/// Capability metadata for a registered Fotodit native command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandCapability {
    pub id: String,
    pub label: String,
    pub category: String,
    pub params_schema: String,
    pub description: String,
}

/// A validated native command ready for transactional dispatch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidatedCommand {
    pub id: String,
    pub params: Value,
}

/// Capability index and parameter validator.
#[derive(Clone, Debug)]
pub struct CommandCatalog {
    commands: HashMap<String, CommandCapability>,
}

impl Default for CommandCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandCatalog {
    /// Initialize the catalog with Fotodit's native command capabilities.
    pub fn new() -> Self {
        let mut commands = HashMap::new();

        let entries = vec![
            CommandCapability {
                id: "file.new".into(),
                label: "New…".into(),
                category: "File".into(),
                params_schema: r#"{"width": 1..300000=1920, "height": 1..300000=1080, "mode": "rgb|gray|cmyk|lab"="rgb", "background": "white|black|transparent|#rrggbb"="white"}"#.into(),
                description: "Create a new document with specified dimensions, color mode, and background".into(),
            },
            CommandCapability {
                id: "layer.removeBackground".into(),
                label: "Remove Background".into(),
                category: "Cutout".into(),
                params_schema: r#"{"layer": id?, "sampleAllLayers": bool, "refine": bool}"#.into(),
                description: "Non-destructive subject isolation and background removal with alpha edge refinement".into(),
            },
            CommandCapability {
                id: "type.create".into(),
                label: "New Type Layer".into(),
                category: "Type".into(),
                params_schema: r##"{"x": px, "y": px, "text": str, "font": str?, "size": pt?, "color": "#rrggbb"?}"##.into(),
                description: "Create an editable vector type layer with position, font, size, and color".into(),
            },
            CommandCapability {
                id: "type.edit".into(),
                label: "Edit Type".into(),
                category: "Type".into(),
                params_schema: r#"{"layer": id?, "text": str}"#.into(),
                description: "Edit text content of an existing type layer preserving formatting".into(),
            },
            CommandCapability {
                id: "type.setStyle".into(),
                label: "Set Type Style".into(),
                category: "Type".into(),
                params_schema: r##"{"layer": id?, "font": str?, "size": pt?, "color": "#rrggbb"?}"##.into(),
                description: "Update typography styling attributes of a type layer".into(),
            },
            CommandCapability {
                id: "layer.newFillLayer.solidColor".into(),
                label: "Solid Color…".into(),
                category: "Layer".into(),
                params_schema: r##"{"color": "#rrggbb"}"##.into(),
                description: "Add a non-destructive solid color fill layer".into(),
            },
            CommandCapability {
                id: "layer.newFillLayer.gradient".into(),
                label: "Gradient…".into(),
                category: "Layer".into(),
                params_schema: r##"{"from": "#rrggbb", "to": "#rrggbb", "angle": deg=90, "style": "linear|radial"="linear"}"##.into(),
                description: "Add a non-destructive gradient fill layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.brightnessContrast".into(),
                label: "Brightness/Contrast…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"brightness": -150..150=0, "contrast": -50..100=0}"#.into(),
                description: "Add a non-destructive Brightness/Contrast adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.curves".into(),
                label: "Curves…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"points": [[x,y],...]}"#.into(),
                description: "Add a non-destructive Curves tonal adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.levels".into(),
                label: "Levels…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"inBlack": 0..253=0, "gamma": 0.01..9.99=1.0, "inWhite": 2..255=255}"#.into(),
                description: "Add a non-destructive Levels adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.vibrance".into(),
                label: "Vibrance…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"vibrance": -100..100=0, "saturation": -100..100=0}"#.into(),
                description: "Add a non-destructive Vibrance and Saturation adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.hueSaturation".into(),
                label: "Hue/Saturation…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"hue": -180..180=0, "saturation": -100..100=0, "lightness": -100..100=0}"#.into(),
                description: "Add a non-destructive Hue/Saturation adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.colorBalance".into(),
                label: "Color Balance…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"shadows": [c,m,y]?, "midtones": [c,m,y]?, "highlights": [c,m,y]?}"#.into(),
                description: "Add a non-destructive Color Balance adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.blackWhite".into(),
                label: "Black & White…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"reds": 40, "yellows": 60, "greens": 40, "cyans": 60, "blues": 20, "magentas": 80}"#.into(),
                description: "Add a non-destructive Black & White adjustment layer with custom channel mixing".into(),
            },
            CommandCapability {
                id: "layer.newAdjustmentLayer.photoFilter".into(),
                label: "Photo Filter…".into(),
                category: "Adjustment".into(),
                params_schema: r#"{"filter": "warming85|warmingLBA|cooling80|sepia", "density": 1..100=25}"#.into(),
                description: "Add a non-destructive Photo Filter adjustment layer".into(),
            },
            CommandCapability {
                id: "layer.setOpacity".into(),
                label: "Set Opacity".into(),
                category: "Layer".into(),
                params_schema: r#"{"layer": id?, "opacity": 0..100}"#.into(),
                description: "Set the opacity percentage of a layer".into(),
            },
            CommandCapability {
                id: "layer.setBlendMode".into(),
                label: "Set Blend Mode".into(),
                category: "Layer".into(),
                params_schema: r#"{"layer": id?, "mode": "normal|multiply|screen|overlay|softLight|hardLight"}"#.into(),
                description: "Set the blend mode of a layer".into(),
            },
            CommandCapability {
                id: "layer.reorder".into(),
                label: "Reorder Layer".into(),
                category: "Layer".into(),
                params_schema: r#"{"layer": id, "to": index}"#.into(),
                description: "Move a layer to a different position in the layer stack".into(),
            },
            CommandCapability {
                id: "layer.duplicate".into(),
                label: "Duplicate Layer".into(),
                category: "Layer".into(),
                params_schema: r#"{"layer": id?}"#.into(),
                description: "Duplicate a layer in the document".into(),
            },
            CommandCapability {
                id: "filter.blur.gaussianBlur".into(),
                label: "Gaussian Blur…".into(),
                category: "Filter".into(),
                params_schema: r#"{"radius": 0.1..1000=1.0}"#.into(),
                description: "Apply a Gaussian blur filter with specified pixel radius".into(),
            },
            CommandCapability {
                id: "filter.sharpen.unsharpMask".into(),
                label: "Unsharp Mask…".into(),
                category: "Filter".into(),
                params_schema: r#"{"amount": 1..500=50, "radius": 0.1..250.0=1.0, "threshold": 0..255=0}"#.into(),
                description: "Sharpen pixels using the unsharp mask algorithm".into(),
            },
            CommandCapability {
                id: "edit.undo".into(),
                label: "Undo".into(),
                category: "Edit".into(),
                params_schema: r#"{}"#.into(),
                description: "Revert the last executed command on the active document".into(),
            },
            CommandCapability {
                id: "edit.redo".into(),
                label: "Redo".into(),
                category: "Edit".into(),
                params_schema: r#"{}"#.into(),
                description: "Reapply the last undone command on the active document".into(),
            },
        ];

        for entry in entries {
            commands.insert(entry.id.clone(), entry);
        }

        Self { commands }
    }

    /// Check if a command ID exists in the capability catalog.
    pub fn has_command(&self, id: &str) -> bool {
        self.commands.contains_key(id)
    }

    /// Look up capability metadata.
    pub fn get(&self, id: &str) -> Option<&CommandCapability> {
        self.commands.get(id)
    }

    /// Search capabilities by keyword or description query.
    pub fn search(&self, query: &str) -> Vec<&CommandCapability> {
        let q_lower = query.to_lowercase();
        self.commands
            .values()
            .filter(|cap| {
                cap.id.to_lowercase().contains(&q_lower)
                    || cap.label.to_lowercase().contains(&q_lower)
                    || cap.description.to_lowercase().contains(&q_lower)
                    || cap.category.to_lowercase().contains(&q_lower)
            })
            .collect()
    }

    /// Strictly validate command ID and parameters against engine requirements.
    pub fn validate_command(&self, id: &str, params: &Value) -> Result<ValidatedCommand, AiError> {
        if !self.has_command(id) {
            return Err(AiError::CommandResolutionError(format!("unknown or unsupported command '{id}'")));
        }

        if !params.is_object() && !params.is_null() {
            return Err(AiError::ValidationError {
                command: id.to_string(),
                message: "parameters must be a JSON object".into(),
            });
        }

        // Native type commands silently ignore unknown JSON keys. Catch schema
        // drift here so a planner typo such as `sizePt` cannot report success
        // while rendering the engine's default 12pt type.
        if matches!(id, "type.create" | "type.setStyle") {
            const TYPE_KEYS: &[&str] = &[
                "x", "y", "text", "box", "name", "orientation", "layer", "layers", "range",
                "font", "family", "fontStyle", "postscriptName", "weight", "italic", "size",
                "color", "tracking", "leading", "baselineShift", "horizontalScale", "verticalScale",
                "underline", "strikethrough", "fauxBold", "fauxItalic", "kerning", "caps", "baseline",
                "ligatures", "discretionaryLigatures", "features", "variations", "language", "align",
                "firstLineIndent", "startIndent", "endIndent", "spaceBefore", "spaceAfter", "autoLeading",
                "direction", "hyphenate", "metricsAsShown", "leadingBy", "baselineShiftBy", "trackingBy",
            ];
            if let Some(map) = params.as_object() {
                for key in map.keys() {
                    if !TYPE_KEYS.contains(&key.as_str()) {
                        return Err(AiError::ValidationError { command: id.into(), message: format!("unsupported native type parameter '{key}'") });
                    }
                }
                if let Some(size) = map.get("size") {
                    if !size.as_f64().is_some_and(|v| v.is_finite() && (0.1..=1296.0).contains(&v)) {
                        return Err(AiError::ValidationError { command: id.into(), message: "size must be a number between 0.1 and 1296 points".into() });
                    }
                }
                for key in ["font", "family", "fontStyle", "postscriptName"] {
                    if map.get(key).is_some_and(|v| !v.is_string()) {
                        return Err(AiError::ValidationError { command: id.into(), message: format!("'{key}' must be a string") });
                    }
                }
            }
        }

        // Schema-specific validation rules
        match id {
            "file.new" => {
                if let Some(w) = params.get("width").and_then(Value::as_i64) {
                    if !(1..=300000).contains(&w) {
                        return Err(AiError::ValidationError {
                            command: id.into(),
                            message: "width must be between 1 and 300000".into(),
                        });
                    }
                }
                if let Some(h) = params.get("height").and_then(Value::as_i64) {
                    if !(1..=300000).contains(&h) {
                        return Err(AiError::ValidationError {
                            command: id.into(),
                            message: "height must be between 1 and 300000".into(),
                        });
                    }
                }
            }
            "type.create" => {
                let x = params.get("x").and_then(Value::as_i64);
                let y = params.get("y").and_then(Value::as_i64);
                let text = params.get("text").and_then(Value::as_str);

                if x.is_none() || y.is_none() {
                    return Err(AiError::ValidationError {
                        command: id.into(),
                        message: "type.create requires 'x' and 'y' integer coordinates".into(),
                    });
                }
                if text.is_none() || text.is_some_and(|t| t.is_empty()) {
                    return Err(AiError::ValidationError {
                        command: id.into(),
                        message: "type.create requires non-empty 'text'".into(),
                    });
                }
            }
            "layer.newFillLayer.solidColor" => {
                let color = params.get("color").and_then(Value::as_str);
                if color.is_none() || !color.is_some_and(|c| c.starts_with('#')) {
                    return Err(AiError::ValidationError {
                        command: id.into(),
                        message: "solidColor requires a valid hex 'color' string (e.g. '#1e293b')".into(),
                    });
                }
            }
            "layer.newAdjustmentLayer.brightnessContrast" => {
                if let Some(b) = params.get("brightness").and_then(Value::as_i64) {
                    if !(-150..=150).contains(&b) {
                        return Err(AiError::ValidationError {
                            command: id.into(),
                            message: "brightness must be in range -150..=150".into(),
                        });
                    }
                }
                if let Some(c) = params.get("contrast").and_then(Value::as_i64) {
                    if !(-50..=100).contains(&c) {
                        return Err(AiError::ValidationError {
                            command: id.into(),
                            message: "contrast must be in range -50..=100".into(),
                        });
                    }
                }
            }
            "layer.newAdjustmentLayer.vibrance" => {
                if let Some(v) = params.get("vibrance").and_then(Value::as_i64) {
                    if !(-100..=100).contains(&v) {
                        return Err(AiError::ValidationError {
                            command: id.into(),
                            message: "vibrance must be in range -100..=100".into(),
                        });
                    }
                }
            }
            "filter.blur.gaussianBlur" => {
                let radius = params.get("radius").and_then(Value::as_f64);
                if radius.is_none() || radius.is_some_and(|r| !(0.1..=1000.0).contains(&r)) {
                    return Err(AiError::ValidationError {
                        command: id.into(),
                        message: "gaussianBlur requires 'radius' in 0.1..=1000".into(),
                    });
                }
            }
            _ => {}
        }

        Ok(ValidatedCommand {
            id: id.to_string(),
            params: params.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reject_ignored_typography_parameters_and_invalid_size() {
        let catalog = CommandCatalog::new();
        for command in ["type.create", "type.setStyle"] {
            for bad in [json!({"x": 0, "y": 0, "text": "Hi", "sizePt": 80}),
                        json!({"x": 0, "y": 0, "text": "Hi", "fontSzie": 80}),
                        json!({"x": 0, "y": 0, "text": "Hi", "size": "80"}),
                        json!({"x": 0, "y": 0, "text": "Hi", "size": 0}),
                        json!({"x": 0, "y": 0, "text": "Hi", "font": null})] {
                assert!(catalog.validate_command(command, &bad).is_err(), "{command} accepted {bad}");
            }
            assert!(catalog.validate_command(command, &json!({"x": 0, "y": 0, "text": "Hi", "size": 80, "font": "Inter", "align": "center"})).is_ok());
        }
    }

    #[test]
    fn validate_valid_type_create() {
        let catalog = CommandCatalog::new();
        let params = json!({
            "x": 100,
            "y": 200,
            "text": "Header Text",
            "size": 24.0,
            "color": "#ffffff"
        });

        let res = catalog.validate_command("type.create", &params);
        assert!(res.is_ok());
    }

    #[test]
    fn reject_missing_text_in_type_create() {
        let catalog = CommandCatalog::new();
        let params = json!({
            "x": 100,
            "y": 200
        });

        let res = catalog.validate_command("type.create", &params);
        assert!(res.is_err());
    }

    #[test]
    fn reject_unknown_command() {
        let catalog = CommandCatalog::new();
        let res = catalog.validate_command("ai.generateMagicPicture", &json!({}));
        assert!(res.is_err());
    }

    #[test]
    fn validate_solid_color_fill() {
        let catalog = CommandCatalog::new();
        let res = catalog.validate_command("layer.newFillLayer.solidColor", &json!({"color": "#0f172a"}));
        assert!(res.is_ok());

        let invalid = catalog.validate_command("layer.newFillLayer.solidColor", &json!({"color": "not_a_hex"}));
        assert!(invalid.is_err());
    }
}
