//! Subsystem D: Creative Design Intelligence
//!
//! Provides design archetypes (invitations, posters, social banners), typography rules & modular scales,
//! layout math (golden ratio, rule-of-thirds, margins, alignments), and harmonic color generation.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};

/// Recognized creative design archetype kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchetypeKind {
    WeddingInvitation,
    PromotionalPoster,
    ProductAdvertising,
    SocialBanner,
    EditorialCover,
    MinimalistCard,
}

/// A structured design archetype specification with curated typographic and chromatic rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DesignArchetype {
    pub kind: ArchetypeKind,
    pub name: String,
    pub description: String,
    pub suggested_fonts: Vec<String>,
    pub primary_palette: Vec<String>,
    pub recommended_aspect: &'static str,
    pub default_width: u32,
    pub default_height: u32,
}

impl DesignArchetype {
    /// Retrieve archetype by kind.
    pub fn from_kind(kind: ArchetypeKind) -> Self {
        match kind {
            ArchetypeKind::WeddingInvitation => Self {
                kind,
                name: "Wedding Invitation".into(),
                description: "Refined, elegant typography with generous margins and luxury warm tones".into(),
                suggested_fonts: vec!["Cinzel".into(), "Playfair Display".into(), "Inter".into()],
                primary_palette: vec![
                    "#fdfbf7".into(), // Warm cream
                    "#d4af37".into(), // Metallic gold
                    "#1e293b".into(), // Deep slate navy
                    "#94a3b8".into(), // Soft silver-gray
                ],
                recommended_aspect: "Portrait",
                default_width: 1200,
                default_height: 1800,
            },
            ArchetypeKind::PromotionalPoster => Self {
                kind,
                name: "Promotional Poster".into(),
                description: "High-impact headlines, dynamic contrast, and bold vibrant accents".into(),
                suggested_fonts: vec!["Impact".into(), "Montserrat".into(), "Inter".into()],
                primary_palette: vec![
                    "#0f172a".into(), // Dark backdrop
                    "#ff3366".into(), // Electric coral/crimson
                    "#38bdf8".into(), // Bright cyan
                    "#ffffff".into(), // High-contrast white
                ],
                recommended_aspect: "Portrait",
                default_width: 1200,
                default_height: 1600,
            },
            ArchetypeKind::ProductAdvertising => Self {
                kind,
                name: "Product Advertising".into(),
                description: "Clean modern focal presentation, balanced hierarchy, and sleek gradients".into(),
                suggested_fonts: vec!["Inter".into(), "Roboto".into(), "Cinzel".into()],
                primary_palette: vec![
                    "#090d16".into(), // Deep midnight
                    "#3b82f6".into(), // Tech sapphire blue
                    "#f8fafc".into(), // Pure ice white
                    "#64748b".into(), // Neutral metallic
                ],
                recommended_aspect: "Landscape",
                default_width: 1920,
                default_height: 1080,
            },
            ArchetypeKind::SocialBanner => Self {
                kind,
                name: "Social Banner".into(),
                description: "Wide horizontal layout optimized for high legibility across screens".into(),
                suggested_fonts: vec!["Inter".into(), "Montserrat".into()],
                primary_palette: vec![
                    "#1e1e24".into(),
                    "#fb7185".into(),
                    "#38bdf8".into(),
                    "#ffffff".into(),
                ],
                recommended_aspect: "WideBanner",
                default_width: 1920,
                default_height: 1080,
            },
            ArchetypeKind::EditorialCover => Self {
                kind,
                name: "Editorial Cover".into(),
                description: "Classic magazine layout with dominant imagery and structured text columns".into(),
                suggested_fonts: vec!["Playfair Display".into(), "Inter".into()],
                primary_palette: vec![
                    "#18181b".into(),
                    "#f43f5e".into(),
                    "#fafaf9".into(),
                ],
                recommended_aspect: "Portrait",
                default_width: 1200,
                default_height: 1600,
            },
            ArchetypeKind::MinimalistCard => Self {
                kind,
                name: "Minimalist Card".into(),
                description: "Restrained elegance, ample whitespace, and singular focal point".into(),
                suggested_fonts: vec!["Inter".into(), "Cinzel".into()],
                primary_palette: vec![
                    "#fafafa".into(),
                    "#171717".into(),
                    "#525252".into(),
                ],
                recommended_aspect: "Landscape",
                default_width: 1400,
                default_height: 900,
            },
        }
    }

    /// Match a string identifier to an archetype.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_lowercase();
        if n.contains("wedding") || n.contains("invitation") {
            Some(Self::from_kind(ArchetypeKind::WeddingInvitation))
        } else if n.contains("poster") {
            Some(Self::from_kind(ArchetypeKind::PromotionalPoster))
        } else if n.contains("advertising") || n.contains("product") {
            Some(Self::from_kind(ArchetypeKind::ProductAdvertising))
        } else if n.contains("banner") || n.contains("social") {
            Some(Self::from_kind(ArchetypeKind::SocialBanner))
        } else if n.contains("editorial") || n.contains("magazine") {
            Some(Self::from_kind(ArchetypeKind::EditorialCover))
        } else if n.contains("minimal") || n.contains("card") {
            Some(Self::from_kind(ArchetypeKind::MinimalistCard))
        } else {
            None
        }
    }
}

/// Typographic modular scale hierarchy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypographyHierarchy {
    pub heading_pt: f32,
    pub subheading_pt: f32,
    pub body_pt: f32,
    pub caption_pt: f32,
    pub line_height_ratio: f32,
    pub heading_font: String,
    pub body_font: String,
}

impl TypographyHierarchy {
    /// Compute typographic hierarchy scale based on canvas height and design archetype.
    pub fn compute(canvas_height: u32, archetype: Option<ArchetypeKind>) -> Self {
        let base_pt = ((canvas_height as f32) / 60.0).clamp(10.0, 24.0);
        let ratio = 1.333; // Perfect Fourth scale

        let (heading_font, body_font) = match archetype {
            Some(ArchetypeKind::WeddingInvitation) => ("Cinzel".into(), "Inter".into()),
            Some(ArchetypeKind::PromotionalPoster) => ("Montserrat".into(), "Inter".into()),
            Some(ArchetypeKind::EditorialCover) => ("Playfair Display".into(), "Inter".into()),
            _ => ("Inter".into(), "Inter".into()),
        };

        let body_pt = base_pt.round();
        let caption_pt = (base_pt / ratio).round().max(9.0);
        let subheading_pt = (base_pt * ratio).round();
        let heading_pt = (subheading_pt * ratio).round().max(24.0);

        Self {
            heading_pt,
            subheading_pt,
            body_pt,
            caption_pt,
            line_height_ratio: 1.4,
            heading_font,
            body_font,
        }
    }
}

/// Color harmony utilities and palette generators.
#[derive(Clone, Debug, Default)]
pub struct ColorHarmony;

impl ColorHarmony {
    /// Parse hex color (`#rrggbb` or `#rgb`) into RGB `[u8; 3]`.
    pub fn parse_hex(hex: &str) -> Option<[u8; 3]> {
        let clean = hex.trim_start_matches('#');
        if clean.len() == 6 {
            let r = u8::from_str_radix(&clean[0..2], 16).ok()?;
            let g = u8::from_str_radix(&clean[2..4], 16).ok()?;
            let b = u8::from_str_radix(&clean[4..6], 16).ok()?;
            Some([r, g, b])
        } else if clean.len() == 3 {
            let r_ch = clean.chars().next()?;
            let g_ch = clean.chars().nth(1)?;
            let b_ch = clean.chars().nth(2)?;
            let r = u8::from_str_radix(&format!("{r_ch}{r_ch}"), 16).ok()?;
            let g = u8::from_str_radix(&format!("{g_ch}{g_ch}"), 16).ok()?;
            let b = u8::from_str_radix(&format!("{b_ch}{b_ch}"), 16).ok()?;
            Some([r, g, b])
        } else {
            None
        }
    }

    /// Format RGB into lowercase hex `#rrggbb`.
    pub fn to_hex(rgb: [u8; 3]) -> String {
        format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
    }

    /// Convert RGB `[0..255]` to HSL `[0.0..360.0, 0.0..1.0, 0.0..1.0]`.
    pub fn rgb_to_hsl(rgb: [u8; 3]) -> [f32; 3] {
        let r = rgb[0] as f32 / 255.0;
        let g = rgb[1] as f32 / 255.0;
        let b = rgb[2] as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let l = (max + min) / 2.0;

        if delta.abs() < 1e-5 {
            return [0.0, 0.0, l];
        }

        let s = if l > 0.5 {
            delta / (2.0 - max - min)
        } else {
            delta / (max + min)
        };

        let mut h = if (max - r).abs() < 1e-5 {
            (g - b) / delta + (if g < b { 6.0 } else { 0.0 })
        } else if (max - g).abs() < 1e-5 {
            (b - r) / delta + 2.0
        } else {
            (r - g) / delta + 4.0
        };
        h *= 60.0;

        [h, s, l]
    }

    /// Convert HSL `[0.0..360.0, 0.0..1.0, 0.0..1.0]` to RGB `[u8; 3]`.
    pub fn hsl_to_rgb(hsl: [f32; 3]) -> [u8; 3] {
        let h = hsl[0].rem_euclid(360.0);
        let s = hsl[1].clamp(0.0, 1.0);
        let l = hsl[2].clamp(0.0, 1.0);

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
        let m = l - c / 2.0;

        let (r1, g1, b1) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        [
            ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ]
    }

    /// Compute complementary color.
    pub fn complementary(hex: &str) -> Option<String> {
        let rgb = Self::parse_hex(hex)?;
        let mut hsl = Self::rgb_to_hsl(rgb);
        hsl[0] = (hsl[0] + 180.0).rem_euclid(360.0);
        Some(Self::to_hex(Self::hsl_to_rgb(hsl)))
    }

    /// Compute triadic color pair.
    pub fn triadic(hex: &str) -> Option<(String, String)> {
        let rgb = Self::parse_hex(hex)?;
        let hsl = Self::rgb_to_hsl(rgb);
        let hsl1 = [(hsl[0] + 120.0).rem_euclid(360.0), hsl[1], hsl[2]];
        let hsl2 = [(hsl[0] + 240.0).rem_euclid(360.0), hsl[1], hsl[2]];
        Some((
            Self::to_hex(Self::hsl_to_rgb(hsl1)),
            Self::to_hex(Self::hsl_to_rgb(hsl2)),
        ))
    }

    /// Calculate linearized relative luminance (WCAG 2.1).
    pub fn relative_luminance(rgb: [u8; 3]) -> f32 {
        let f = |val: u8| -> f32 {
            let s = val as f32 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2])
    }

    /// Calculate WCAG contrast ratio between two hex colors (1.0 to 21.0).
    pub fn contrast_ratio(hex_a: &str, hex_b: &str) -> f32 {
        let rgb_a = Self::parse_hex(hex_a).unwrap_or([255, 255, 255]);
        let rgb_b = Self::parse_hex(hex_b).unwrap_or([0, 0, 0]);

        let lum_a = Self::relative_luminance(rgb_a);
        let lum_b = Self::relative_luminance(rgb_b);

        let lighter = lum_a.max(lum_b);
        let darker = lum_a.min(lum_b);

        (lighter + 0.05) / (darker + 0.05)
    }
}

/// Mathematical spatial layout engine.
#[derive(Clone, Debug, Default)]
pub struct LayoutEngine;

impl LayoutEngine {
    /// Create layout engine.
    pub fn new() -> Self {
        Self
    }

    /// Compute balanced whitespace margins `[top, right, bottom, left]` in pixels.
    pub fn compute_margins(&self, width: u32, height: u32, percentage: f32) -> [i32; 4] {
        let p = percentage.clamp(0.02, 0.4);
        let mx = ((width as f32) * p).round() as i32;
        let my = ((height as f32) * p).round() as i32;
        [my, mx, my, mx]
    }

    /// Compute Rule-of-Thirds intersection coordinates `(x, y)` in canvas space.
    pub fn rule_of_thirds(&self, width: u32, height: u32) -> Vec<[i32; 2]> {
        let x1 = ((width as f32) / 3.0).round() as i32;
        let x2 = (((width as f32) * 2.0) / 3.0).round() as i32;
        let y1 = ((height as f32) / 3.0).round() as i32;
        let y2 = (((height as f32) * 2.0) / 3.0).round() as i32;

        vec![[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
    }

    /// Center a rectangle inside canvas bounds: returns `[x, y, width, height]`.
    pub fn centered_box(&self, box_w: i32, box_h: i32, canvas_w: u32, canvas_h: u32) -> [i32; 4] {
        let x = ((canvas_w as i32) - box_w) / 2;
        let y = ((canvas_h as i32) - box_h) / 2;
        [x, y, box_w, box_h]
    }

    /// Compute horizontal centering coordinate for an item of given width.
    pub fn center_x(&self, item_width: i32, canvas_width: u32) -> i32 {
        ((canvas_width as i32) - item_width) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archetype_lookup_and_palette() {
        let wedding = DesignArchetype::from_name("elegant wedding invitation").unwrap();
        assert_eq!(wedding.kind, ArchetypeKind::WeddingInvitation);
        assert!(!wedding.primary_palette.is_empty());
        assert_eq!(wedding.recommended_aspect, "Portrait");
    }

    #[test]
    fn color_harmony_parsing_and_complement() {
        let hex = "#3b82f6";
        let comp = ColorHarmony::complementary(hex).unwrap();
        assert!(comp.starts_with('#'));
        assert_ne!(comp, hex);
    }

    #[test]
    fn color_contrast_wcag() {
        let ratio = ColorHarmony::contrast_ratio("#ffffff", "#000000");
        assert!((ratio - 21.0).abs() < 0.2);

        let low = ColorHarmony::contrast_ratio("#ffffff", "#fefefe");
        assert!(low < 1.5);
    }

    #[test]
    fn layout_margins_and_thirds() {
        let layout = LayoutEngine::new();
        let margins = layout.compute_margins(1000, 1000, 0.1);
        assert_eq!(margins, [100, 100, 100, 100]);

        let thirds = layout.rule_of_thirds(900, 600);
        assert_eq!(thirds.len(), 4);
        assert_eq!(thirds[0], [300, 200]);
        assert_eq!(thirds[3], [600, 400]);
    }
}
