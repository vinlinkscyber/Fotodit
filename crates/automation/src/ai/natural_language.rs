//! Subsystem A: Natural Language Intelligence
//!
//! Parses natural language instructions, identifies intent categories, extracts entities and
//! parameters, resolves document targets and context, and maintains conversational continuity.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::perception::VisualElement;
use super::perception::VisualScene;
use super::{AiContext, AiError};

/// High-level editing intent category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentCategory {
    /// Comprehensive photo enhancement (tonal curves, vibrance, sharpening, color grade).
    PhotoEnhancement,
    /// Isolate foreground subject / remove background non-destructively.
    SubjectIsolation,
    /// Replace background color or gradient while preserving subject.
    BackgroundReplacement,
    /// Graphic design generation (wedding invitation, promotional poster, advertising banner).
    GraphicDesign,
    /// Typography edits, positioning, size changes, font changes, alignment.
    TypographyRefinement,
    /// Global color scheme harmonization, color lookup, duotone or black & white.
    ColorStyling,
    /// Spatial repositioning, alignment, canvas dimensions, cropping.
    LayoutAdjustment,
    /// Explicit filter or adjustment request (blur, noise, exposure).
    DirectFilterOrAdjustment,
    /// Custom multi-step or general editing intent.
    GeneralEditing,
}

/// Parsed natural language intent with structured entities and parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditingIntent {
    /// Original instruction verbatim.
    pub raw_prompt: String,
    /// Identified primary category.
    pub category: IntentCategory,
    /// Specific design or visual archetype if requested.
    pub archetype: Option<String>,
    /// Targeted layer name or identifier if resolved.
    pub target_layer: Option<String>,
    /// Target layer ID if identified from document scene.
    pub target_layer_id: Option<u64>,
    /// Extracted color (hex format, e.g. `#1e293b`).
    pub extracted_color: Option<String>,
    /// Extracted numeric value (e.g. font size, adjustment amount, percentage).
    pub numeric_value: Option<f64>,
    /// Directional vector for movement/alignment (`[dx, dy]`).
    pub direction: Option<[i32; 2]>,
    /// Desired stylistic tone (e.g. "professional", "elegant", "premium", "minimalist").
    pub stylistic_tone: Option<String>,
    /// Extracted text content to set or add.
    pub text_content: Option<String>,
}

/// Natural Language Intelligence engine.
#[derive(Clone, Debug, Default)]
pub struct NaturalLanguageEngine;

impl NaturalLanguageEngine {
    /// Create new natural language engine.
    pub fn new() -> Self {
        Self
    }

    /// Parse an ordinary English prompt into a structured `EditingIntent`.
    pub fn parse_intent(
        &self,
        prompt: &str,
        scene: &VisualScene,
        context: &AiContext,
    ) -> Result<EditingIntent, AiError> {
        let p_lower = prompt.trim().to_lowercase();
        if p_lower.is_empty() {
            return Err(AiError::IntentError("empty editing prompt".into()));
        }

        let tone = self.extract_tone(&p_lower);
        let extracted_color = self.extract_color(&p_lower);
        let numeric_value = self.extract_numeric(&p_lower);
        let direction = self.extract_direction(&p_lower);
        let archetype = self.extract_archetype(&p_lower);

        // Classify intent category based on keyword semantics and document context
        let category = if p_lower.contains("remove") && (p_lower.contains("background") || p_lower.contains("bg")) {
            IntentCategory::SubjectIsolation
        } else if (p_lower.contains("change") || p_lower.contains("replace") || p_lower.contains("set"))
            && (p_lower.contains("background") || p_lower.contains("bg"))
        {
            IntentCategory::BackgroundReplacement
        } else if p_lower.contains("invitation") || p_lower.contains("poster") || p_lower.contains("banner") || p_lower.contains("advertisement") || p_lower.contains("advertising") {
            IntentCategory::GraphicDesign
        } else if p_lower.contains("typography")
            || p_lower.contains("heading")
            || p_lower.contains("headline")
            || p_lower.contains("font")
            || p_lower.contains("text")
            || p_lower.contains("title")
        {
            IntentCategory::TypographyRefinement
        } else if p_lower.contains("professional")
            || p_lower.contains("retouch")
            || p_lower.contains("enhance")
            || p_lower.contains("lighting")
            || p_lower.contains("portrait")
            || p_lower.contains("clarity")
        {
            IntentCategory::PhotoEnhancement
        } else if p_lower.contains("color scheme")
            || p_lower.contains("palette")
            || p_lower.contains("black and white")
            || p_lower.contains("black & white")
            || p_lower.contains("warm tone")
            || p_lower.contains("cool tone")
        {
            IntentCategory::ColorStyling
        } else if p_lower.contains("align") || p_lower.contains("center") || p_lower.contains("move") || p_lower.contains("distribute") {
            IntentCategory::LayoutAdjustment
        } else if p_lower.contains("blur") || p_lower.contains("sharpen") || p_lower.contains("brightness") || p_lower.contains("contrast") {
            IntentCategory::DirectFilterOrAdjustment
        } else {
            IntentCategory::GeneralEditing
        };

        // Resolve target layer using scene analysis and context
        let (target_layer, target_layer_id) = self.resolve_target_layer(&p_lower, scene, context);

        // Extract any explicit text string if user is specifying text: quoted
        // spans first, then names introduced by "called"/"named" (preserved as
        // typed - the user's own words are the design's required content).
        // Named-entity fallback is scoped to design intent so prompts like
        // "retouch the photo called sunset" do not gain spurious text content.
        let text_content = self.extract_quoted_text(prompt).or_else(|| {
            if matches!(category, IntentCategory::GraphicDesign | IntentCategory::TypographyRefinement) {
                extract_named_entity(prompt)
            } else {
                None
            }
        });

        Ok(EditingIntent {
            raw_prompt: prompt.to_string(),
            category,
            archetype,
            target_layer,
            target_layer_id,
            extracted_color,
            numeric_value,
            direction,
            stylistic_tone: tone,
            text_content,
        })
    }

    fn extract_tone(&self, text: &str) -> Option<String> {
        let tones = ["professional", "elegant", "premium", "luxury", "minimalist", "modern", "vintage", "bold", "dramatic", "clean"];
        for tone in tones {
            if text.contains(tone) {
                return Some(tone.to_string());
            }
        }
        None
    }

    fn extract_archetype(&self, text: &str) -> Option<String> {
        if text.contains("wedding invitation") || text.contains("invitation") {
            Some("wedding_invitation".into())
        } else if text.contains("poster") {
            Some("promotional_poster".into())
        } else if text.contains("advertising") || text.contains("advertisement") || text.contains("product") {
            Some("product_advertising".into())
        } else if text.contains("social banner") || text.contains("banner") {
            Some("social_banner".into())
        } else {
            None
        }
    }

    fn extract_color(&self, text: &str) -> Option<String> {
        // Hex colors
        if let Some(pos) = text.find('#') {
            let hex_slice = &text[pos..];
            let end = hex_slice.find(|c: char| !c.is_ascii_hexdigit() && c != '#').unwrap_or(hex_slice.len());
            let hex_str = &hex_slice[..end];
            if hex_str.len() == 7 || hex_str.len() == 4 {
                return Some(hex_str.to_string());
            }
        }

        // Common named colors mapped to hex
        let named_colors: &[(&str, &str)] = &[
            ("navy blue", "#0f172a"),
            ("navy", "#1e293b"),
            ("royal blue", "#1d4ed8"),
            ("sky blue", "#38bdf8"),
            ("blue", "#2563eb"),
            ("gold", "#d97706"),
            ("golden", "#f59e0b"),
            ("sage green", "#84a98c"),
            ("sage", "#94a3b8"),
            ("emerald", "#059669"),
            ("green", "#16a34a"),
            ("blush pink", "#fbcfe8"),
            ("rose", "#f43f5e"),
            ("pink", "#ec4899"),
            ("cream", "#fef3c7"),
            ("champagne", "#fde68a"),
            ("white", "#ffffff"),
            ("pure white", "#ffffff"),
            ("black", "#000000"),
            ("dark slate", "#0f172a"),
            ("gray", "#64748b"),
            ("charcoal", "#334155"),
            ("coral", "#fb7185"),
            ("teal", "#0d9488"),
            ("red", "#dc2626"),
            ("orange", "#ea580c"),
            ("yellow", "#eab308"),
            ("purple", "#9333ea"),
            ("violet", "#8b5cf6"),
            ("brown", "#92400e"),
        ];

        for (name, hex) in named_colors {
            if text.contains(name) {
                return Some((*hex).to_string());
            }
        }

        None
    }

    /// Every distinct colour the prompt names, longest names first and
    /// deduplicated by hex ("navy blue" counts once, not also as "blue").
    /// More than one means a colour criterion cannot tell WHICH colour binds
    /// to WHICH object ("keep the background red, make the shirt blue") and
    /// must stay unverified rather than checking the wrong one.
    pub(crate) fn colour_terms(&self, text: &str) -> Vec<String> {
        // Same table as extract_color, longest names first.
        let named_colors: &[(&str, &str)] = &[
            ("navy blue", "#0f172a"), ("royal blue", "#1d4ed8"), ("sky blue", "#38bdf8"),
            ("sage green", "#84a98c"), ("blush pink", "#fbcfe8"), ("pure white", "#ffffff"),
            ("dark slate", "#0f172a"), ("champagne", "#fde68a"), ("charcoal", "#334155"),
            ("emerald", "#059669"), ("golden", "#f59e0b"), ("navy", "#1e293b"),
            ("blue", "#2563eb"), ("gold", "#d97706"), ("sage", "#94a3b8"),
            ("green", "#16a34a"), ("rose", "#f43f5e"), ("pink", "#ec4899"),
            ("cream", "#fef3c7"), ("white", "#ffffff"), ("black", "#000000"),
            ("gray", "#64748b"), ("coral", "#fb7185"), ("teal", "#0d9488"),
            ("red", "#dc2626"), ("orange", "#ea580c"), ("yellow", "#eab308"),
            ("purple", "#9333ea"), ("violet", "#8b5cf6"), ("brown", "#92400e"),
        ];
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut found: Vec<String> = Vec::new();
        for (name, hex) in named_colors {
            let mut at = 0;
            while let Some(pos) = text[at..].find(name) {
                let (s, e) = (at + pos, at + pos + name.len());
                at = e;
                if spans.iter().any(|(cs, ce)| s >= *cs && e <= *ce) { continue; }
                spans.push((s, e));
                if !found.iter().any(|h| h == hex) { found.push((*hex).to_string()); }
            }
        }
        found
    }

    fn extract_numeric(&self, text: &str) -> Option<f64> {
        let words = text.split_whitespace();
        for word in words {
            let clean = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-');
            if let Ok(val) = clean.parse::<f64>() {
                return Some(val);
            }
        }
        None
    }

    fn extract_direction(&self, text: &str) -> Option<[i32; 2]> {
        if text.contains("upwards") || text.contains("up") || text.contains("higher") {
            Some([0, -40])
        } else if text.contains("downwards") || text.contains("down") || text.contains("lower") {
            Some([0, 40])
        } else if text.contains("left") {
            Some([-40, 0])
        } else if text.contains("right") {
            Some([40, 0])
        } else if text.contains("center") || text.contains("centered") {
            Some([0, 0])
        } else {
            None
        }
    }

    fn extract_quoted_text(&self, text: &str) -> Option<String> {
        extract_quoted_span(text)
    }

    /// User-supplied subject terms that a design must carry: quoted spans and
    /// names introduced by "called"/"named", preserved exactly as typed (case,
    /// spaces, punctuation). Empty when the prompt names nothing - callers must
    /// not invent substitute content.
    pub(crate) fn subject_terms(&self, prompt: &str) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        if let Some(quoted) = extract_quoted_span(prompt) {
            terms.push(quoted);
        }
        if let Some(named) = extract_named_entity(prompt) {
            if !terms.iter().any(|t| t.eq_ignore_ascii_case(&named)) {
                terms.push(named);
            }
        }
        terms
    }

    fn resolve_target_layer(
        &self,
        prompt: &str,
        scene: &VisualScene,
        context: &AiContext,
    ) -> (Option<String>, Option<u64>) {
        // 1. Heading / Title reference
        if prompt.contains("heading") || prompt.contains("headline") || prompt.contains("title") {
            if let Some(top_text) = scene.find_primary_heading() {
                return (Some(top_text.name.clone()), Some(top_text.id));
            }
        }

        // 2. Background reference
        if prompt.contains("background") || prompt.contains("bg") {
            if let Some(bg) = scene.find_background_layer() {
                return (Some(bg.name.clone()), Some(bg.id));
            }
        }

        // 3. Match by explicit layer name
        for layer in &scene.layers {
            if prompt.contains(&layer.name.to_lowercase()) {
                return (Some(layer.name.clone()), Some(layer.id));
            }
        }

        // 4. Anaphoric / Contextual ("it", "the text", "this")
        if (prompt.contains(" it ") || prompt.contains("its ") || prompt.contains("this"))
            && let Some(active) = &context.active_target
        {
            if let Some(layer) = scene.layers.iter().find(|l| l.name == *active) {
                return (Some(layer.name.clone()), Some(layer.id));
            }
        }

        // 5. Fallback to active layer in scene
        if let Some(active_id) = scene.active_layer_id {
            if let Some(layer) = scene.layers.iter().find(|l| l.id == active_id) {
                return (Some(layer.name.clone()), Some(layer.id));
            }
        }

        (None, None)
    }
}

/// Case-insensitive search for an ASCII marker that returns the byte offset
/// in the ORIGINAL string. Byte-window matching is boundary-safe: ASCII marker
/// bytes can never match UTF-8 continuation bytes (>= 0x80), so every match
/// starts and ends on a char boundary. Lowercasing first would be wrong here:
/// it can expand a character (`'İ'` -> `"i\u{307}"`), shifting offsets so a
/// slice into the original string lands mid-character and panics.
fn find_ascii_ci(haystack: &str, marker: &str) -> Option<usize> {
    debug_assert!(marker.is_ascii() && !marker.is_empty());
    haystack
        .as_bytes()
        .windows(marker.len())
        .position(|w| w.eq_ignore_ascii_case(marker.as_bytes()))
}

/// True when `pos` starts a fresh word (prompt start or preceded by
/// whitespace), so "renamed" or "so-called" do not trigger the
/// "named"/"called" markers.
fn at_word_boundary(text: &str, pos: usize) -> bool {
    pos == 0 || text[..pos].chars().next_back().is_some_and(char::is_whitespace)
}

/// Cut a trailing purpose clause ("chndrax to promote it" -> "chndrax") only
/// when the word after the separator is a purpose verb. Names that
/// legitimately contain "to"/"for"/"with" ("A to Z", "Fit For Life") are
/// kept whole because what follows is not a purpose verb.
fn cut_purpose_clause(entity: &str) -> &str {
    const PURPOSE_VERBS: [&str; 12] = [
        "promote", "promoting", "advertise", "advertising", "sell", "selling",
        "showcase", "announce", "announcing", "market", "marketing", "attract",
    ];
    for marker in [" to ", " for ", " with ", " that "] {
        if let Some(pos) = find_ascii_ci(entity, marker) {
            let next_word = entity[pos + marker.len()..].split_whitespace().next().unwrap_or("");
            if PURPOSE_VERBS.iter().any(|v| next_word.eq_ignore_ascii_case(v)) {
                return entity[..pos].trim_end();
            }
        }
    }
    entity
}

/// First quoted span in the prompt, if any.
pub(crate) fn extract_quoted_span(text: &str) -> Option<String> {
    extract_delimited(text, '"').or_else(|| extract_delimited(text, '\''))
}

/// Span between a matched delimiter pair. A single quote only opens a span at
/// a word boundary, so possessives ("chndrax's") are not mistaken for quoted
/// text.
fn extract_delimited(text: &str, quote: char) -> Option<String> {
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c != quote {
            continue;
        }
        match start {
            None => {
                if quote == '\'' && !at_word_boundary(text, i) {
                    continue;
                }
                start = Some(i);
            }
            Some(s) => {
                let inner = text[s + quote.len_utf8()..i].trim();
                if !inner.is_empty() {
                    return Some(inner.to_string());
                }
                start = None;
            }
        }
    }
    None
}

/// Name the user gave their subject, introduced by "called X", "named X" or
/// "name is X" at a word boundary, cut at the end of that clause and preserved
/// exactly as typed.
pub(crate) fn extract_named_entity(prompt: &str) -> Option<String> {
    for marker in ["called ", "named ", "name is "] {
        let mut from = 0;
        while let Some(rel) = find_ascii_ci(&prompt[from..], marker) {
            let pos = from + rel;
            if at_word_boundary(prompt, pos) {
                let rest = &prompt[pos + marker.len()..];
                let end = rest
                    .find(|c: char| matches!(c, ',' | '.' | '!' | '?' | ';' | '\n'))
                    .unwrap_or(rest.len());
                let entity = rest[..end]
                    .trim()
                    .trim_matches(|c: char| c == '"' || c == '\'')
                    .trim();
                let entity = cut_purpose_clause(entity);
                if !entity.is_empty() {
                    return Some(entity.to_string());
                }
            }
            from = pos + 1;
        }
    }
    None
}

/// Purpose phrase the user stated for the design ("for my taxi business" ->
/// "taxi business"), preserved as typed. None when no "for ..." clause exists.
pub(crate) fn subject_purpose(prompt: &str) -> Option<String> {
    let mut from = 0;
    let start = loop {
        let rel = find_ascii_ci(&prompt[from..], "for ")?;
        let pos = from + rel;
        if at_word_boundary(prompt, pos) {
            break pos + 4;
        }
        from = pos + 1;
    };
    let rest = prompt[start..].trim_start();
    let rest = rest
        .strip_prefix("my ")
        .or_else(|| rest.strip_prefix("a "))
        .or_else(|| rest.strip_prefix("an "))
        .or_else(|| rest.strip_prefix("the "))
        .unwrap_or(rest);
    let end = rest
        .find(|c: char| matches!(c, ',' | '.' | '!' | '?' | ';' | '\n'))
        .unwrap_or(rest.len());
    let phrase = rest[..end].trim();
    // A following "called"/"named" clause names the subject; it is not part
    // of the purpose phrase.
    let phrase = [" called ", " named "]
        .iter()
        .fold(phrase, |p, m| match find_ascii_ci(p, m) {
            Some(i) => p[..i].trim_end(),
            None => p,
        });
    let phrase = cut_purpose_clause(phrase);
    if phrase.is_empty() {
        None
    } else {
        Some(phrase.to_string())
    }
}

/// Display-copy cleanup for purpose phrases the planner places on a design.
/// The user's named subject (quoted or "called"/"named") is preserved
/// verbatim elsewhere; a purpose phrase is read for intent, not copied
/// blindly - whitespace is collapsed, the first letter is capitalized, and a
/// short list of common English misspellings is corrected word-by-word.
/// Nothing is invented: unrecognized words pass through unchanged.
pub(crate) fn normalize_display_copy(text: &str) -> String {
    const COMMON_MISSPELLINGS: &[(&str, &str)] = &[
        ("busines", "business"),
        ("recieve", "receive"),
        ("seperate", "separate"),
        ("adress", "address"),
        ("occassion", "occasion"),
        ("definately", "definitely"),
        ("restaraunt", "restaurant"),
        ("teh", "the"),
        ("untill", "until"),
        ("wich", "which"),
    ];
    let mut out = String::new();
    for (index, word) in text.split_whitespace().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        // Correct the alphabetic core so trailing punctuation ("busines.")
        // still matches.
        let core_end = word.find(|c: char| !c.is_alphabetic()).unwrap_or(word.len());
        let (core, rest) = word.split_at(core_end);
        let corrected = COMMON_MISSPELLINGS
            .iter()
            .find(|(wrong, _)| core.eq_ignore_ascii_case(wrong))
            .map(|(_, right)| {
                if core.chars().any(char::is_alphabetic) && !core.chars().any(char::is_lowercase) {
                    right.to_uppercase()
                } else {
                    right.to_string()
                }
            })
            .unwrap_or_else(|| core.to_string());
        if index == 0 {
            let mut chars = corrected.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push_str(&corrected);
        }
        out.push_str(rest);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_remove_background_intent() {
        let engine = NaturalLanguageEngine::new();
        let scene = VisualScene::default();
        let context = AiContext::new();

        let intent = engine
            .parse_intent("Remove the background without damaging the subject", &scene, &context)
            .unwrap();
        assert_eq!(intent.category, IntentCategory::SubjectIsolation);
    }

    #[test]
    fn parse_professional_enhancement_intent() {
        let engine = NaturalLanguageEngine::new();
        let scene = VisualScene::default();
        let context = AiContext::new();

        let intent = engine.parse_intent("Make this photograph look professional", &scene, &context).unwrap();
        assert_eq!(intent.category, IntentCategory::PhotoEnhancement);
        assert_eq!(intent.stylistic_tone, Some("professional".into()));
    }

    #[test]
    fn parse_wedding_invitation_intent() {
        let engine = NaturalLanguageEngine::new();
        let scene = VisualScene::default();
        let context = AiContext::new();

        let intent = engine.parse_intent("Create an elegant wedding invitation", &scene, &context).unwrap();
        assert_eq!(intent.category, IntentCategory::GraphicDesign);
        assert_eq!(intent.archetype, Some("wedding_invitation".into()));
        assert_eq!(intent.stylistic_tone, Some("elegant".into()));
    }

    #[test]
    fn parse_heading_move_and_resize() {
        let engine = NaturalLanguageEngine::new();
        let mut scene = VisualScene::default();
        scene.layers.push(VisualElement {
            id: 10,
            name: "Main Heading".into(),
            kind: "Type".into(),
            text_content: Some("Summer Sale".into()),
            font_size_pt: Some(24.0),
            bounds: [50, 100, 300, 40],
            is_visible: true,
            has_mask: false,
            confidence: 1.0,
            ..Default::default()
        });

        let context = AiContext::new();
        let intent = engine
            .parse_intent("Move the heading upwards and increase its size to 36", &scene, &context)
            .unwrap();

        assert_eq!(intent.category, IntentCategory::TypographyRefinement);
        assert_eq!(intent.target_layer_id, Some(10));
        assert_eq!(intent.direction, Some([0, -40]));
        assert_eq!(intent.numeric_value, Some(36.0));
    }

    #[test]
    fn parse_poster_prompt_preserves_business_name() {
        let engine = NaturalLanguageEngine::new();
        let scene = VisualScene::default();
        let context = AiContext::new();

        let intent = engine
            .parse_intent("create a poster for my taxi busines to promote it its called chndrax", &scene, &context)
            .unwrap();

        assert_eq!(intent.category, IntentCategory::GraphicDesign);
        assert_eq!(intent.text_content.as_deref(), Some("chndrax"));
    }

    #[test]
    fn subject_terms_prefers_quoted_then_named_entity() {
        let engine = NaturalLanguageEngine::new();
        assert_eq!(
            engine.subject_terms("make a poster called chndrax"),
            vec!["chndrax".to_string()]
        );
        assert_eq!(
            engine.subject_terms("make a poster \"Grand Opening\" called chndrax"),
            vec!["Grand Opening".to_string(), "chndrax".to_string()]
        );
        assert!(engine.subject_terms("make it look professional").is_empty());
    }

    #[test]
    fn named_entity_respects_marker_word_boundaries() {
        assert_eq!(extract_named_entity("the layer renamed Sunset Hills"), None);
        assert_eq!(extract_named_entity("a so-called premium look"), None);
        assert_eq!(extract_named_entity("make a poster called chndrax"), Some("chndrax".to_string()));
        assert_eq!(extract_named_entity("it is named Corner Cafe"), Some("Corner Cafe".to_string()));
    }

    #[test]
    fn named_entity_keeps_names_with_clause_words() {
        // "for"/"to" inside a name are not purpose clauses.
        assert_eq!(extract_named_entity("a poster called Fit For Life"), Some("Fit For Life".to_string()));
        assert_eq!(extract_named_entity("a poster called A to Z"), Some("A to Z".to_string()));
        assert_eq!(extract_named_entity("called chndrax to promote it"), Some("chndrax".to_string()));
    }

    #[test]
    fn named_entity_preserves_multibyte_text_without_panic() {
        // Lowercasing-based offset lookup would mis-slice or panic on these;
        // boundary-safe matching must preserve the original bytes exactly.
        assert_eq!(extract_named_entity("make a poster called İstanbul Eats"), Some("İstanbul Eats".to_string()));
        assert_eq!(extract_named_entity("poster for café Üniqe called Zürich Bäckerei"), Some("Zürich Bäckerei".to_string()));
    }

    #[test]
    fn quoted_span_ignores_possessive_apostrophes() {
        assert_eq!(extract_quoted_span("fix chndrax's colors"), None);
        assert_eq!(extract_quoted_span("a poster titled 'Grand Opening' tonight"), Some("Grand Opening".to_string()));
        assert_eq!(extract_quoted_span("a poster titled \"Grand Opening\""), Some("Grand Opening".to_string()));
    }

    #[test]
    fn subject_purpose_extracts_user_phrase() {
        assert_eq!(
            subject_purpose("create a poster for my taxi busines to promote it its called chndrax"),
            Some("taxi busines".to_string())
        );
        assert_eq!(subject_purpose("make a banner for the bakery"), Some("bakery".to_string()));
        assert_eq!(subject_purpose("enhance this photo"), None);
    }

    #[test]
    fn normalize_display_copy_interprets_purpose_without_inventing() {
        assert_eq!(normalize_display_copy("taxi busines"), "Taxi business");
        assert_eq!(normalize_display_copy("  grand   opening "), "Grand opening");
        assert_eq!(normalize_display_copy("BAKERY"), "BAKERY");
        assert_eq!(normalize_display_copy("busines."), "Business.");
        // Unknown words pass through unchanged; nothing is invented.
        assert_eq!(normalize_display_copy("zxqwv quux"), "Zxqwv quux");
        assert_eq!(normalize_display_copy(""), "");
    }

    #[test]
    fn parse_change_background_color_preserving_person() {
        let engine = NaturalLanguageEngine::new();
        let scene = VisualScene::default();
        let context = AiContext::new();

        let intent = engine
            .parse_intent("Change the background colour to navy blue but preserve the person", &scene, &context)
            .unwrap();

        assert_eq!(intent.category, IntentCategory::BackgroundReplacement);
        assert_eq!(intent.extracted_color, Some("#0f172a".into()));
    }
}
