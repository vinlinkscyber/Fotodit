//! Validate new filter invocations, without changing how old smart-filter records decode.

use serde_json::Value;

use crate::{EngineError, Result};

/// One parameter rule of a core filter's public dialog-unit contract. Public so
/// AI/tooling surfaces can mirror the engine's own validation instead of
/// maintaining a second hand-written copy that drifts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Number(f64, f64),
    Unsigned(u64),
    Choice(&'static str),
    Bool,
}

impl Rule {
    /// Whether one JSON value satisfies this rule.
    pub fn accepts(self, value: &Value) -> bool {
        match self {
            Self::Number(min, max) => value.as_f64().is_some_and(|n| n.is_finite() && (min..=max).contains(&n)),
            Self::Unsigned(max) => value.as_u64().is_some_and(|n| n <= max),
            Self::Choice(choices) => value.as_str().is_some_and(|s| choices.split('|').any(|choice| choice == s)),
            Self::Bool => value.is_boolean(),
        }
    }

    fn expected(self) -> String {
        match self {
            Self::Number(min, max) if min == f64::MIN && max == f64::MAX => "a finite number".into(),
            Self::Number(min, max) => format!("a number in {min}..{max}"),
            Self::Unsigned(max) => format!("an integer in 0..{max}"),
            Self::Choice(choices) => format!("one of {choices}"),
            Self::Bool => "a boolean".into(),
        }
    }
}

// The bounds and choices are the filters' public dialog-unit contract. Defaults remain in
// params_for; gallery filters are checked in gallery_cmds. This table is the single source
// of truth: the engine validates from it and AI/tooling surfaces mirror it through
// `fields`/`FILTER_FIELDS` instead of keeping a second copy.
pub static FILTER_FIELDS: &[(&str, &[(&str, Rule)])] = &[
    ("filter.blur.gaussianBlur", &[("radius", Rule::Number(0.1, 1000.0))]),
    ("filter.other.highPass", &[("radius", Rule::Number(0.1, 1000.0))]),
    ("filter.blur.boxBlur", &[("radius", Rule::Number(1.0, 2000.0))]),
    ("filter.blur.motionBlur", &[("angle", Rule::Number(-360.0, 360.0)), ("distance", Rule::Number(1.0, 2000.0))]),
    ("filter.blur.radialBlur", &[
        ("amount", Rule::Number(1.0, 100.0)),
        ("method", Rule::Choice("spin|zoom")),
        ("quality", Rule::Choice("draft|good|best")),
        ("centerX", Rule::Number(0.0, 1.0)),
        ("centerY", Rule::Number(0.0, 1.0)),
    ]),
    ("filter.blur.smartBlur", &[
        ("radius", Rule::Number(0.1, 100.0)),
        ("threshold", Rule::Number(0.1, 100.0)),
        ("quality", Rule::Choice("high|medium|low")),
        ("mode", Rule::Choice("normal|edgeOnly|overlayEdge")),
    ]),
    ("filter.blur.shapeBlur", &[("radius", Rule::Number(5.0, 1000.0)), ("shape", Rule::Choice("circle|ring|square|diamond|triangle|hexagon|star|heart|cross"))]),
    ("filter.noise.reduceNoise", &[
        ("strength", Rule::Number(0.0, 10.0)),
        ("preserveDetails", Rule::Number(0.0, 100.0)),
        ("reduceColorNoise", Rule::Number(0.0, 100.0)),
        ("sharpenDetails", Rule::Number(0.0, 100.0)),
        ("removeJpegArtifact", Rule::Bool),
    ]),
    ("filter.blur.surfaceBlur", &[("radius", Rule::Number(1.0, 100.0)), ("threshold", Rule::Number(2.0, 255.0))]),
    ("filter.sharpen.unsharpMask", &[("amount", Rule::Number(1.0, 500.0)), ("radius", Rule::Number(0.1, 1000.0)), ("threshold", Rule::Number(0.0, 255.0))]),
    ("filter.sharpen.smartSharpen", &[("amount", Rule::Number(1.0, 500.0)), ("radius", Rule::Number(0.1, 64.0)), ("reduceNoise", Rule::Number(0.0, 100.0))]),
    ("filter.noise.addNoise", &[
        ("amount", Rule::Number(0.1, 400.0)),
        ("distribution", Rule::Choice("uniform|gaussian")),
        ("monochromatic", Rule::Bool),
        ("seed", Rule::Unsigned(u32::MAX as u64)),
    ]),
    ("filter.noise.median", &[("radius", Rule::Number(1.0, 500.0))]),
    ("filter.noise.dustAndScratches", &[("radius", Rule::Number(1.0, 500.0)), ("threshold", Rule::Number(0.0, 255.0))]),
    ("filter.pixelate.mosaic", &[("cellSize", Rule::Number(2.0, 200.0))]),
    ("filter.stylize.emboss", &[("angle", Rule::Number(-180.0, 180.0)), ("height", Rule::Number(1.0, 100.0)), ("amount", Rule::Number(1.0, 500.0))]),
    ("filter.distort.twirl", &[("angle", Rule::Number(-999.0, 999.0))]),
    ("filter.distort.pinch", &[("amount", Rule::Number(-100.0, 100.0))]),
    ("filter.distort.spherize", &[("amount", Rule::Number(-100.0, 100.0)), ("mode", Rule::Choice("normal|horizontalOnly|verticalOnly"))]),
    ("filter.distort.wave", &[
        ("generators", Rule::Number(1.0, 999.0)),
        ("wavelengthMin", Rule::Number(1.0, 998.0)),
        ("wavelengthMax", Rule::Number(2.0, 999.0)),
        ("amplitudeMin", Rule::Number(1.0, 998.0)),
        ("amplitudeMax", Rule::Number(1.0, 999.0)),
        ("type", Rule::Choice("sine|triangle|square")),
        ("undefinedAreas", Rule::Choice("wrap|repeat")),
        ("seed", Rule::Unsigned(u32::MAX as u64)),
    ]),
    ("filter.distort.ripple", &[("amount", Rule::Number(-999.0, 999.0)), ("size", Rule::Choice("small|medium|large"))]),
    ("filter.distort.polarCoordinates", &[("mode", Rule::Choice("rectangularToPolar|polarToRectangular"))]),
    ("filter.other.minimum", &[("radius", Rule::Number(0.2, 500.0)), ("preserve", Rule::Choice("squareness|roundness"))]),
    ("filter.other.maximum", &[("radius", Rule::Number(0.2, 500.0)), ("preserve", Rule::Choice("squareness|roundness"))]),
    ("filter.other.offset", &[
        // The hint specifies pixels, with no range: keep the existing rounding/saturation.
        ("horizontal", Rule::Number(f64::MIN, f64::MAX)),
        ("vertical", Rule::Number(f64::MIN, f64::MAX)),
        ("undefinedAreas", Rule::Choice("wrap|repeat|transparent")),
    ]),
    // Paramless filters: fixed settings, nothing to validate.
    ("filter.blur.blur", &[]),
    ("filter.blur.blurMore", &[]),
    ("filter.sharpen.sharpen", &[]),
    ("filter.sharpen.sharpenMore", &[]),
    ("filter.sharpen.sharpenEdges", &[]),
    ("filter.noise.despeckle", &[]),
    ("filter.stylize.findEdges", &[]),
    ("filter.stylize.solarize", &[]),
];

/// The rule table for one core filter id, when it has one.
pub fn fields(id: &str) -> Option<&'static [(&'static str, Rule)]> {
    FILTER_FIELDS.iter().find(|(fid, _)| *fid == id).map(|(_, rules)| *rules)
}

// Remaining extended filters: only their documented choices are checked for now, since some
// accept extra keys (colours, map documents) and their numeric ranges clamp in params_for.
fn choices(id: &str) -> Option<&'static [(&'static str, &'static str)]> {
    Some(match id {
        "filter.pixelate.mezzotint" => {
            &[("type", "fineDots|mediumDots|grainyDots|coarseDots|shortLines|mediumLines|longLines|shortStrokes|mediumStrokes|longStrokes")]
        }
        "filter.stylize.diffuse" => &[("mode", "normal|darkenOnly|lightenOnly|anisotropic")],
        "filter.stylize.extrude" => &[("type", "blocks|pyramids"), ("depthMode", "random|levelBased")],
        "filter.stylize.tiles" => &[("fill", "background|foreground|inverse|unaltered")],
        "filter.stylize.traceContour" => &[("edge", "lower|upper")],
        "filter.stylize.wind" => &[("method", "wind|blast|stagger"), ("direction", "fromRight|fromLeft")],
        "filter.distort.displace" => &[("fit", "stretch|tile"), ("undefinedAreas", "repeat|wrap")],
        "filter.distort.shear" => &[("undefinedAreas", "wrap|repeat")],
        "filter.distort.zigZag" => &[("style", "pondRipples|outFromCenter|aroundCenter")],
        "filter.render.lensFlare" => &[("lens", "zoom|prime35|prime105|moviePrime")],
        "filter.render.lightingEffects" => &[("lightType", "spot|point|infinite"), ("texture", "none|red|green|blue|alpha|luminance")],
        "filter.blur.lensBlur" => &[
            ("shape", "hexagon|triangle|square|pentagon|heptagon|octagon"),
            ("depthMap", "none|transparency|layerMask|channel"),
            ("distribution", "uniform|gaussian"),
        ],
        "filter.other.hsbHsl" => &[("inputMode", "rgb|hsb|hsl"), ("rowOrder", "hsb|hsl|rgb")],
        "filter.video.deInterlace" => &[("eliminate", "oddFields|evenFields"), ("createBy", "interpolation|duplication")],
        _ => return None,
    })
}

/// Runs both at dispatch (before committing a floating selection) and at the shared runner
/// (direct specs/Last Filter). Filters outside the core and extended families are unaffected.
pub(crate) fn validate_params(id: &str, params: &Value) -> Result<()> {
    let bad = |msg| EngineError::BadParams { cmd: id.into(), msg };
    if let Some(choices) = choices(id) {
        for &(key, allowed) in choices {
            if let Some(value) = params.get(key)
                && !Rule::Choice(allowed).accepts(value)
            {
                return Err(bad(format!("`{key}` must be {}", Rule::Choice(allowed).expected())));
            }
        }
        return Ok(());
    }
    let Some(fields) = fields(id) else { return crate::gallery_cmds::validate_params(id, params) };
    if params.is_null() {
        return Ok(());
    }
    let params = params.as_object().ok_or_else(|| bad("params must be a JSON object".into()))?;
    for (key, value) in params {
        let (valid, expected) = if let Some((_, rule)) = fields.iter().find(|(name, _)| *name == key) {
            (rule.accepts(value), rule.expected())
        } else {
            match key.as_str() {
                "shapeId" | "shapeName" if id == "filter.blur.shapeBlur" => (value.as_str().is_some_and(|s| !s.is_empty()), "a nonempty string".into()),
                "shapePath" if id == "filter.blur.shapeBlur" => (
                    crate::vector_cmds::parse_path(value).is_ok_and(|path| !path.is_empty() && path.control_bounds().is_some()),
                    "a nonempty vector path".into(),
                ),
                "shapeOriginType" if id == "filter.blur.shapeBlur" => (value.as_i64() == Some(9), "the recorded custom-shape origin 9".into()),
                "layer" => (value.as_u64().is_some() || value.is_string(), "an unsigned layer id or layer name".into()),
                "coalesce" => (value.is_string(), "a string".into()),
                "target" => (
                    value.as_str().is_some_and(|s| matches!(s, "pixels" | "mask" | "filterMask" | "quickMask"))
                        || value.as_object().is_some_and(|o| {
                            o.len() == 1
                                && (o.get("channel").and_then(Value::as_u64).is_some()
                                    || o.get("color").and_then(Value::as_u64).is_some()
                                    || o.get("colors")
                                        .and_then(Value::as_array)
                                        .is_some_and(|a| !a.is_empty() && a.len() <= 64 && a.iter().all(|v| v.as_u64().is_some())))
                        }),
                    "pixels, mask, filterMask, quickMask, {\"channel\":index}, {\"color\":index} or {\"colors\":[indices]}".into(),
                ),
                _ => return Err(bad(format!("unknown parameter `{key}`"))),
            }
        };
        if !valid {
            return Err(bad(format!("`{key}` must be {expected}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
