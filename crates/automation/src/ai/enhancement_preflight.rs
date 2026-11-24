//! Early honesty for photo enhancement, not a safety or authorization boundary.
//!
//! Inspection does not expose effective locks. The admitted filter commands
//! remain authoritative and must refuse locks before any adjustment is created.
//! A filter step may omit `layer`: engine filters then target the live active
//! layer, which this check has just verified. Requiring the key would deadlock
//! against the provider contract, which forbids it (D1 in docs/ai-editing-plan.md).
//! Dry-run plans are not certificates of executability. Deferred execution still needs
//! the host's document identity/revision binding; layer IDs alone do not provide it.

use serde_json::Value;

use super::{AiError, EditingPlan};

fn refusal(reason: impl Into<String>) -> AiError {
    AiError::PlanningError(format!("photo enhancement refused: {}", reason.into()))
}

fn active_layer(layers: &[Value], id: u64) -> Option<&Value> {
    for layer in layers {
        if layer.get("id").and_then(Value::as_u64) == Some(id) {
            return Some(layer);
        }
        if let Some(children) = layer.get("children").and_then(Value::as_array)
            && let Some(found) = active_layer(children, id)
        {
            return Some(found);
        }
    }
    None
}

pub(super) fn validate(document: &Value, plan: &EditingPlan) -> Result<(), AiError> {
    // Document state gates every enhancement plan, adjustment-only included:
    // adjustment creation consumes selections and channel state, exactly what
    // this verifies. Only the sharpening-specific sequencing relaxes for
    // plans without a pixel-targeting filter step.
    let id = validate_document_state(document)?;
    let Some(first_filter) = plan.steps.iter().position(|s| s.command.starts_with("filter.")) else {
        return Ok(());
    };
    if let Some(first_adjustment) = plan.steps.iter().position(|s| s.command.starts_with("layer.newAdjustmentLayer."))
        && first_adjustment < first_filter
    {
        return Err(refusal("a pixel filter step must be the first step, before adjustment layers"));
    }
    let first = &plan.steps[first_filter];
    // Engine filter commands take no `layer` parameter and always target the
    // active layer. A paramless step therefore targets the live active layer
    // this function just verified; an explicit `layer` must match it exactly,
    // or the plan would filter a different layer than intended.
    if let Some(target) = first.params.get("layer").and_then(Value::as_u64) {
        if target != id {
            return Err(refusal("the planned sharpening target differs from the current active layer; plan again"));
        }
    }
    if first.params.get("target").is_some() {
        return Err(refusal("the sharpening step must not override the pixel target"));
    }
    Ok(())
}

/// Document-state enhancement eligibility: every check is decidable from inspection
/// alone, without a plan. Shared by execution's preflight and by the initial-analysis
/// gate in `mod.rs` so the two eligibility decisions cannot drift apart. This is an
/// early honesty check, not a safety or authorization boundary.
pub(super) fn validate_document_state(document: &Value) -> Result<u64, AiError> {
    let id = document.get("activeLayer").and_then(Value::as_u64).ok_or_else(|| refusal("no active layer"))?;
    let layers = document.get("layers").and_then(Value::as_array).ok_or_else(|| refusal("layer inspection is missing or malformed"))?;
    let layer = active_layer(layers, id).ok_or_else(|| refusal(format!("active layer {id} is absent from inspection")))?;
    let kind = layer.get("kind").and_then(Value::as_str).ok_or_else(|| refusal("active layer kind is missing"))?;
    let name = layer.get("name").and_then(Value::as_str).unwrap_or("unnamed");
    match kind {
        "Pixel" => {}
        "Smart Object" if layer.get("bounds").and_then(Value::as_array).is_some_and(|b| b.len() == 4 && b.iter().all(Value::is_i64)) => {}
        "Smart Object" => return Err(refusal(format!("layer {name:?} ({kind}) has no inspected pixel cache"))),
        _ => return Err(refusal(format!("layer {name:?} ({kind}) is not an eligible pixel layer"))),
    }
    match document.get("hasSelection").and_then(Value::as_bool) {
        Some(false) => {}
        Some(true) => return Err(refusal("clear the pixel selection first; adjustment creation consumes selections")),
        None => return Err(refusal("pixel selection state is missing or malformed")),
    }
    match document.get("quickMask").and_then(Value::as_bool) {
        Some(false) => {}
        Some(true) => return Err(refusal("leave Quick Mask mode first")),
        None => return Err(refusal("Quick Mask state is missing or malformed")),
    }
    let channels = document.get("channels").and_then(Value::as_object).ok_or_else(|| refusal("channel inspection is missing or malformed"))?;
    let target = channels.get("target").and_then(Value::as_object).ok_or_else(|| refusal("channel target is missing or malformed"))?;
    if target.len() != 1 || target.get("kind").and_then(Value::as_str) != Some("composite") {
        return Err(refusal("select the composite/all-colors target, not a channel or mask"));
    }
    // Composite serializes selectedColors as null, never a selected component list.
    if channels.get("selectedColors") != Some(&Value::Null) {
        return Err(refusal("all-colors state is missing or is restricted to selected channels"));
    }
    if document.get("layerMaskView") != Some(&Value::Null) || channels.get("layerMaskView") != Some(&Value::Null) {
        return Err(refusal("leave layer/filter mask view first, or provide complete mask-view inspection"));
    }
    Ok(id)
}
