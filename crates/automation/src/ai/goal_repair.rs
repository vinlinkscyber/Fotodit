//! Goal repair adapter for the goal-session coordinator.
//!
//! [`SemanticGoalRepair`] consumes the latest completed-but-unmet attempt and
//! the CURRENT document scene to produce a bounded corrective plan. It never
//! re-runs the first-pass planner blindly: corrections edit existing type
//! layers in place through the catalog-admitted `type.edit` command
//! (`{"layer": id, "text": str}` - replaces all text, keeps styles), so no
//! duplicate layers are created and the goal is never switched.
//!
//! Grounded command semantics (verified against the engine, not assumed):
//! - Admission: `type.edit` is in the automation `CommandCatalog`, so the
//!   coordinator's pre-dispatch `validate_command` accepts it. At execution,
//!   `Session::admit` resolves an explicit `"layer": id` target via
//!   `explicit_layer_enabled` (crates/engine/src/session/execute.rs, #466):
//!   the precondition is checked on the NAMED layer and the active layer is
//!   restored unchanged - the plan step itself owns target selection, no
//!   separate select step and no GUI involvement.
//! - Text replacement: `"text"` replaces the layer's whole content while
//!   preserving its styling runs (crates/engine/src/cmd/type_cmds.rs
//!   `"text":str? (replace all, styles kept)`), so position, font and color
//!   survive the correction.
//! - History: the edit is journaled under the "Edit Type" label
//!   (`with_text_layer` -> `Session::edit`), one undoable history entry per
//!   step, compatible with the executor's rollback contract.
//!
//! A layer that already carries a user-required term is NEVER a replacement
//! target: the repair only rewrites layers whose current text contains none
//! of the requested terms, so a satisfied term cannot be destroyed to place
//! a missing one. On top of that, replacement candidates are restricted to
//! type layers CREATED BY THIS GOAL's own attempts (recovered from the
//! coordinator's attempt ledger: `type.create` command outputs), so a repair
//! on a pre-existing document can never overwrite the user's own prior text.
//! When a semantic repair is unsupported, the adapter says so honestly via
//! [`GoalRepairDecision::NeedsInput`] or [`GoalRepairDecision::Stop`].
//!
//! Integration seam: this file references the coordinator contract
//! (`super::goal_session::{GoalRepairAdapter, GoalRepairDecision,
//! GoalRepairInput}`) authored separately. The `pub mod goal_repair;` export
//! line is coordinated through the module seam owner; this file does not touch
//! `mod.rs` itself. It compiles only once that contract file lands.
//!
//! Inspect-derived scenes may omit optional color statistics, so no color
//! assumptions are made here; repairs are limited to text-layer content.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde_json::json;

use std::collections::BTreeSet;

use super::goal_session::{GoalAttempt, GoalRepairAdapter, GoalRepairDecision, GoalRepairInput};
use super::natural_language::NaturalLanguageEngine;
use super::perception::{VisualElement, VisualScene};
use super::planner::PlanStep;
use super::{AiError, EditingIntent, EditingPlan, IntentCategory};

/// Deterministic repair adapter: fixes unmet user-supplied subject terms by
/// rewriting the text of existing type layers in place.
#[derive(Clone, Debug, Default)]
pub struct SemanticGoalRepair;

impl SemanticGoalRepair {
    /// Create a new semantic goal repair adapter.
    pub fn new() -> Self {
        Self
    }

    /// Visible type layers ordered by prominence (largest font first), matching
    /// the evaluator's view of the scene.
    fn text_layers_by_prominence(scene: &VisualScene) -> Vec<&VisualElement> {
        let mut layers: Vec<&VisualElement> = scene
            .layers
            .iter()
            .filter(|l| l.kind == "Type" && l.is_visible)
            .collect();
        layers.sort_by(|a, b| {
            let size_a = a.font_size_pt.unwrap_or(12.0);
            let size_b = b.font_size_pt.unwrap_or(12.0);
            size_b.partial_cmp(&size_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        layers
    }

    /// Whether a layer's text already carries any user-required term.
    fn carries_required_term(layer: &VisualElement, terms: &[String]) -> bool {
        layer
            .text_content
            .as_deref()
            .map(|text| {
                let lower = text.to_lowercase();
                terms.iter().any(|t| lower.contains(&t.to_lowercase()))
            })
            .unwrap_or(false)
    }

    /// Required user-supplied terms that no current visible text layer contains.
    fn missing_terms(scene: &VisualScene, terms: &[String]) -> Vec<String> {
        let layers = Self::text_layers_by_prominence(scene);
        terms
            .iter()
            .filter(|term| !layers.iter().any(|l| Self::carries_required_term(l, std::slice::from_ref(term))))
            .cloned()
            .collect()
    }

    /// Layer ids this goal's attempts actually created (`type.create` outputs
    /// recorded in the attempt ledger). Only records without an error are
    /// trusted: a failed `type.create` may carry a stale or partial output
    /// that does not correspond to a real layer. Observation records
    /// (`visual_digest` etc.) never match the command filter.
    fn goal_created_layers(attempts: &[GoalAttempt]) -> BTreeSet<u64> {
        let mut created = BTreeSet::new();
        for attempt in attempts {
            for record in &attempt.commands {
                if record.command == "type.create" && record.error.is_none() {
                    if let Some(id) = record
                        .output
                        .as_ref()
                        .and_then(|o| o.get("layer"))
                        .and_then(serde_json::Value::as_u64)
                    {
                        created.insert(id);
                    }
                }
            }
        }
        created
    }

    /// A single corrective step: replace the full text of an existing type
    /// layer with the required term, preserving the layer, its position, and
    /// its styling. Uses the catalog-admitted `type.edit` with an explicit
    /// layer target; the engine admits it against that named layer without
    /// changing the active selection.
    fn replacement_step(layer: &VisualElement, term: &str) -> PlanStep {
        PlanStep {
            command: "type.edit".into(),
            params: json!({
                "layer": layer.id,
                "text": term,
            }),
            description: format!(
                "Replace text of existing layer '{}' with the requested '{term}'",
                layer.name
            ),
        }
    }

    /// Core decision, separated from the trait so it stays directly testable.
    /// `prior_steps` are the steps of the most recent attempt's plan, used to
    /// stop before re-dispatching an identical corrective plan (the coordinator
    /// also fingerprints plans; this keeps the adapter honest standalone).
    /// `created` holds the layers this goal's attempts created; when it is
    /// empty the scene's text layers are the user's own and repair must stop.
    fn decide(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        prior_steps: Option<&[PlanStep]>,
        created: &BTreeSet<u64>,
    ) -> GoalRepairDecision {
        if intent.category != IntentCategory::GraphicDesign {
            return GoalRepairDecision::Stop(format!(
                "semantic repair only supports graphic-design goals, not {:?}",
                intent.category
            ));
        }

        let terms = NaturalLanguageEngine::new().subject_terms(&intent.raw_prompt);
        if terms.is_empty() {
            return GoalRepairDecision::Stop(
                "the original prompt supplied no subject text to repair".into(),
            );
        }

        let missing = Self::missing_terms(scene, &terms);
        if missing.is_empty() {
            return GoalRepairDecision::Stop(
                "all requested subject terms are already present in the scene".into(),
            );
        }

        // Replacement targets: visible type layers created by THIS GOAL's
        // attempts whose current text carries NONE of the required terms,
        // most prominent first. Layers that already carry satisfied user copy
        // and all pre-existing user layers are never rewritten.
        let targets: Vec<&VisualElement> = Self::text_layers_by_prominence(scene)
            .into_iter()
            .filter(|l| created.contains(&l.id) && !Self::carries_required_term(l, &terms))
            .collect();
        if targets.is_empty() {
            if created.is_empty() {
                return GoalRepairDecision::Stop(
                    "the goal created no text layers; the scene's text is the user's own and must not be overwritten"
                        .into(),
                );
            }
            return GoalRepairDecision::NeedsInput(
                "every goal-created text layer already carries requested wording; there is no placeholder layer to correct in place"
                    .into(),
            );
        }
        if targets.len() < missing.len() {
            return GoalRepairDecision::NeedsInput(format!(
                "scene has {} replaceable text layer(s) but {} requested term(s) are missing; cannot place them without duplicating layers or overwriting requested wording",
                targets.len(),
                missing.len()
            ));
        }

        let steps: Vec<PlanStep> = missing
            .iter()
            .zip(targets.iter())
            .map(|(term, layer)| Self::replacement_step(layer, term))
            .collect();

        if prior_steps.map(|prev| prev == steps.as_slice()).unwrap_or(false) {
            return GoalRepairDecision::Stop(
                "the identical corrective plan already ran and the goal remains unmet".into(),
            );
        }

        GoalRepairDecision::Plan(EditingPlan {
            title: "Semantic goal correction".into(),
            rationale: format!(
                "Replaces existing text layer content with the user-requested wording ({}) in place, without creating duplicate layers.",
                missing.join(", ")
            ),
            category: IntentCategory::GraphicDesign,
            steps,
            creates_document: false,
            target_size: None,
        })
    }
}

impl GoalRepairAdapter for SemanticGoalRepair {
    fn plan_repair(&self, input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError> {
        let prior_steps = input.attempts.last().map(|a| a.plan.steps.as_slice());
        let created = Self::goal_created_layers(input.attempts);
        Ok(self.decide(input.original_intent, input.current_scene, prior_steps, &created))
    }
}

/// Provider-assisted corrective planning candidate.
///
/// Prepared per the integration plan but intentionally inert: no provider
/// calls are made until provider scope, cost, and privacy are verified with
/// the owner. Until then every request stops honestly. This stub is not a
/// reusable provider integration.
#[derive(Clone, Debug, Default)]
pub struct ProviderGoalRepair;

impl ProviderGoalRepair {
    /// Create a new provider repair candidate (currently always gated off).
    pub fn new() -> Self {
        Self
    }

    /// Whether live provider repair is authorized. Hard-coded `false` until
    /// the owner verifies provider scope, cost, and privacy; never read from
    /// ambient environment to avoid accidental activation.
    pub fn is_authorized(&self) -> bool {
        false
    }
}

impl GoalRepairAdapter for ProviderGoalRepair {
    fn plan_repair(&self, _input: GoalRepairInput<'_>) -> Result<GoalRepairDecision, AiError> {
        Ok(GoalRepairDecision::Stop(
            "provider-assisted repair is not enabled; provider scope, cost, and privacy are not yet verified"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_layer(id: u64, name: &str, text: &str, size: f32) -> VisualElement {
        VisualElement {
            id,
            name: name.into(),
            kind: "Type".into(),
            text_content: Some(text.into()),
            font_size_pt: Some(size),
            bounds: [0, 0, 100, 40],
            is_visible: true,
            has_mask: false,
            confidence: 1.0,
        }
    }

    fn pixel_layer(id: u64) -> VisualElement {
        VisualElement {
            id,
            name: "Background".into(),
            kind: "Pixel".into(),
            text_content: None,
            font_size_pt: None,
            bounds: [0, 0, 800, 600],
            is_visible: true,
            has_mask: false,
            confidence: 1.0,
        }
    }

    fn scene(layers: Vec<VisualElement>) -> VisualScene {
        VisualScene {
            revision: Some(1),
            width: 800,
            height: 600,
            aspect_ratio: "4:3".into(),
            resolution_dpi: 72.0,
            active_layer_id: None,
            layers,
            typography: Vec::new(),
            dominant_colors: Vec::new(),
            whitespace_margins: [0, 0, 0, 0],
            has_selection: false,
            selection_bounds: None,
        }
    }

    fn created(ids: &[u64]) -> BTreeSet<u64> {
        ids.iter().copied().collect()
    }

    fn design_intent(prompt: &str) -> EditingIntent {
        EditingIntent {
            raw_prompt: prompt.into(),
            category: IntentCategory::GraphicDesign,
            archetype: None,
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: None,
            text_content: None,
        }
    }

    #[test]
    fn replaces_canned_headline_with_requested_term() {
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster called chndrax");
        let scene = scene(vec![
            pixel_layer(1),
            text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0),
        ]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::Plan(_)));
        if let GoalRepairDecision::Plan(plan) = decision {
            assert_eq!(plan.steps.len(), 1);
            assert_eq!(plan.steps[0].command, "type.edit");
            assert_eq!(plan.steps[0].params["layer"], 2);
            assert_eq!(plan.steps[0].params["text"], "chndrax");
            assert!(!plan.creates_document);
        }
    }

    #[test]
    fn never_overwrites_layer_carrying_satisfied_term() {
        // "Grand Opening" is already on the canvas; only the brand is missing.
        // The repair must target the canned subhead, not the good headline.
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster \"Grand Opening\" called chndrax");
        let scene = scene(vec![
            text_layer(2, "Headline", "Grand Opening", 48.0),
            text_layer(3, "Subhead", "A Celebration of Art & Technology", 24.0),
        ]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::Plan(_)));
        if let GoalRepairDecision::Plan(plan) = decision {
            assert_eq!(plan.steps.len(), 1);
            assert_eq!(plan.steps[0].params["layer"], 3);
            assert_eq!(plan.steps[0].params["text"], "chndrax");
        }
    }

    #[test]
    fn stops_when_term_already_present() {
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster called chndrax");
        let scene = scene(vec![text_layer(2, "Headline", "chndrax", 48.0)]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::Stop(_)));
    }

    #[test]
    fn needs_input_when_fewer_replaceable_layers_than_missing_terms() {
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster \"Grand Opening\" called chndrax");
        let scene = scene(vec![text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0)]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::NeedsInput(_)));
    }

    #[test]
    fn places_each_missing_term_in_its_own_layer() {
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster \"Grand Opening\" called chndrax");
        let scene = scene(vec![
            text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0),
            text_layer(3, "Subhead", "A Celebration of Art & Technology", 24.0),
        ]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::Plan(_)));
        if let GoalRepairDecision::Plan(plan) = decision {
            assert_eq!(plan.steps.len(), 2);
            // Most prominent layer gets the first (quoted) term.
            assert_eq!(plan.steps[0].params["layer"], 2);
            assert_eq!(plan.steps[0].params["text"], "Grand Opening");
            assert_eq!(plan.steps[1].params["layer"], 3);
            assert_eq!(plan.steps[1].params["text"], "chndrax");
        }
    }

    #[test]
    fn stops_identical_corrective_plan_repeat() {
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster called chndrax");
        let scene = scene(vec![text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0)]);
        let first = repair.decide(&intent, &scene, None, &created(&[2]));
        assert!(matches!(first, GoalRepairDecision::Plan(_)));
        if let GoalRepairDecision::Plan(plan) = first {
            let second = repair.decide(&intent, &scene, Some(plan.steps.as_slice()), &created(&[2]));
            assert!(matches!(second, GoalRepairDecision::Stop(_)));
        }
    }

    #[test]
    fn collector_trusts_only_successful_type_create_records() {
        use super::super::goal_session::{GoalAttempt, GoalCommandRecord};

        fn plan_shell() -> EditingPlan {
            EditingPlan {
                title: "attempt".into(),
                rationale: "test".into(),
                category: IntentCategory::GraphicDesign,
                steps: Vec::new(),
                creates_document: false,
                target_size: None,
            }
        }
        fn record(command: &str, output: Option<serde_json::Value>, error: Option<&str>, observation: bool) -> GoalCommandRecord {
            GoalCommandRecord {
                command: command.into(),
                output,
                error: error.map(str::to_string),
                observation,
            }
        }

        let attempt = GoalAttempt {
            plan: plan_shell(),
            execution: None,
            evaluation: None,
            evidence: None,
            commands: vec![
                // Successful creation: trusted.
                record("type.create", Some(json!({"layer": 2})), None, false),
                // Errored creation with a stale-looking output: NOT trusted.
                record("type.create", Some(json!({"layer": 3})), Some("engine refused"), false),
                // Observation record with a layer-shaped payload: NOT an edit.
                record("visual_digest", Some(json!({"layer": 4})), None, true),
                // Successful creation whose output lacks a layer id: skipped.
                record("type.create", Some(json!({"ok": true})), None, false),
                // Other successful edit commands: irrelevant.
                record("type.edit", Some(json!({"layer": 5})), None, false),
            ],
            error: None,
        };

        let collected = SemanticGoalRepair::goal_created_layers(&[attempt]);
        assert!(collected.contains(&2));
        assert!(!collected.contains(&3), "errored type.create output must not be trusted");
        assert!(!collected.contains(&4), "observation records are not edits");
        assert!(!collected.contains(&5));
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn stops_instead_of_overwriting_preexisting_user_text() {
        // Preservation counterexample: an existing document holds the user's
        // own text layer that carries no required term. The goal created
        // nothing, so there is no legitimate target - repair must stop rather
        // than rewrite the user's content.
        let repair = SemanticGoalRepair::new();
        let intent = design_intent("make a poster called chndrax");
        let scene_user_only = scene(vec![text_layer(9, "My Notes", "meeting at five", 18.0)]);
        let decision = repair.decide(&intent, &scene_user_only, None, &created(&[]));
        assert!(matches!(decision, GoalRepairDecision::Stop(_)));

        // Even when the goal created other layers, a pre-existing user layer
        // outside the goal ledger is never a target.
        let scene = scene(vec![
            text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0),
            text_layer(9, "My Notes", "meeting at five", 18.0),
        ]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2]));
        assert!(matches!(decision, GoalRepairDecision::Plan(_)));
        if let GoalRepairDecision::Plan(plan) = decision {
            assert_eq!(plan.steps.len(), 1);
            assert_eq!(plan.steps[0].params["layer"], 2);
        }
    }

    #[test]
    fn stops_unsupported_category() {
        let repair = SemanticGoalRepair::new();
        let mut intent = design_intent("make a poster called chndrax");
        intent.category = IntentCategory::PhotoEnhancement;
        let scene = scene(vec![text_layer(2, "Headline", "CREATIVE EXPO 2026", 48.0)]);
        let decision = repair.decide(&intent, &scene, None, &created(&[2, 3]));
        assert!(matches!(decision, GoalRepairDecision::Stop(_)));
    }

    #[test]
    fn provider_candidate_never_calls() {
        let repair = ProviderGoalRepair::new();
        assert!(!repair.is_authorized());
    }

    /// Real engine round-trip: build an actual document, run the corrective
    /// plan through the admitted execution path, re-inspect the live scene,
    /// and let the real evaluator judge the after state. This proves the
    /// repair plan passes admission, targets the named layer without any
    /// select step, preserves the second layer, and satisfies the evaluator.
    #[test]
    fn engine_round_trip_repair_satisfies_evaluator() {
        use super::super::evaluation::EvaluationEngine;
        use super::super::execution::ExecutionEngine;
        use super::super::perception::PerceptionEngine;
        use crate::ai::live::LiveEditSession;
        use crate::headless::Headless;

        let mut headless = Headless::trusted_local();
        let Ok(_) = headless.command_run("file.new", json!({"width": 800, "height": 600, "background": "#0f172a"})) else {
            assert!(false, "file.new must succeed");
            return;
        };
        let Ok(headline) = headless.command_run(
            "type.create",
            json!({"x": 100, "y": 120, "text": "CREATIVE EXPO 2026", "size": 48.0, "color": "#f8fafc"}),
        ) else {
            assert!(false, "type.create headline must succeed");
            return;
        };
        let Ok(subhead) = headless.command_run(
            "type.create",
            json!({"x": 100, "y": 220, "text": "A Celebration of Art & Technology", "size": 24.0, "color": "#cbd5e1"}),
        ) else {
            assert!(false, "type.create subhead must succeed");
            return;
        };
        let (Some(headline_id), Some(subhead_id)) = (headline["layer"].as_u64(), subhead["layer"].as_u64()) else {
            assert!(false, "type.create must return layer ids");
            return;
        };
        let goal_layers = created(&[headline_id, subhead_id]);

        let perception = PerceptionEngine::new();
        let Ok(Some(doc_json)) = headless.inspect_active() else {
            assert!(false, "inspect must return the document");
            return;
        };
        let before_scene = perception.perceive(&doc_json);

        let intent = design_intent("make a poster \"Grand Opening\" called chndrax");
        let repair = SemanticGoalRepair::new();
        let decision = repair.decide(&intent, &before_scene, None, &goal_layers);
        assert!(matches!(decision, GoalRepairDecision::Plan(_)));
        let GoalRepairDecision::Plan(plan) = decision else {
            return;
        };

        let execution = ExecutionEngine::execute_plan_in(&mut headless, &plan);
        assert!(execution.is_ok(), "repair plan must pass admission and execute");
        let Ok(execution) = execution else { return };
        assert!(execution.is_success, "repair execution must complete: {:?}", execution.error);

        let Ok(Some(after_json)) = headless.inspect_active() else {
            assert!(false, "post-repair inspect must return the document");
            return;
        };
        let after_scene = perception.perceive(&after_json);
        let report = EvaluationEngine::new().evaluate(&intent, &before_scene, &after_scene, &execution);
        assert!(
            report.intent_satisfied,
            "evaluator must see both required terms after repair: {:?}",
            report.observations
        );
        // The satisfied quoted term must survive on its own layer.
        let surviving: Vec<&str> = after_scene
            .layers
            .iter()
            .filter(|l| l.kind == "Type")
            .filter_map(|l| l.text_content.as_deref())
            .collect();
        assert!(surviving.iter().any(|t| t.contains("Grand Opening")));
        assert!(surviving.iter().any(|t| t.contains("chndrax")));
    }
}
