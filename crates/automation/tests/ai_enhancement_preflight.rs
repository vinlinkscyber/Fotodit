//! Offline tests only. Real commands use Headless admission, not a production raw Session adapter.
use fotodit_automation::ai::{AiContext, AiEngine, AiError, EditingPlan, ExecutionEngine, LiveEditSession};
use fotodit_automation::{AutomationError, Headless};
use serde_json::{Value, json};

const PROMPT: &str = "Make this photograph look professional and enhance clarity";

#[derive(Default)]
struct MockSession {
    document: Option<Value>,
    calls: Vec<String>,
}

impl LiveEditSession for MockSession {
    fn inspect_active(&self) -> Result<Option<Value>, AutomationError> {
        Ok(self.document.clone())
    }
    fn command_run(&mut self, command: &str, _: Value) -> Result<Value, AutomationError> {
        self.calls.push(command.into());
        Ok(Value::Null)
    }
}

fn fixture() -> Headless {
    let mut host = Headless::trusted_local();
    host.command_run("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    host
}

fn planned(host: &Headless) -> EditingPlan {
    AiEngine::new().plan_instruction(host, PROMPT, &AiContext::new()).unwrap()
}

fn inspect(host: &Headless) -> Value {
    host.inspect_active().unwrap().unwrap()
}

fn refuses(document: Option<Value>, plan: &EditingPlan, reason: &str) {
    let mut host = MockSession { document, ..Default::default() };
    let error = ExecutionEngine::execute_plan_in(&mut host, plan).unwrap_err();
    assert!(error.to_string().contains(reason), "{error}");
    assert!(host.calls.is_empty(), "refusal dispatched {:?}", host.calls);
}

#[test]
fn binds_active_id_and_sharpens_before_adjustments() {
    let host = fixture();
    let plan = planned(&host);
    assert_eq!(
        plan.steps.iter().map(|s| s.command.as_str()).collect::<Vec<_>>(),
        vec!["filter.sharpen.unsharpMask", "layer.newAdjustmentLayer.curves", "layer.newAdjustmentLayer.vibrance",]
    );
    assert_eq!(plan.steps[0].params["layer"], inspect(&host)["activeLayer"]);
}

#[test]
fn real_raster_and_cached_smart_succeed_including_nested_active() {
    for smart in [false, true] {
        for nested in [false, true] {
            let mut host = fixture();
            if smart {
                host.command_run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
            }
            let original_id = inspect(&host)["activeLayer"].clone();
            if nested {
                host.command_run("layer.new.groupFromLayers", json!({})).unwrap();
                host.command_run("layer.select", json!({"layer": original_id})).unwrap();
            }
            let result = AiEngine::new().process_instruction(&mut host, PROMPT, &mut AiContext::new()).unwrap();
            assert!(result.execution.is_success);
            assert_eq!(result.plan.steps[0].params["layer"], original_id);
            let st = host.session.active().unwrap();
            let layer = st.doc.layer(fotodit_doc::LayerId(original_id.as_u64().unwrap())).unwrap();
            if let fotodit_doc::LayerContent::Smart(sm) = &layer.content {
                assert_eq!(sm.smart_filters.len(), 1);
                assert_eq!(sm.smart_filters[0].command, "filter.sharpen.unsharpMask");
            }
        }
    }
}

#[test]
fn own_and_inherited_pixel_or_full_locks_refuse_at_step_zero_without_history_changes() {
    for smart in [false, true] {
        for inherited in [false, true] {
            for lock in [json!({"pixels": true}), json!({"all": true})] {
                let mut host = fixture();
                if smart {
                    host.command_run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
                }
                let id = inspect(&host)["activeLayer"].clone();
                if inherited {
                    host.command_run("layer.new.groupFromLayers", json!({})).unwrap();
                }
                host.command_run("layer.lockLayers", lock.clone()).unwrap();
                host.command_run("layer.select", json!({"layer": id})).unwrap();
                let before = inspect(&host);
                let doc = host.session.active().unwrap().doc.clone();
                let plan = planned(&host); // Dry-run does not claim to validate locks.
                let error = ExecutionEngine::execute_plan(&mut host, &plan).unwrap_err();
                assert!(matches!(error, AiError::ExecutionError { step: 0, .. }), "{error}");
                assert!(error.to_string().contains("locked"), "{error}");
                assert_eq!(inspect(&host), before);
                assert_eq!(host.session.active().unwrap().doc, doc);
            }
        }
    }
}

#[test]
fn unsupported_and_uncached_kinds_refuse_with_identity_even_when_nested() {
    let host = fixture();
    let plan = planned(&host);
    for kind in ["Type", "Adjustment", "Group", "Fill", "Shape", "Smart Object", "Unknown"] {
        for nested in [false, true] {
            let mut doc = inspect(&host);
            let mut layer = doc["layers"][0].clone();
            layer["kind"] = json!(kind);
            layer["name"] = json!("Intended subject");
            layer["bounds"] = Value::Null;
            doc["layers"] = if nested { json!([{"id": 99999, "kind": "Group", "children": [layer]}]) } else { json!([layer]) };
            refuses(Some(doc), &plan, &format!("Intended subject\" ({kind})"));
        }
    }
}

#[test]
fn no_document_no_active_missing_and_malformed_metadata_refuse_without_dispatch() {
    let host = fixture();
    let plan = planned(&host);
    refuses(None, &plan, "no document open");
    let mut doc = inspect(&host);
    doc["activeLayer"] = Value::Null;
    refuses(Some(doc), &plan, "no active layer");
    for field in ["hasSelection", "quickMask", "channels", "layerMaskView"] {
        let mut doc = inspect(&host);
        doc.as_object_mut().unwrap().remove(field);
        refuses(
            Some(doc),
            &plan,
            if field == "channels" {
                "channel"
            } else if field == "hasSelection" {
                "selection"
            } else if field == "quickMask" {
                "Quick Mask"
            } else {
                "mask"
            },
        );
    }
    for field in ["target", "selectedColors", "layerMaskView"] {
        let mut doc = inspect(&host);
        doc["channels"].as_object_mut().unwrap().remove(field);
        refuses(
            Some(doc),
            &plan,
            if field == "layerMaskView" {
                "mask"
            } else if field == "selectedColors" {
                "all-colors"
            } else {
                "target"
            },
        );
    }
}

#[test]
fn selection_channels_quick_mask_and_mask_view_refuse_without_mutation() {
    let host = fixture();
    let plan = planned(&host);
    for target in ["alpha", "color", "layerMask", "vectorMask", "unknown"] {
        let mut doc = inspect(&host);
        doc["channels"]["target"] = json!({"kind": target});
        refuses(Some(doc), &plan, "composite");
    }
    for (field, reason) in [("hasSelection", "selection"), ("quickMask", "Quick Mask")] {
        let mut doc = inspect(&host);
        doc[field] = json!(true);
        refuses(Some(doc), &plan, reason);
    }
    let mut doc = inspect(&host);
    doc["channels"]["selectedColors"] = json!([0]);
    refuses(Some(doc), &plan, "all-colors");
    for filter in [false, true] {
        let mut doc = inspect(&host);
        doc["layerMaskView"] = json!({"layer": doc["activeLayer"], "mode": "gray", "filter": filter});
        refuses(Some(doc), &plan, "mask");
    }
}

#[test]
fn stale_misdirected_and_misordered_plans_refuse_before_adjustments() {
    let host = fixture();
    let plan = planned(&host);
    let doc = inspect(&host);
    let mut changed = plan.clone();
    changed.steps.swap(0, 1);
    refuses(Some(doc.clone()), &changed, "first step");
    let mut changed = plan.clone();
    changed.steps[0].params["layer"] = json!(99999);
    refuses(Some(doc.clone()), &changed, "differs");
    let mut changed = plan.clone();
    changed.steps[0].params["target"] = json!("mask");
    refuses(Some(doc), &changed, "override");
}

#[test]
fn dry_run_is_nonmutating_but_execution_and_generic_fallback_are_guarded() {
    let host = fixture();
    let mut doc = inspect(&host);
    doc["hasSelection"] = json!(true);
    let mut mock = MockSession { document: Some(doc), ..Default::default() };
    let ai = AiEngine::new();
    for prompt in [PROMPT, "Do something useful"] {
        let plan = ai.plan_instruction_in(&mock, prompt, &AiContext::new()).unwrap();
        assert_eq!(plan.category, fotodit_automation::ai::IntentCategory::PhotoEnhancement);
        assert!(mock.calls.is_empty());
        let mut context = AiContext::new();
        let error = ai.process_instruction_in(&mut mock, prompt, &mut context).unwrap_err();
        assert!(error.to_string().contains("selection"), "{error}");
        assert!(mock.calls.is_empty());
        assert!(context.executed_plans.is_empty());
    }
}

#[test]
fn admitted_documents_receive_initial_color_analysis_before_planning() {
    for smart in [false, true] {
        let mut host = fixture();
        if smart {
            host.command_run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        }
        let doc = inspect(&host);
        let mut mock = MockSession { document: Some(doc), ..Default::default() };
        let mut context = AiContext::new();
        let result = AiEngine::new().process_instruction_in(&mut mock, PROMPT, &mut context).unwrap();
        assert!(result.execution.is_success);
        assert_eq!(
            mock.calls.first().map(String::as_str),
            Some("image.analysis.colorStats"),
            "admitted document must receive initial analysis, got {:?}",
            mock.calls
        );
        assert_eq!(
            mock.calls.last().map(String::as_str),
            Some("image.analysis.colorStats"),
            "admitted document keeps the post-execution analysis, got {:?}",
            mock.calls
        );
    }
}

#[test]
fn refused_enhancement_states_skip_initial_analysis_and_dispatch_nothing() {
    let host = fixture();
    let cases: [(fn(&mut Value), &str); 3] = [
        (|doc| doc["hasSelection"] = json!(true), "selection"),
        (|doc| doc["quickMask"] = json!(true), "Quick Mask"),
        (
            |doc| {
                doc.as_object_mut().unwrap().remove("hasSelection");
            },
            "missing or malformed",
        ),
    ];
    for (mutate, reason) in cases {
        let mut doc = inspect(&host);
        mutate(&mut doc);
        let mut mock = MockSession { document: Some(doc), ..Default::default() };
        let error = AiEngine::new().process_instruction_in(&mut mock, PROMPT, &mut AiContext::new()).unwrap_err();
        assert!(error.to_string().contains(reason), "{error}");
        assert!(mock.calls.is_empty(), "refused state dispatched {:?}", mock.calls);
    }
}

#[test]
fn multiple_violations_refuse_in_inspection_order_without_dispatch() {
    let host = fixture();
    let mut doc = inspect(&host);
    doc["hasSelection"] = json!(true);
    doc["quickMask"] = json!(true);
    let mut mock = MockSession { document: Some(doc), ..Default::default() };
    let error = AiEngine::new().process_instruction_in(&mut mock, PROMPT, &mut AiContext::new()).unwrap_err();
    assert!(error.to_string().contains("selection"), "{error}");
    assert!(mock.calls.is_empty(), "refused state dispatched {:?}", mock.calls);
}

#[test]
fn no_document_without_creative_intent_errors_before_any_dispatch() {
    let mut mock = MockSession::default();
    let error = AiEngine::new()
        .process_instruction_in(&mut mock, "Do something useful", &mut AiContext::new())
        .unwrap_err();
    assert!(matches!(error, AiError::NoDocumentOpen), "{error}");
    assert!(mock.calls.is_empty(), "no-document path dispatched {:?}", mock.calls);
}

#[test]
fn valid_non_enhancement_runs_with_absent_initial_and_permitted_final_stats() {
    let host = fixture();
    let mut doc = inspect(&host);
    doc["hasSelection"] = json!(true);
    let mut mock = MockSession { document: Some(doc), ..Default::default() };
    let mut context = AiContext::new();
    let result = AiEngine::new()
        .process_instruction_in(&mut mock, "Create an elegant wedding invitation", &mut context)
        .unwrap();
    assert_eq!(result.plan.category, fotodit_automation::ai::IntentCategory::GraphicDesign);
    assert!(result.execution.is_success);
    let analyses = mock.calls.iter().filter(|c| c.as_str() == "image.analysis.colorStats").count();
    assert_eq!(analyses, 1, "only the permitted final analysis, got {:?}", mock.calls);
    assert_eq!(mock.calls.last().map(String::as_str), Some("image.analysis.colorStats"));
}

#[test]
fn real_selection_refusal_preserves_pixels_selection_history_and_context() {
    let mut host = fixture();
    host.command_run("select.all", json!({})).unwrap();
    let before = inspect(&host);
    let doc = host.session.active().unwrap().doc.clone();
    let mut context = AiContext::new();
    let error = AiEngine::new().process_instruction(&mut host, PROMPT, &mut context).unwrap_err();
    assert!(error.to_string().contains("selection"), "{error}");
    assert_eq!(inspect(&host), before);
    assert_eq!(host.session.active().unwrap().doc, doc);
    assert!(context.executed_plans.is_empty());
}

#[test]
fn resolved_target_different_from_active_is_not_silently_edited() {
    let host = fixture();
    let mut doc = inspect(&host);
    let mut other = doc["layers"][0].clone();
    other["id"] = json!(99999);
    other["name"] = json!("OtherSubject");
    doc["layers"].as_array_mut().unwrap().push(other);
    let mock = MockSession { document: Some(doc), ..Default::default() };
    let error = AiEngine::new().plan_instruction_in(&mock, "Enhance OtherSubject professionally", &AiContext::new()).unwrap_err();
    assert!(error.to_string().contains("select the intended layer"), "{error}");
    assert!(mock.calls.is_empty());
}
