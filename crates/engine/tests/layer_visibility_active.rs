//! Runtime check carried from the provider-path review: hiding the engine's ACTIVE
//! layer does not reassign active selection, and the engine's filter precondition does
//! not check visibility. The provider-plan validation (v6 targetsem r2) is the layer
//! that refuses a pixel filter whose active layer was hidden earlier in the plan; this
//! test locks the underlying engine behavior that validation defends against, so a
//! future engine change (auto-reassigning active on hide, or refusing filters on
//! hidden layers) surfaces here instead of silently invalidating the validation's model.

use fotodit_engine::Session;
use serde_json::json;

fn active_layer(doc: &serde_json::Value) -> u64 {
    doc["activeLayer"].as_u64().expect("a document with layers has an active layer")
}
fn layer<'a>(doc: &'a serde_json::Value, id: u64) -> &'a serde_json::Value {
    doc["layers"].as_array().unwrap().iter().find(|l| l["id"].as_u64() == Some(id)).expect("layer present")
}

#[test]
fn hiding_the_active_layer_keeps_the_active_selection_and_does_not_block_filters() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 32, "height": 32, "background": "transparent"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    let id = active_layer(&fotodit_engine::inspect::document(s.active().unwrap()));

    // Hiding the active layer: the active selection is NOT reassigned or cleared.
    s.execute("layer.setProps", json!({"layer": id, "visible": false})).unwrap();
    let doc = fotodit_engine::inspect::document(s.active().unwrap());
    assert_eq!(active_layer(&doc), id, "hiding the active layer must not reassign active selection");
    assert_eq!(layer(&doc, id)["visible"].as_bool(), Some(false));

    // The engine itself still filters the hidden active pixel layer: visibility is not
    // part of the filter precondition. Provider plans are kept from this sequence by
    // the plan validation, not by the engine.
    s.execute("filter.blur.gaussianBlur", json!({"radius": 1.0})).unwrap();

    // Un-hide: the same layer is visible again with the active selection untouched,
    // and a filter then edits the now-visible layer.
    s.execute("layer.setProps", json!({"layer": id, "visible": true})).unwrap();
    let doc = fotodit_engine::inspect::document(s.active().unwrap());
    assert_eq!(active_layer(&doc), id);
    assert_eq!(layer(&doc, id)["visible"].as_bool(), Some(true));
    s.execute("filter.blur.gaussianBlur", json!({"radius": 1.0})).unwrap();
}
