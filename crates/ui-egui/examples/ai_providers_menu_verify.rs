//! LOCAL VERIFICATION ONLY - not for commit to main; lives on the scratch verification branch.
//! Rendered + semantic verification that Edit > AI Providers is enabled and opens the
//! provider key window. This harness exists because the r5 build shipped the entry and
//! the dialog but greyed the menu item: a flag-level check alone missed it, so this
//! clicks the real menu and renders real PNG evidence.

use egui_kittest::kittest::Queryable;
use egui_kittest::{Harness, SnapshotOptions};
use fotodit_engine::Session;
use fotodit_ui_egui::theme::{self, ThemeKind};
use fotodit_ui_egui::{FotoditApp, ai_providers_ui, menus};

const OUT: &str = "/tmp/verify-snaps";

fn snap(harness: &mut Harness<'static, FotoditApp>, name: &str) {
    let img = harness.render().expect("wgpu render");
    let options = SnapshotOptions::default().output_path(OUT);
    // No reference snapshot exists: kittest writes {OUT}/{name}.new.png and returns a
    // "new snapshot" error, which is expected here and intentionally ignored.
    let _ = egui_kittest::try_image_snapshot_options(&img, name, &options);
    // False-positive guard: a failed render or write must fail the harness, not pass silently.
    let written = format!("{OUT}/{name}.new.png");
    assert!(std::path::Path::new(&written).exists(), "rendered evidence missing: {written}");
    println!("snap {name}: {}x{}", img.width(), img.height());
}

fn menu_harness(document: bool) -> Harness<'static, FotoditApp> {
    let mut ready = false;
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1200.0, 2400.0))
        .wgpu()
        .build_ui_state(
            move |ui, app: &mut FotoditApp| {
                if !ready {
                    egui_extras::install_image_loaders(ui.ctx());
                    theme::install_fonts(ui.ctx());
                    theme::apply(ui.ctx(), ThemeKind::ALL[0]);
                    ready = true;
                    return;
                }
                menus::menu_bar(app, ui);
                // The window draws on the same context; with the flag closed this is a no-op.
                ai_providers_ui::show(app, &ui.ctx());
            },
            FotoditApp::new(Session::new(), Default::default()),
        );
    if document {
        harness.state_mut().run("file.new", serde_json::json!({"width": 64, "height": 48})).unwrap();
    }
    harness.run_steps(6);
    harness
}

fn main() {
    // Semantic + pointer proof, with and without an open document (the r5 bug disabled
    // the entry in both states, so both are covered).
    for document in [false, true] {
        let mut harness = menu_harness(document);
        let items = menus::menu_items(harness.state());
        let item = items.iter().find(|i| i.id == "edit.aiProviders").expect("Edit > AI Providers entry exists");
        assert_eq!(item.path, ["Edit"]);
        assert!(item.enabled, "AI Providers must be enabled, document={document}");

        harness.get_by_label("Edit").click();
        harness.run_steps(3);
        harness.get_by_label_contains("AI Providers").click();
        harness.run_steps(3);

        // Rendered proof, not only a flag read: the settings window's content (a
        // provider row and its masked key field) must be in the accessibility tree
        // after the click. The window TITLE is not asserted: egui does not provably
        // expose it as a queryable node (the r6 run failed exactly there, while the
        // flag-based unit regression proved the click does open the window).
        assert!(harness.state().ai_providers.open, "menu click did not open provider settings, document={document}");
        assert!(
            harness.query_by_label_contains("Google Gemini").is_some(),
            "settings window content did not render after the menu click, document={document}"
        );
        assert!(
            harness.query_all_by_role(egui::accesskit::Role::TextInput).next().is_some(),
            "no key entry field rendered, document={document}"
        );
        println!("menu click opens provider settings, document={document}");

        if !document {
            snap(&mut harness, "ai-providers-settings-open");
        }
    }

    // Rendered look at the open Edit menu with the enabled row.
    let mut harness = menu_harness(false);
    harness.get_by_label("Edit").click();
    harness.run_steps(3);
    snap(&mut harness, "edit-menu-ai-providers-enabled");

    println!("ALL VERIFICATIONS PASSED");
}
