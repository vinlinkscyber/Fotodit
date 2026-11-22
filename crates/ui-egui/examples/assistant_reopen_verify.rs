//! LOCAL VERIFICATION ONLY - not for commit to main; lives on the scratch verification branch.
//! Rendered + semantic verification of the toolbar AI-assistant reopen badge.
//!
//! Reviewer revision of the executor-authored harness: the snapshot writer actually
//! produces `{OUT}/{name}.new.png` when no reference exists (verified against
//! egui_kittest snapshot.rs at the pinned egui rev), and each snap now asserts the
//! rendered PNG really landed, so a failed write cannot false-pass.

use egui_kittest::kittest::Queryable;
use egui_kittest::{Harness, SnapshotOptions};
use fotodit_engine::Session;
use fotodit_ui_egui::theme::{self, ThemeKind};
use fotodit_ui_egui::{FotoditApp, panels};

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

fn toolbar_harness(height: f32) -> Harness<'static, FotoditApp> {
    let mut ready = false;
    let mut harness = Harness::builder()
        .with_size(egui::vec2(400.0, height))
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
                panels::toolbar(app, ui);
            },
            FotoditApp::new(Session::new(), Default::default()),
        );
    harness.run_steps(6);
    harness
}

fn dock_harness(assistant: bool) -> Harness<'static, FotoditApp> {
    let mut ready = false;
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1100.0, 700.0))
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
                panels::dock_area(app, ui);
            },
            FotoditApp::new(Session::new(), Default::default()),
        );
    harness.state_mut().ui.panels.assistant = assistant;
    harness.run_steps(6);
    harness
}

fn main() {
    for height in [700.0f32, 360.0] {
        let mut harness = toolbar_harness(height);
        assert!(harness.state().ui.panels.assistant, "rail must be open by default");
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, height));

        let badge = harness.get_by_label("AI Assistant");
        assert!(screen.contains_rect(badge.rect()), "open: badge off-viewport at {height}: {:?}", badge.rect());
        badge.click();
        harness.run_steps(3);
        assert!(!harness.state().ui.panels.assistant, "badge click must close the rail");

        let badge = harness.get_by_label("AI Assistant");
        assert!(screen.contains_rect(badge.rect()), "closed: badge off-viewport at {height}: {:?}", badge.rect());
        badge.click();
        harness.run_steps(3);
        assert!(harness.state().ui.panels.assistant, "badge click must reopen the rail");
        println!("toolbar toggle + viewport OK at {height}px");

        if height == 700.0 {
            harness.get_by_label("AI Assistant").click();
            harness.run_steps(3);
            snap(&mut harness, "toolbar-closed-badge-visible");
            harness.get_by_label("AI Assistant").click();
            harness.run_steps(3);
            snap(&mut harness, "toolbar-open-badge-selected");
        }
    }

    let mut dock_open = dock_harness(true);
    snap(&mut dock_open, "dock-assistant-rail-open");
    let mut dock_closed = dock_harness(false);
    snap(&mut dock_closed, "dock-assistant-rail-closed");

    println!("ALL VERIFICATIONS PASSED");
}
