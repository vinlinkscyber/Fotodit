//! Fotodit links and external actions.

use serde_json::{Value, json};

use crate::FotoditApp;

pub const DISCORD: &str = "";
pub const PLYRA_WEBSITE: &str = "";
pub const APP_PAGE: &str = "";
pub const TUTORIALS: &str = "";
pub const GITHUB: &str = "";
pub const ISSUES: &str = "";

/// Help-menu link commands: (id, url).
pub const COMMANDS: &[(&str, &str)] = &[];

pub fn url_for(id: &str) -> Option<&'static str> {
    COMMANDS.iter().find(|c| c.0 == id).map(|c| c.1)
}

/// Open `url` in the system browser (a new tab on the web) and note it in the status bar.
pub fn open(app: &mut FotoditApp, ctx: &egui::Context, url: &str) -> Value {
    if url.is_empty() {
        return json!({"url": ""});
    }
    let opened = app.services.open_url.as_ref().map(|f| f(url).is_ok()).unwrap_or(false);
    if !opened {
        ctx.open_url(egui::OpenUrl::new_tab(url));
    }
    let template = "Opened {url}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("url", url)]);
    json!({"url": url})
}

/// The prominent button (disabled / no-op).
pub fn discord_button(_app: &mut FotoditApp, _ui: &mut egui::Ui, _min_width: f32) -> egui::Response {
    _ui.allocate_response(egui::Vec2::ZERO, egui::Sense::hover())
}

/// Links row (disabled / clean).
pub fn link_row(_app: &mut FotoditApp, _ui: &mut egui::Ui) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording_app() -> (FotoditApp, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::sync::{Arc, Mutex};
        let opened: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let rec = opened.clone();
        let services = crate::Services {
            open_url: Some(Box::new(move |u: &str| {
                rec.lock().unwrap().push(u.to_string());
                Ok(())
            })),
            ..Default::default()
        };
        (FotoditApp::new(fotodit_engine::Session::new(), services), opened)
    }

    #[test]
    fn links_open_through_the_platform_service() {
        // Issue #14: links must open in the browser through the `open_url` service rather than
        // the unreliable `ctx.open_url`. The branded upstream links are removed, so the service
        // path is checked with an arbitrary URL, and every remaining link command (none today).
        let (mut app, opened) = recording_app();
        let ctx = egui::Context::default();
        for (id, url) in COMMANDS {
            crate::menus::invoke(&mut app, &ctx, id, json!({})).unwrap();
            assert_eq!(opened.lock().unwrap().last().map(String::as_str), Some(*url), "{id}");
        }
        open(&mut app, &ctx, "https://example.com/docs");
        assert_eq!(opened.lock().unwrap().last().map(String::as_str), Some("https://example.com/docs"));
        assert_eq!(app.ui.status, "Opened https://example.com/docs");
    }

    #[test]
    fn help_menu_lists_updates_then_separator_then_system_info_and_about() {
        let app = FotoditApp::new(fotodit_engine::Session::new(), Default::default());
        let help: Vec<String> = crate::menus::menu_items(&app).into_iter().filter(|i| i.path == ["Help"]).map(|i| i.id).collect();
        assert_eq!(help, ["help.checkForUpdates", "---", "help.systemInfo", "help.about"]);
        for (id, _) in COMMANDS {
            assert!(crate::menus::is_live(id) && crate::menus::is_enabled(&app, id), "{id}");
        }
    }

    #[test]
    fn removed_links_are_empty_and_opening_them_does_nothing() {
        assert!(COMMANDS.is_empty());
        for url in [DISCORD, PLYRA_WEBSITE, APP_PAGE, TUTORIALS, GITHUB, ISSUES] {
            assert!(url.is_empty());
        }
        let (mut app, opened) = recording_app();
        let ctx = egui::Context::default();
        let status = app.ui.status.clone();
        assert_eq!(open(&mut app, &ctx, ""), json!({"url": ""}));
        assert!(opened.lock().unwrap().is_empty(), "an empty URL never reaches the service");
        assert_eq!(app.ui.status, status);
    }
}
