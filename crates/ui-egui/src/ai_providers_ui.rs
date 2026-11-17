//! The AI providers window: save, replace or remove the API key for each provider.
//!
//! Keys go straight from the text field to [`fotodit_keystore`] (Windows DPAPI, current user). They never pass
//! through preferences, the command bus, the journal or MCP, and a saved key is never shown again or exported.
//! "Test connection" is a function pair installed by the app (`set_checker`), so this crate does not depend on the
//! provider code. It runs only after the owner clicks and confirms; it reads the saved key itself, so the key does not
//! pass through this window. Providers without a documented validation call show why and send nothing. Saving a key
//! never turns a provider on.

use fotodit_keystore::{KeyStore, PROVIDERS, Provider, Sealer, Source};

use std::sync::OnceLock;

use crate::FotoditApp;

/// The key check supplied by the host app.
#[derive(Clone, Copy)]
pub struct Checker {
    /// `Ok` when a check exists for the provider slot, else the reason there is none.
    pub availability: fn(&str) -> Result<(), &'static str>,
    /// Sends the one validation request with the saved key and returns the message to show. Only on a click.
    pub run: fn(&str) -> String,
}

static CHECKER: OnceLock<Checker> = OnceLock::new();

/// Installs the key check. Call once at start-up.
pub fn set_checker(checker: Checker) {
    let _ = CHECKER.set(checker);
}

/// What the window knows about one provider, without any key material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowStatus {
    pub source: Result<Source, String>,
    /// The legacy environment variable for this provider is set (its value is never read here).
    pub env_present: bool,
}

#[derive(Default)]
struct Row {
    input: String,
    notice: Option<String>,
    /// The owner pressed Test connection and has not yet confirmed or cancelled.
    confirm_test: bool,
}

#[derive(Default)]
pub struct State {
    open: bool,
    rows: Vec<Row>,
    /// Provider statuses, read from the key file when the window opens and after each action, not every frame.
    cache: Option<Vec<RowStatus>>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State").field("open", &self.open).finish_non_exhaustive()
    }
}

pub fn open(app: &mut FotoditApp) {
    app.ai_providers.open = true;
    app.ai_providers.cache = None;
}

fn wipe(s: &mut String) {
    // Overwrites this String's buffer. The text widget keeps its own copies while editing; those are not covered.
    fotodit_keystore::wipe(s);
}

/// Status of every provider in [`PROVIDERS`] order.
pub fn statuses<S: Sealer>(store: &KeyStore<S>) -> Vec<RowStatus> {
    PROVIDERS
        .iter()
        .map(|p| RowStatus {
            source: store.source(p.slot).map_err(|e| e.to_string()),
            env_present: std::env::var_os(p.env_var).is_some_and(|v| !v.is_empty()),
        })
        .collect()
}

/// Saves the typed key and clears the field either way. Returns a message for the user.
pub fn save<S: Sealer>(store: &KeyStore<S>, provider: &Provider, input: &mut String) -> Result<(), String> {
    let result = store.set(provider.slot, input).map_err(|e| e.to_string());
    wipe(input);
    result
}

/// Removes the saved key and records the removal: no other credential source stands in for it afterwards.
pub fn remove<S: Sealer>(store: &KeyStore<S>, provider: &Provider) -> Result<(), String> {
    store.remove(provider.slot).map_err(|e| e.to_string())
}

/// Records the choice to use the environment variable or `.env` file for this provider.
pub fn use_environment<S: Sealer>(store: &KeyStore<S>, provider: &Provider) -> Result<(), String> {
    store.use_external(provider.slot).map_err(|e| e.to_string())
}

/// Clears the choice (and any removal marker): no key is used until a new choice is made.
pub fn clear_choice<S: Sealer>(store: &KeyStore<S>, provider: &Provider) -> Result<(), String> {
    store.reset(provider.slot).map_err(|e| e.to_string())
}

/// One line describing where the provider's key comes from.
pub fn source_text(status: &RowStatus) -> String {
    match &status.source {
        Ok(Source::Saved) if status.env_present => tl!("Using the key saved in Fotodit. The environment variable for this provider is ignored.").to_string(),
        Ok(Source::Saved) => tl!("Using the key saved in Fotodit.").to_string(),
        Ok(Source::Disabled) => tl!("Removed. No key is used for this provider.").to_string(),
        Ok(Source::External) if status.env_present => tl!("Using the environment variable for this provider (your choice).").to_string(),
        Ok(Source::External) => tl!("Using the environment variable or a .env file for this provider (your choice). The environment variable is not set.").to_string(),
        Ok(Source::Unconfigured) if status.env_present => tl!("No key is used. An environment variable for this provider is set. Choose below to use it.").to_string(),
        Ok(Source::Unconfigured) => tl!("No key is used. Paste a key below, or choose to use an environment variable or .env file.").to_string(),
        Err(_) => tl!("The saved key file could not be read. No key is used until you replace or remove it.").to_string(),
    }
}

const LIMITS: &str = "Saved keys are encrypted by Windows (DPAPI) for your Windows account, which under normal Windows use keeps other \
Windows users and other computers from reading them. Programs running as you can still ask Windows to decrypt them, as Fotodit does, so this does not \
protect against malware on your account. Keys are never shown again, exported, or written to preferences or project files. An environment \
variable or .env file is not encrypted.";

pub fn show(app: &mut FotoditApp, ctx: &egui::Context) {
    if !app.ai_providers.open {
        return;
    }
    let st = &mut app.ai_providers;
    if st.rows.len() != PROVIDERS.len() {
        st.rows = PROVIDERS.iter().map(|_| Row::default()).collect();
    }
    let store = fotodit_keystore::default_store();
    if st.cache.is_none() {
        st.cache = store.as_ref().map(statuses);
    }
    let cached = st.cache.clone();
    let mut dirty = false;
    let mut open = true;
    egui::Window::new(tl!("AI providers"))
        .id(egui::Id::unique("ai-providers-window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(520.0)
        .show(ctx, |ui| {
            if !fotodit_keystore::supported() {
                ui.colored_label(ui.visuals().warn_fg_color, tl!("Saving keys is not supported on this platform yet. Nothing is stored."));
            }
            ui.label(tl!(LIMITS));
            ui.add_space(6.0);
            let statuses = cached.clone();
            for (i, provider) in PROVIDERS.iter().enumerate() {
                ui.separator();
                ui.strong(provider.label);
                let status = statuses.as_ref().map(|s| s[i].clone());
                match &status {
                    Some(s) => {
                        ui.label(source_text(s));
                    }
                    None => {
                        ui.label(tl!("No per-user data folder is available. Nothing can be saved."));
                    }
                }
                let saved = matches!(status.as_ref().map(|s| &s.source), Some(Ok(Source::Saved)));
                let row = &mut st.rows[i];
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut row.input).password(true).desired_width(280.0).hint_text(tl!("Paste API key")));
                    let can_save = store.is_some() && fotodit_keystore::supported() && !row.input.trim().is_empty();
                    let label = if saved { tl!("Replace") } else { tl!("Save") };
                    if ui.add_enabled(can_save, egui::Button::new(label)).clicked()
                        && let Some(store) = &store
                    {
                        dirty = true;
                        row.notice = Some(match save(store, provider, &mut row.input) {
                            Ok(()) => tl!("Saved.").to_string(),
                            Err(e) => e,
                        });
                    }
                    if ui.add_enabled(saved && store.is_some(), egui::Button::new(tl!("Remove"))).clicked()
                        && let Some(store) = &store
                    {
                        dirty = true;
                        row.notice = Some(match remove(store, provider) {
                            Ok(()) => tl!("Removed.").to_string(),
                            Err(e) => e,
                        });
                    }
                });
                let source = status.as_ref().and_then(|s| s.source.as_ref().ok().copied());
                if matches!(source, Some(Source::Unconfigured | Source::Disabled))
                    && ui.small_button(tl!("Use the environment variable or .env file")).clicked()
                    && let Some(store) = &store
                {
                    dirty = true;
                        row.notice = Some(match use_environment(store, provider) {
                        Ok(()) => tl!("Using the environment variable or .env file.").to_string(),
                        Err(e) => e,
                    });
                }
                if matches!(source, Some(Source::External | Source::Disabled))
                    && ui.small_button(tl!("Clear my choice (use no key)")).clicked()
                    && let Some(store) = &store
                {
                    dirty = true;
                        row.notice = Some(match clear_choice(store, provider) {
                        Ok(()) => tl!("Choice cleared. No key is used.").to_string(),
                        Err(e) => e,
                    });
                }
                if saved {
                    match CHECKER.get() {
                        None => {
                            ui.weak(tl!("Test connection is not connected in this build."));
                        }
                        Some(c) => match (c.availability)(provider.slot) {
                            Err(reason) => {
                                ui.weak(tl!(reason));
                            }
                            Ok(()) if row.confirm_test => {
                                ui.label(tl!(
                                    "This sends one request to the provider, using your saved key. It does not generate anything, but the provider decides whether it counts against your account."
                                ));
                                ui.horizontal(|ui| {
                                    if ui.button(tl!("Send request")).clicked() {
                                        row.notice = Some((c.run)(provider.slot));
                                        row.confirm_test = false;
                                    }
                                    if ui.button(tl!("Cancel")).clicked() {
                                        row.confirm_test = false;
                                    }
                                });
                            }
                            Ok(()) => {
                                if ui.small_button(tl!("Test connection")).clicked() {
                                    row.confirm_test = true;
                                }
                            }
                        },
                    }
                }
                if let Some(n) = &row.notice {
                    ui.label(n);
                }
            }
            ui.add_space(6.0);
            ui.label(tl!("Fotodit never sends a request to check a key on its own. Saving a key does not turn a provider on."));
        });
    if dirty {
        st.cache = None;
    }
    if !open {
        for row in &mut st.rows {
            wipe(&mut row.input);
            row.notice = None;
            row.confirm_test = false;
        }
        st.open = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fotodit_keystore::testing::MockSealer;

    fn store(tag: &str) -> (KeyStore<MockSealer>, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("fotodit-ai-providers-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (KeyStore::new(dir.join("keys.json"), MockSealer), dir)
    }

    #[test]
    fn save_clears_the_field_and_updates_status() {
        let (s, dir) = store("save");
        let mut input = "test-key-123".to_string();
        save(&s, &PROVIDERS[0], &mut input).unwrap();
        assert!(input.is_empty());
        assert_eq!(statuses(&s)[0].source, Ok(Source::Saved));
        assert_eq!(statuses(&s)[1].source, Ok(Source::Unconfigured));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_failed_save_still_clears_the_field() {
        let (s, dir) = store("fail");
        let mut input = "bad\nkey".to_string();
        assert!(save(&s, &PROVIDERS[0], &mut input).is_err());
        assert!(input.is_empty());
        assert_eq!(statuses(&s)[0].source, Ok(Source::Unconfigured));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn remove_is_remembered_and_clearing_the_choice_forgets_it() {
        let (s, dir) = store("remove");
        let mut input = "test-key-123".to_string();
        save(&s, &PROVIDERS[2], &mut input).unwrap();
        remove(&s, &PROVIDERS[2]).unwrap();
        assert_eq!(statuses(&s)[2].source, Ok(Source::Disabled));
        use_environment(&s, &PROVIDERS[2]).unwrap();
        assert_eq!(statuses(&s)[2].source, Ok(Source::External));
        clear_choice(&s, &PROVIDERS[2]).unwrap();
        assert_eq!(statuses(&s)[2].source, Ok(Source::Unconfigured));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn status_text_names_the_source_without_any_key() {
        let t = source_text(&RowStatus { source: Ok(Source::Saved), env_present: true });
        assert!(t.contains("ignored"));
        let t = source_text(&RowStatus { source: Ok(Source::Disabled), env_present: true });
        assert!(t.contains("Removed"));
        let t = source_text(&RowStatus { source: Ok(Source::Unconfigured), env_present: false });
        assert!(t.contains("No key is used"));
        let t = source_text(&RowStatus { source: Err("corrupt".into()), env_present: true });
        assert!(t.contains("could not be read"));
    }

    #[test]
    fn state_debug_hides_typed_input() {
        let mut st = State::default();
        st.rows.push(Row { input: "secret-typed".into(), notice: None, confirm_test: false });
        assert!(!format!("{st:?}").contains("secret-typed"));
    }
}
