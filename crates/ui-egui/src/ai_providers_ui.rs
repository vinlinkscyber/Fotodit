//! The AI providers window: save, replace or remove the API key for each provider.
//!
//! Keys go straight from the text field to [`fotodit_keystore`] (Windows DPAPI, current user). They never pass
//! through preferences, the command bus, the journal or MCP, and a saved key is never shown again or exported.
//! "Test connection" is a function pair installed by the app (`set_checker`), so this crate does not depend on the
//! provider code. It runs only after the owner clicks and confirms; it reads the saved key itself, so the key does not
//! pass through this window. Providers without a documented validation call show why and send nothing. Saving a key
//! enables supported chat providers without sending a request.

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
static EDITING_IDENTITY: OnceLock<fn(&str) -> Result<[u8; 32], String>> = OnceLock::new();

/// Local credential/source identity only. Never sends a network request.
pub fn set_editing_identity(check: fn(&str) -> Result<[u8; 32], String>) {
    let _ = EDITING_IDENTITY.set(check);
}

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
    editing: EditingConsent,
}

/// Result checks one image-aware goal may make after its planning request, so
/// one goal costs at most 1 + VISUAL_MAX_CHECKS provider requests.
pub const VISUAL_MAX_CHECKS: u32 = 2;

/// Editing consent. The owner's rule: saving a credential enables its provider directly, and the
/// choice persists in the key store so the saved key keeps working across restarts. No request is
/// sent until a chat Send. Only Gemini, Cohere and Lajiwa can drive chat editing.
#[derive(Default, Debug)]
pub struct EditingConsent {
    active: Option<&'static str>,
    pending: Option<&'static str>,
    identity: Option<[u8; 32]>,
    generation: u64,
    /// Image-aware mode: a downscaled PNG of the document rides with each
    /// request. Only Gemini's wired transport can carry it, so this is refused
    /// for every other slot and cleared by any consent change.
    visual: bool,
}

impl EditingConsent {
    pub fn active(&self) -> Option<&'static str> { self.active }
    pub fn request(&mut self, slot: &'static str) {
        self.disable();
        if matches!(slot, "gemini" | "cohere" | "lajiwa") { self.pending = Some(slot); }
    }
    pub fn confirm_identity(&mut self, identity: [u8; 32]) { self.active = self.pending.take(); self.identity = self.active.map(|_| identity); }
    /// Enables editing for `slot` directly (save-to-enable and restart restore): no per-session
    /// confirmation step. Refused for providers that cannot drive chat editing.
    pub fn activate(&mut self, slot: &'static str, identity: [u8; 32]) {
        self.disable();
        if matches!(slot, "gemini" | "cohere" | "lajiwa") {
            self.active = Some(slot);
            self.identity = Some(identity);
        }
    }
    pub fn binding(&self) -> (Option<&'static str>, Option<[u8; 32]>, u64) { (self.active, self.identity, self.generation) }
    pub fn disable(&mut self) { self.active = None; self.pending = None; self.identity = None; self.visual = false; self.generation = self.generation.wrapping_add(1); }
    /// Image-aware mode sticks only while Gemini editing is active.
    pub fn set_visual(&mut self, on: bool) { self.visual = on && self.active == Some("gemini"); }
    pub fn visual_active(&self) -> bool { self.visual && self.active == Some("gemini") }
}

impl State {
    pub fn editing_consent(&self) -> &EditingConsent { &self.editing }
    pub fn editing_consent_mut(&mut self) -> &mut EditingConsent { &mut self.editing }

    /// Restores the persisted editing-enabled provider: a saved key keeps working across restarts
    /// without re-enabling. Any failure - no store, no stored choice, missing credential, no
    /// identity check - leaves consent off; per-send validation still re-checks the credential.
    fn restored_with<S: Sealer>(store: Option<&KeyStore<S>>, identity: Option<[u8; 32]>) -> Self {
        let mut st = State::default();
        let (Some(store), Some(identity)) = (store, identity) else { return st; };
        let Ok(Some(slot)) = store.editing() else { return st; };
        let Some(provider) = PROVIDERS.iter().find(|p| p.slot == slot.as_str()) else { return st; };
        if !matches!(provider.slot, "gemini" | "cohere" | "lajiwa") { return st; }
        // The persisted choice stands only while its credential does.
        if !matches!(store.source(provider.slot), Ok(Source::Saved | Source::External)) { return st; }
        st.editing.activate(provider.slot, identity);
        st
    }

    /// Restores consent from the default key store with the installed identity check.
    pub fn restored() -> Self {
        let store = fotodit_keystore::default_store();
        let identity = store
            .as_ref()
            .and_then(|s| s.editing().ok().flatten())
            .and_then(|slot| EDITING_IDENTITY.get().and_then(|check| check(&slot).ok()));
        Self::restored_with(store.as_ref(), identity)
    }
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
                let mut saved_to_enable = false;
                let mut removed_ok = false;
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut row.input).password(true).desired_width(280.0).hint_text(tl!("Paste API key")));
                    let can_save = store.is_some() && fotodit_keystore::supported() && !row.input.trim().is_empty();
                    let label = if saved { tl!("Replace") } else { tl!("Save") };
                    if ui.add_enabled(can_save, egui::Button::new(label)).clicked()
                        && let Some(store) = &store
                    {
                        dirty = true;
                        match save(store, provider, &mut row.input) {
                            Ok(()) => {
                                row.notice = Some(tl!("Saved.").to_string());
                                saved_to_enable = matches!(provider.slot, "gemini" | "cohere" | "lajiwa");
                            }
                            Err(e) => row.notice = Some(e),
                        }
                    }
                    if ui.add_enabled(saved && store.is_some(), egui::Button::new(tl!("Remove"))).clicked()
                        && let Some(store) = &store
                    {
                        dirty = true;
                        match remove(store, provider) {
                            Ok(()) => {
                                row.notice = Some(tl!("Removed.").to_string());
                                removed_ok = true;
                            }
                            Err(e) => row.notice = Some(e),
                        }
                    }
                });
                // The owner's rule: saving a key turns its provider on and remembers the choice, so
                // the key keeps working across restarts. No request is sent by saving.
                if saved_to_enable {
                    // Persist first: the host identity includes the protected store's bytes.
                    let persisted = store.as_ref().ok_or_else(|| "Protected provider store is unavailable".to_string())
                        .and_then(|s| s.set_editing(provider.slot).map_err(|e| e.to_string()));
                    match persisted.and_then(|()| EDITING_IDENTITY.get().ok_or_else(|| "Editing is not connected in this build".to_string()).and_then(|check| check(provider.slot))) {
                        Ok(identity) => {
                            st.editing.activate(provider.slot, identity);
                            row.notice = Some(tl!("Saved. Provider editing is on; no request is sent until you press Send in chat.").to_string());
                        }
                        Err(error) => { st.editing.disable(); row.notice = Some(format!("{} {error}", tl!("Saved."))); }
                    }
                }
                if removed_ok && st.editing.active == Some(provider.slot) {
                    // The key store already forgot the persisted choice with the key.
                    st.editing.disable();
                }
                let source = status.as_ref().and_then(|s| s.source.as_ref().ok().copied());
                if matches!(source, Some(Source::Unconfigured | Source::Disabled))
                    && ui.small_button(tl!("Use the environment variable or .env file")).clicked()
                    && let Some(store) = &store
                {
                    dirty = true;
                    match use_environment(store, provider) {
                        Ok(()) => {
                            row.notice = Some(tl!("Using the environment variable or .env file.").to_string());
                            if st.editing.active == Some(provider.slot) {
                                // The credential source changed; the saved-key enablement no longer matches it.
                                st.editing.disable();
                                if let Err(e) = store.clear_editing() {
                                    row.notice = Some(format!("Source changed, but the saved editing choice could not be cleared: {e}"));
                                }
                            }
                        }
                        Err(e) => row.notice = Some(e),
                    }
                }
                if matches!(source, Some(Source::External | Source::Disabled))
                    && ui.small_button(tl!("Clear my choice (use no key)")).clicked()
                    && let Some(store) = &store
                {
                    dirty = true;
                    match clear_choice(store, provider) {
                        Ok(()) => {
                            row.notice = Some(tl!("Choice cleared. No key is used.").to_string());
                            if st.editing.active == Some(provider.slot) {
                                st.editing.disable();
                            }
                        }
                        Err(e) => row.notice = Some(e),
                    }
                }
                if matches!(provider.slot, "gemini" | "cohere" | "lajiwa") {
                    if st.editing.active == Some(provider.slot) {
                        ui.label(tl!("Provider editing is on. It stays on across restarts until you disable it. Chat Send contacts this provider only."));
                        ui.label(match provider.slot {
                            "gemini" => "Destination: Google, https://generativelanguage.googleapis.com",
                            "cohere" => "Destination: Cohere, https://api.cohere.com",
                            "lajiwa" => "Destination: third-party Lajiwa worker, https://lajiwa.roydeburgs.workers.dev (not Google or Cohere). This worker receives your key and prompt.",
                            _ => "Unavailable",
                        });
                        ui.label(tl!("Chat Send sends your prompt, canvas size, active layer ID and layer names, text, IDs, kinds and bounds to this provider using your key. Context is capped at 64 layers, 128 characters per name and 512 characters per text layer; truncation is disclosed to the provider. Requests above 4000 prompt characters or 32KB context are refused. Editing is single-turn: earlier chat is not sent or remembered by the provider. Ambiguous follow-ups are refused; restate each edit completely. Requests may consume quota or cost money under your provider account; Fotodit cannot guarantee free use. No request is sent until you press Send. Other providers will not be used."));
                        if provider.slot == "gemini" {
                            let mut visual = st.editing.visual_active();
                            if ui.checkbox(&mut visual, tl!("Let Gemini see the canvas (sends a document image)")).changed() {
                                st.editing.set_visual(visual);
                            }
                            if st.editing.visual_active() {
                                ui.label(tl!("Image-aware editing: each chat Send makes 1 planning request, then up to 2 result-check requests, so at most 3 requests per goal. Every request includes a downscaled PNG image (1024px maximum edge) of your document with your prompt and layer metadata. The image leaves this device and goes to Google using your key. Each request may consume quota or cost money under your Google account; Fotodit cannot guarantee free use. Gemini reports whether the result looks right; that is its own judgment, not proof. A request already sent before disabling may still finish, including its document image; its result will not be applied and no further requests will follow."));
                            }
                        }
                        if ui.button(tl!("Disable provider editing")).clicked() {
                            st.editing.disable();
                            if let Some(store) = &store
                                && let Err(e) = store.clear_editing()
                            {
                                row.notice = Some(format!("Editing is off now, but the saved choice could not be cleared. It may return after restart: {e}"));
                            }
                            dirty = true;
                        }
                    } else if matches!(source, Some(Source::Saved | Source::External))
                        && ui.button(tl!("Enable provider editing")).clicked()
                    {
                        let persisted = store.as_ref().ok_or_else(|| "Protected provider store is unavailable".to_string())
                            .and_then(|s| s.set_editing(provider.slot).map_err(|e| e.to_string()));
                        match persisted.and_then(|()| EDITING_IDENTITY.get().ok_or_else(|| "Editing is not connected in this build".to_string()).and_then(|check| check(provider.slot))) {
                            Ok(identity) => { st.editing.activate(provider.slot, identity); dirty = true; }
                            Err(error) => { st.editing.disable(); row.notice = Some(error); }
                        }
                    }
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
            ui.label(tl!("Fotodit never sends a request to check a key on its own. Saving a key turns on AI editing for providers that support it; no request is sent until you press Send in chat."));
        });
    if dirty {
        // These mutations came from this settings window. Refresh the still-active binding
        // after all writes: the host fingerprints the store file, including other provider rows.
        // External file changes still revoke the binding on Send/poll as before.
        if let Some(slot) = st.editing.active {
            match EDITING_IDENTITY.get().ok_or_else(|| "Editing is not connected in this build".to_string()).and_then(|check| check(slot)) {
                Ok(identity) => st.editing.identity = Some(identity),
                Err(error) => {
                    st.editing.disable();
                    if let Some(i) = PROVIDERS.iter().position(|p| p.slot == slot) {
                        st.rows[i].notice = Some(error);
                    }
                }
            }
        }
        st.cache = None;
    }
    if !open {
        for row in &mut st.rows {
            wipe(&mut row.input);
            row.notice = None;
            row.confirm_test = false;
        }
        st.editing.pending = None;
        st.open = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fotodit_keystore::testing::MockSealer;

    // Keep these in the provider lifecycle suite used by the Windows qualification
    // workflow. Testing open() alone missed the disabled menu in the shipped build.
    #[test]
    fn provider_menu_is_enabled_and_opens_with_or_without_document() {
        for document in [false, true] {
            let mut app = FotoditApp::new(fotodit_engine::Session::new(), Default::default());
            if document {
                app.run("file.new", serde_json::json!({"width": 64, "height": 48})).unwrap();
            }
            let items = crate::menus::menu_items(&app);
            let item = items.iter().find(|item| item.id == "edit.aiProviders").expect("provider menu exists");
            assert_eq!(item.path, ["Edit"]);
            assert!(item.enabled, "provider configuration must not need an engine command or document");
            assert!(crate::menus::is_live(&item.id));
            assert!(!app.ai_providers.open);
            crate::menus::invoke(&mut app, &egui::Context::default(), &item.id, serde_json::json!({})).unwrap();
            assert!(app.ai_providers.open, "the admitted menu action opens provider settings");
            assert_eq!(app.ai_providers.editing_consent().active(), None, "opening settings grants no AI consent");
        }
    }

    #[test]
    fn provider_menu_pointer_click_opens_settings() {
        use egui_kittest::{Harness, kittest::Queryable};
        for document in [false, true] {
            let mut app = FotoditApp::new(fotodit_engine::Session::new(), Default::default());
            if document {
                app.run("file.new", serde_json::json!({"width": 64, "height": 48})).unwrap();
            }
            // A tall surface keeps the last Edit entry visible without introducing
            // scrolling into this menu-admission regression.
            let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 2400.0)).build_ui_state(
                |ui, app| { crate::menus::menu_bar(app, ui); },
                app,
            );
            FotoditApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
            harness.run_steps(3);
            harness.get_by_label("Edit").click();
            harness.run_steps(3);
            harness.get_by_label_contains("AI Providers").click();
            harness.run_steps(3);
            assert!(harness.state().ai_providers.open, "click opens provider settings, document={document}");
            assert_eq!(harness.state().ai_providers.editing_consent().active(), None);
        }
    }

    fn store(tag: &str) -> (KeyStore<MockSealer>, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("fotodit-ai-providers-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (KeyStore::new(dir.join("keys.json"), MockSealer), dir)
    }

    #[test]
    fn editing_activates_directly_and_resets() {
        let mut consent = EditingConsent::default();
        assert_eq!(consent.active(), None);
        consent.confirm_identity([7; 32]);
        assert_eq!(consent.active(), None, "identity alone does not enable");
        // Direct activation (save-to-enable and restart restore).
        consent.activate("gemini", [7; 32]);
        assert_eq!(consent.active(), Some("gemini"));
        assert_eq!(consent.binding().1, Some([7; 32]));
        consent.activate("cohere", [9; 32]);
        assert_eq!(consent.active(), Some("cohere"), "switching replaces the provider and its identity");
        assert_eq!(consent.binding().1, Some([9; 32]));
        consent.disable();
        assert_eq!(consent.active(), None);
        consent.activate("pollinations", [7; 32]);
        assert_eq!(consent.active(), None, "image-only provider cannot enable text editing");
        consent.activate("openrouter", [7; 32]);
        assert_eq!(consent.active(), None, "unavailable provider remains disabled");
        // The staged request/confirm path still works for callers that use it.
        consent.request("gemini");
        assert_eq!(consent.active(), None);
        consent.confirm_identity([7; 32]);
        assert_eq!(consent.active(), Some("gemini"));
        consent.request("cohere");
        assert_eq!(consent.active(), None, "request disables the previous provider before confirmation");
    }

    #[test]
    fn restore_keeps_a_saved_key_working_across_restarts() {
        let (s, dir) = store("restore");
        s.set("gemini", "k1").unwrap();
        s.set_editing("gemini").unwrap();
        // A fresh State over the same store is the restart: consent comes back on.
        let st = State::restored_with(Some(&s), Some([7; 32]));
        assert_eq!(st.editing_consent().active(), Some("gemini"));
        assert_eq!(st.editing_consent().binding().1, Some([7; 32]));
        // Explicitly disabling persists across a fresh State without deleting the key.
        s.clear_editing().unwrap();
        assert_eq!(State::restored_with(Some(&s), Some([7; 32])).editing_consent().active(), None);
        assert_eq!(s.source("gemini").unwrap(), Source::Saved);
        s.set_editing("gemini").unwrap();
        // Persisted unsupported provider choices do not enable editing.
        s.set("pollinations", "other").unwrap();
        s.set_editing("pollinations").unwrap();
        assert_eq!(State::restored_with(Some(&s), Some([7; 32])).editing_consent().active(), None);
        s.set_editing("gemini").unwrap();
        // Without the identity hook (feature off) consent stays off.
        assert_eq!(State::restored_with(Some(&s), None).editing_consent().active(), None);
        // No store or no stored choice: off.
        assert_eq!(State::restored_with::<MockSealer>(None, Some([7; 32])).editing_consent().active(), None);
        let (s2, dir2) = store("restore-empty");
        assert_eq!(State::restored_with(Some(&s2), Some([7; 32])).editing_consent().active(), None);
        // The stored choice without its credential does not enable.
        s.remove("gemini").unwrap();
        assert_eq!(State::restored_with(Some(&s), Some([7; 32])).editing_consent().active(), None);
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    #[test]
    fn visual_consent_requires_active_gemini_and_resets() {
        let mut consent = EditingConsent::default();
        consent.set_visual(true);
        assert!(!consent.visual_active(), "no image consent without active editing");
        consent.request("cohere");
        consent.confirm_identity([7; 32]);
        consent.set_visual(true);
        assert!(!consent.visual_active(), "image mode is wired for Gemini only");
        consent.request("gemini");
        assert!(!consent.visual_active(), "requesting a new consent clears image mode");
        consent.confirm_identity([7; 32]);
        consent.set_visual(true);
        assert!(consent.visual_active());
        consent.request("gemini");
        consent.confirm_identity([7; 32]);
        assert!(!consent.visual_active(), "re-enabling editing does not keep image mode");
        consent.set_visual(true);
        consent.disable();
        assert!(!consent.visual_active(), "disabling editing clears image mode");
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
