//! The AI Assistant panel: chat-style natural-language editing. Layout follows the
//! approved reference design: header, Chat / Quick Actions tabs, quick-action chips,
//! scrolling transcript with bubbles, and the input row pinned at the bottom.

use super::{FotoditApp, Tokens};

/// Deep navy chrome from the approved design.
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(17, 21, 44);
const CHIP_BG: egui::Color32 = egui::Color32::from_rgb(28, 34, 66);
const CHIP_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(38, 46, 86);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(32, 39, 74);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(63, 110, 246);

/// One chat exchange: what the user asked, and what the app answered.
#[derive(Clone)]
struct Exchange {
    user: String,
    assistant: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Chat,
    QuickActions,
}

/// Panel state kept in egui memory for the session: the running transcript and the draft.
#[derive(Clone)]
struct AssistantState {
    transcript: Vec<Exchange>,
    draft: String,
    tab: Tab,
}

impl Default for AssistantState {
    fn default() -> Self {
        Self { transcript: Vec::new(), draft: String::new(), tab: Tab::Chat }
    }
}

/// Canned one-tap edits from the approved design; each is a real natural-language prompt.
const QUICK_ACTIONS: &[&str] = &[
    "Remove the background",
    "Enhance image quality",
    "Adjust colors",
    "Add text",
];

fn state_id() -> egui::Id {
    egui::Id::new("assistant-panel-state")
}

/// Demo mode (FOTODIT_ASSISTANT_DEMO=1, used by the screenshot workflow): seed a sample
/// conversation and float a large preview window so the layout is reviewable on its own.
fn demo_enabled() -> bool {
    std::env::var_os("FOTODIT_ASSISTANT_DEMO").is_some()
}

fn seed_demo(state: &mut AssistantState) {
    if state.transcript.is_empty() {
        state.transcript.push(Exchange {
            user: "make it brighter and remove the background".into(),
            assistant: "Done - exposure lifted a touch and the subject is isolated on transparency. Say \"undo\" if that's not what you meant.".into(),
        });
        state.transcript.push(Exchange {
            user: "actually make it a bit warmer too".into(),
            assistant: "Warmth nudged up on the same edit. Anything else?".into(),
        });
    }
}

pub(crate) fn assistant_panel(app: &mut FotoditApp, ui: &mut egui::Ui) {
    let demo = demo_enabled();
    let mut state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(state_id()).unwrap_or_default());
    if demo {
        seed_demo(&mut state);
    }

    panel_chrome(ui, &mut state, Some(app));

    ui.ctx().data_mut(|d| d.insert_temp(state_id(), state.clone()));

    if demo {
        let mut state = state;
        let mut open = true;
        egui::Window::new("AI Assistant")
            .default_pos([80.0, 90.0])
            .default_size([320.0, 640.0])
            .resizable(true)
            .collapsible(false)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                panel_chrome(ui, &mut state, None);
            });
        ui.ctx().data_mut(|d| d.insert_temp(state_id(), state));
    }
}

/// Navy frame + header + tabs, with the active tab's body inside. In the dock, `app`
/// is present and the header's close control hides the group (reopen from the Window menu).
fn panel_chrome(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    egui::Frame::NONE
        .fill(PANEL_BG)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            header(ui, app.as_mut().map(|a| &mut **a));
            ui.add_space(6.0);
            tabs(ui, state);
            ui.add_space(8.0);
            match state.tab {
                Tab::Chat => chat_body(ui, state),
                Tab::QuickActions => quick_actions_body(ui, state),
            }
        });
}

/// Monogram + title + close control, like the reference design's header.
fn header(ui: &mut egui::Ui, app: Option<&mut FotoditApp>) {
    ui.horizontal(|ui| {
        egui::Frame::NONE
            .fill(ACCENT)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::symmetric(7, 3))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("AI").color(egui::Color32::WHITE).strong().size(12.0));
            });
        ui.label(egui::RichText::new(tl!("AI Assistant")).color(egui::Color32::WHITE).strong().size(15.0));
        if let Some(app) = app {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let close = egui::Button::new(egui::RichText::new("×").size(16.0).color(egui::Color32::from_rgb(160, 168, 200)))
                    .fill(egui::Color32::TRANSPARENT);
                if ui.add(close).on_hover_text(tl!("Close panel")).clicked() {
                    app.ui.panels.assistant = false;
                }
            });
        }
    });
}

/// Chat | Quick Actions switcher, active tab underlined in the accent colour.
fn tabs(ui: &mut egui::Ui, state: &mut AssistantState) {
    ui.horizontal(|ui| {
        tab_label(ui, state, Tab::Chat, &tl!("Chat"));
        ui.add_space(18.0);
        tab_label(ui, state, Tab::QuickActions, &tl!("Quick Actions"));
    });
}

fn tab_label(ui: &mut egui::Ui, state: &mut AssistantState, tab: Tab, label: &str) {
    let active = state.tab == tab;
    let color = if active { egui::Color32::WHITE } else { Tokens::get(ui.ctx()).text_dim };
    let response = ui.add(egui::Label::new(egui::RichText::new(label).color(color).size(13.0)).sense(egui::Sense::click()));
    if active {
        let rect = response.rect;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(egui::pos2(rect.min.x, rect.max.y + 2.0), egui::vec2(rect.width(), 2.0)),
            1.0,
            ACCENT,
        );
    }
    if response.clicked() {
        state.tab = tab;
    }
}

/// One-tap canned edits, styled as the dark rounded chips in the reference design.
fn chip(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let button = egui::Button::new(egui::RichText::new(label).color(egui::Color32::from_rgb(210, 216, 240)).size(13.0))
        .fill(CHIP_BG)
        .corner_radius(egui::CornerRadius::same(8))
        .min_size(egui::vec2(ui.available_width(), 34.0));
    let _ = CHIP_BG_HOVER;
    ui.add(button)
}

fn quick_actions_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    for &action in QUICK_ACTIONS {
        if chip(ui, action).clicked() {
            send(state, action.to_string());
        }
        ui.add_space(6.0);
    }
    ui.add_space(2.0);
    if chip(ui, &tl!("More options…")).clicked() {
        state.tab = Tab::Chat;
    }
}

/// The chat itself: scrolling transcript with the input row pinned below.
fn chat_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    let footer_height = 44.0;
    egui::ScrollArea::vertical()
        .id_salt("assistant-transcript")
        .auto_shrink([false, false])
        .max_height((ui.available_height() - footer_height).max(40.0))
        .show(ui, |ui| {
            if state.transcript.is_empty() {
                greeting(ui);
                ui.add_space(10.0);
                for &action in QUICK_ACTIONS {
                    if chip(ui, action).clicked() {
                        send(state, action.to_string());
                    }
                    ui.add_space(6.0);
                }
            }
            for exchange in &state.transcript {
                bubble(ui, true, &exchange.user);
                bubble(ui, false, &exchange.assistant);
            }
        });

    ui.add_space(8.0);
    input_row(ui, state);
}

/// The empty-state greeting bubble with the AI monogram avatar, from the reference design.
fn greeting(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        egui::Frame::NONE
            .fill(ACCENT)
            .corner_radius(egui::CornerRadius::same(11))
            .inner_margin(egui::Margin::symmetric(6, 4))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("AI").color(egui::Color32::WHITE).strong().size(11.0));
            });
        egui::Frame::NONE
            .fill(ASSISTANT_BUBBLE)
            .corner_radius(egui::CornerRadius::same(10))
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(tl!("Hello! How can I help you edit your image today?")).color(egui::Color32::from_rgb(225, 229, 245)),
                ).wrap());
            });
    });
}

/// Draft field with an attach-style "+" affordance and the accent send button.
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState) {
    ui.horizontal(|ui| {
        let attach = egui::Button::new(egui::RichText::new("+").size(16.0).color(egui::Color32::from_rgb(160, 168, 200)))
            .fill(CHIP_BG)
            .corner_radius(egui::CornerRadius::same(8))
            .min_size(egui::vec2(30.0, 30.0));
        ui.add(attach).on_hover_text(tl!("Attach an image (coming soon)"));

        let width = ui.available_width() - 40.0;
        let edit = egui::TextEdit::singleline(&mut state.draft)
            .hint_text(tl!("Describe what you want..."))
            .desired_width(width.max(60.0));
        let response = ui.add(edit);
        let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let sendable = !state.draft.trim().is_empty();
        let send_button = egui::Button::new(egui::RichText::new("→").size(16.0).color(egui::Color32::WHITE))
            .fill(ACCENT)
            .corner_radius(egui::CornerRadius::same(8))
            .min_size(egui::vec2(32.0, 30.0));
        if (ui.add_enabled(sendable, send_button).clicked() || enter) && sendable {
            let asked = state.draft.trim().to_string();
            state.draft.clear();
            send(state, asked);
            response.request_focus();
        }
    });
}

/// Record an exchange. The engine hookup replaces the placeholder reply when wired.
fn send(state: &mut AssistantState, prompt: String) {
    state.tab = Tab::Chat;
    state.transcript.push(Exchange {
        user: prompt,
        assistant: tl!("(Preview) the editing engine plugs in here.").to_string(),
    });
}

/// One chat bubble: the user's right-aligned in the accent colour, the assistant's on navy.
fn bubble(ui: &mut egui::Ui, user: bool, text: &str) {
    let width = ui.available_width();
    let (fill, color) = if user { (ACCENT, egui::Color32::WHITE) } else { (ASSISTANT_BUBBLE, egui::Color32::from_rgb(225, 229, 245)) };
    let layout = if user {
        egui::Layout::right_to_left(egui::Align::TOP)
    } else {
        egui::Layout::left_to_right(egui::Align::TOP)
    };
    ui.with_layout(layout, |ui| {
        egui::Frame::NONE
            .fill(fill)
            .corner_radius(egui::CornerRadius::same(10))
            .inner_margin(egui::Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.set_max_width(width * 0.85);
                ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
            });
    });
    ui.add_space(4.0);
}
