//! The AI Assistant panel: chat-style natural-language editing. Layout follows the
//! approved reference design: header, Chat / Quick Actions tabs, quick-action chips,
//! scrolling transcript with bubbles, and the input row pinned at the bottom.

use super::{FotoditApp, Tokens};

/// Deep navy chrome from the approved design; shared with the home screen's AI rail.
pub(crate) const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(17, 21, 44);
pub(crate) const CHIP_BG: egui::Color32 = egui::Color32::from_rgb(28, 34, 66);
pub(crate) const CHIP_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(38, 46, 86);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(32, 39, 74);
pub(crate) const ACCENT: egui::Color32 = egui::Color32::from_rgb(63, 110, 246);
/// Muted label colour on the navy chrome.
pub(crate) const TEXT_SOFT: egui::Color32 = egui::Color32::from_rgb(160, 168, 200);
/// Gradient stops for the panel chrome, top to bottom, from the approved design.
const CHROME_TOP: egui::Color32 = egui::Color32::from_rgb(19, 24, 55);
const CHROME_BOTTOM: egui::Color32 = egui::Color32::from_rgb(13, 16, 38);

/// Paint a soft vertical gradient over `rect` (thin strips; egui has no gradient fills).
fn gradient_fill(ui: &egui::Ui, rect: egui::Rect, top: egui::Color32, bottom: egui::Color32) {
    let steps = 24;
    let h = rect.height() / steps as f32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let strip = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, h * i as f32), egui::vec2(rect.width(), h + 1.0));
        ui.painter().rect_filled(strip, 0.0, lerp_color(top, bottom, t));
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    egui::Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}
/// Body text colour on the navy chrome.
pub(crate) const TEXT_ON_NAVY: egui::Color32 = egui::Color32::from_rgb(225, 229, 245);

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
const QUICK_ACTIONS: &[(&str, &str)] = &[
    ("eraser-background", "Remove the background"),
    ("wand-sparkles", "Enhance image quality"),
    ("palette", "Adjust colors"),
    ("type", "Add text"),
];

fn state_id() -> egui::Id {
    egui::Id::new("assistant-panel-state")
}

/// Put a prompt into the panel's draft (used by the home screen's AI rail) and switch it
/// to the Chat tab, ready for the user to review and send.
pub(crate) fn seed_prompt(ctx: &egui::Context, prompt: &str) {
    ctx.data_mut(|d| {
        let mut state = d.get_temp::<AssistantState>(state_id()).unwrap_or_default();
        state.draft = prompt.to_string();
        state.tab = Tab::Chat;
        d.insert_temp(state_id(), state);
    });
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
        // The floating preview shows the empty state (greeting + chips); the docked
        // panel keeps the seeded transcript so one capture reviews both layouts.
        let demo_id = egui::Id::new("assistant-demo-state");
        let mut demo_state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(demo_id).unwrap_or_default());
        let mut open = true;
        egui::Window::new("AI Assistant")
            .default_pos([80.0, 90.0])
            .default_size([320.0, 640.0])
            .resizable(true)
            .collapsible(false)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                panel_chrome(ui, &mut demo_state, None);
            });
        ui.ctx().data_mut(|d| d.insert_temp(demo_id, demo_state));
        ui.ctx().data_mut(|d| d.insert_temp(state_id(), state));
    }
}

/// Navy frame + header + tabs, with the active tab's body inside. In the dock, `app`
/// is present and the header's close control hides the group (reopen from the Window menu).
fn panel_chrome(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    let area = ui.available_rect_before_wrap();
    gradient_fill(ui, area, CHROME_TOP, CHROME_BOTTOM);
    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(12, 10))
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
        let tile = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(24.0, 24.0));
        ui.painter().rect_filled(tile, 7.0, ACCENT);
        crate::icons::paint(ui, tile, "sparkles", 15.0, egui::Color32::WHITE);
        ui.add_space(30.0);
        ui.label(egui::RichText::new(tl!("AI Assistant")).color(egui::Color32::WHITE).strong().size(15.0));
        if let Some(app) = app {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (rect, close) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
                let close = close.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Close panel"));
                if close.hovered() {
                    ui.painter().rect_filled(rect, 6.0, CHIP_BG_HOVER);
                }
                crate::icons::paint(ui, rect, "x", 13.0, TEXT_SOFT);
                if close.clicked() {
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

/// One-tap canned edits: dark rounded chip with a line icon, as in the reference design.
fn chip(ui: &mut egui::Ui, icon: &str, label: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 38.0), egui::Sense::click());
    let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if response.hovered() { CHIP_BG_HOVER } else { CHIP_BG };
    ui.painter().rect_filled(rect, 9.0, fill);
    ui.painter().rect_stroke(rect, 9.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 55, 100)), egui::StrokeKind::Inside);
    let icon_box = egui::Rect::from_center_size(egui::pos2(rect.left() + 24.0, rect.center().y), egui::vec2(20.0, 20.0));
    crate::icons::paint(ui, icon_box, icon, 17.0, ACCENT);
    ui.painter().text(
        egui::pos2(rect.left() + 42.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        egui::Color32::from_rgb(214, 219, 240),
    );
    response
}

fn quick_actions_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    for &(icon, action) in QUICK_ACTIONS {
        if chip(ui, icon, action).clicked() {
            send(state, action.to_string());
        }
        ui.add_space(8.0);
    }
    ui.add_space(2.0);
    if chip(ui, "circle-dashed", &tl!("More options…")).clicked() {
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
                for &(icon, action) in QUICK_ACTIONS {
                    if chip(ui, icon, action).clicked() {
                        send(state, action.to_string());
                    }
                    ui.add_space(8.0);
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
        let tile = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(24.0, 24.0));
        ui.painter().rect_filled(tile, 12.0, ACCENT);
        crate::icons::paint(ui, tile, "sparkles", 14.0, egui::Color32::WHITE);
        ui.add_space(30.0);
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

/// The input bar: one rounded container with "+" at the left and the accent send button
/// inside at the right, as in the reference design.
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState) {
    egui::Frame::NONE
        .fill(CHIP_BG)
        .corner_radius(egui::CornerRadius::same(10))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 55, 100)))
        .inner_margin(egui::Margin::symmetric(6, 5))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (attach_rect, attach) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::click());
                let attach = attach.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Attach an image (coming soon)"));
                crate::icons::paint(ui, attach_rect, "paperclip", 15.0, TEXT_SOFT);

                let width = ui.available_width() - 38.0;
                let edit = egui::TextEdit::singleline(&mut state.draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT))
                    .desired_width(width.max(60.0))
                    .frame(egui::Frame::NONE);
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !state.draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), egui::Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                ui.painter().rect_filled(send_rect, 8.0, if sendable { ACCENT } else { egui::Color32::from_rgb(45, 55, 100) });
                crate::icons::paint(ui, send_rect, "send", 13.0, egui::Color32::WHITE);
                if (send_click.clicked() || enter) && sendable {
                    let asked = state.draft.trim().to_string();
                    state.draft.clear();
                    send(state, asked);
                    response.request_focus();
                }
            });
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
