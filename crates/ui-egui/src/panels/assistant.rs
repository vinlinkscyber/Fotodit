//! The AI Assistant panel: chat-style natural-language editing. Layout follows the
//! approved reference design: header, Chat / Quick Actions tabs, quick-action chips,
//! scrolling transcript with bubbles, and the input row pinned at the bottom.

use super::{FotoditApp, Tokens};

/// Navy chrome palette from the approved design; shared with the home screen's AI rail.
pub(crate) const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(20, 25, 60);
pub(crate) const CHIP_BG: egui::Color32 = egui::Color32::from_rgb(35, 45, 95);
pub(crate) const CHIP_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(46, 58, 118);
const CHIP_BORDER: egui::Color32 = egui::Color32::from_rgb(70, 86, 165);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(37, 47, 98);
const BUBBLE_BORDER: egui::Color32 = egui::Color32::from_rgb(66, 80, 155);
const COMPOSER_BG: egui::Color32 = egui::Color32::from_rgb(46, 58, 118);
pub(crate) const ACCENT: egui::Color32 = egui::Color32::from_rgb(56, 108, 250);
/// Periwinkle icon tint on the navy chrome.
const ICON_BLUE: egui::Color32 = egui::Color32::from_rgb(140, 158, 255);
/// Purple of the assistant sparkle glyph.
const SPARKLE_PURPLE: egui::Color32 = egui::Color32::from_rgb(178, 110, 255);
/// Muted label colour on the navy chrome.
pub(crate) const TEXT_SOFT: egui::Color32 = egui::Color32::from_rgb(150, 158, 195);
/// Body text colour on the navy chrome.
pub(crate) const TEXT_ON_NAVY: egui::Color32 = egui::Color32::from_rgb(225, 229, 245);
/// Gradient stops for the panel chrome, top to bottom, from the approved design.
const CHROME_TOP: egui::Color32 = egui::Color32::from_rgb(20, 25, 60);
const CHROME_MID: egui::Color32 = egui::Color32::from_rgb(28, 38, 118);
const CHROME_BOTTOM: egui::Color32 = egui::Color32::from_rgb(58, 50, 158);

/// Paint a soft vertical gradient over `rect` (thin strips; egui has no gradient fills),
/// plus a faint blue glow rising from the bottom-left corner, as in the reference design.
fn gradient_fill(ui: &egui::Ui, rect: egui::Rect) {
    let steps = 48;
    let h = rect.height() / steps as f32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        // Three stops: navy -> vivid blue -> violet at the bottom.
        let c = if t < 0.55 {
            lerp_color(CHROME_TOP, CHROME_MID, t / 0.55)
        } else {
            lerp_color(CHROME_MID, CHROME_BOTTOM, (t - 0.55) / 0.45)
        };
        let strip = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, h * i as f32), egui::vec2(rect.width(), h + 1.0));
        ui.painter().rect_filled(strip, 0.0, c);
    }
    // Fake glow: concentric translucent circles low on the left edge.
    let center = egui::pos2(rect.left() + 10.0, rect.bottom() - 30.0);
    for i in 0..7 {
        let r = 230.0 - i as f32 * 28.0;
        ui.painter().circle_filled(center, r, egui::Color32::from_rgba_unmultiplied(70, 100, 255, 7));
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    egui::Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

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
    pinned: bool,
}

impl Default for AssistantState {
    fn default() -> Self {
        Self { transcript: Vec::new(), draft: String::new(), tab: Tab::Chat, pinned: false }
    }
}

/// Canned one-tap edits from the approved design; each is a real natural-language prompt.
const QUICK_ACTIONS: &[(&str, &str)] = &[
    ("scissors", "Remove background"),
    ("sparkles", "Enhance image quality"),
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

/// Demo mode (FOTODIT_ASSISTANT_DEMO=1, used by the screenshot workflow): show the panel
/// full-height on the left, as in the reference design, so the layout is reviewable as
/// it will feel in use.
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
        // The demo overlay shows the empty state (greeting + chips) full-height on the
        // left, like the reference design; the docked panel keeps the seeded transcript
        // so one capture reviews both layouts.
        let demo_id = egui::Id::new("assistant-demo-state");
        let mut demo_state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(demo_id).unwrap_or_default());
        let screen = ui.ctx().screen_rect();
        let rect = egui::Rect::from_min_size(
            egui::pos2(screen.left() + 36.0, screen.top() + 26.0),
            egui::vec2(336.0, (screen.height() - 52.0).max(200.0)),
        );
        egui::Area::new(egui::Id::new("assistant-demo-area"))
            .order(egui::Order::Foreground)
            .fixed_pos(rect.min)
            .show(ui.ctx(), |ui| {
                ui.set_min_size(rect.size());
                ui.set_max_size(rect.size());
                panel_chrome(ui, &mut demo_state, None);
            });
        ui.ctx().data_mut(|d| d.insert_temp(demo_id, demo_state));
        ui.ctx().data_mut(|d| d.insert_temp(state_id(), state));
    }
}

/// Navy frame + header + tabs, with the active tab's body inside. In the dock, `app`
/// is present and the header's collapse control hides the group (reopen from the Window menu).
fn panel_chrome(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    let area = ui.available_rect_before_wrap();
    gradient_fill(ui, area);
    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            header(ui, state, app.as_mut().map(|a| &mut **a));
            ui.add_space(8.0);
            tabs(ui, state);
            ui.add_space(12.0);
            match state.tab {
                Tab::Chat => chat_body(ui, state),
                Tab::QuickActions => quick_actions_body(ui, state),
            }
        });
}

/// Sparkle glyph + title + pin/collapse controls, like the reference design's header.
fn header(ui: &mut egui::Ui, state: &mut AssistantState, app: Option<&mut FotoditApp>) {
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        let glyph = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(20.0, 20.0));
        crate::icons::paint(ui, glyph, "sparkles", 20.0, SPARKLE_PURPLE);
        ui.add_space(26.0);
        ui.label(egui::RichText::new(tl!("AI Assistant")).color(egui::Color32::WHITE).strong().size(15.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Collapse (hides the panel; reopen from the Window menu), then pin.
            let (chev_rect, chev) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
            let chev = chev.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Close panel"));
            if chev.hovered() {
                ui.painter().rect_filled(chev_rect, 6.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, chev_rect, "chevron-down", 15.0, TEXT_SOFT);
            let (pin_rect, pin) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
            let pin = pin.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Pin panel"));
            if pin.hovered() {
                ui.painter().rect_filled(pin_rect, 6.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, pin_rect, "pin", 15.0, if state.pinned { ICON_BLUE } else { TEXT_SOFT });
            if pin.clicked() {
                state.pinned = !state.pinned;
            }
            if chev.clicked() {
                if let Some(app) = app {
                    app.ui.panels.assistant = false;
                }
            }
        });
    });
}

/// Chat | Quick Actions switcher, active tab with a rounded accent underline pill.
fn tabs(ui: &mut egui::Ui, state: &mut AssistantState) {
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        tab_label(ui, state, Tab::Chat, &tl!("Chat"));
        ui.add_space(22.0);
        tab_label(ui, state, Tab::QuickActions, &tl!("Quick Actions"));
    });
}

fn tab_label(ui: &mut egui::Ui, state: &mut AssistantState, tab: Tab, label: &str) {
    let active = state.tab == tab;
    let color = if active { egui::Color32::WHITE } else { Tokens::get(ui.ctx()).text_dim };
    let response = ui.add(egui::Label::new(egui::RichText::new(label).color(color).size(14.0)).sense(egui::Sense::click()));
    if active {
        let rect = response.rect;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(egui::pos2(rect.min.x - 2.0, rect.max.y + 3.0), egui::vec2(rect.width() + 4.0, 3.5)),
            2.0,
            ACCENT,
        );
    }
    if response.clicked() {
        state.tab = tab;
    }
}

/// One-tap canned edits: translucent navy chip with a periwinkle line icon, as in the
/// reference design.
fn chip(ui: &mut egui::Ui, icon: &str, label: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 44.0), egui::Sense::click());
    let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if response.hovered() { CHIP_BG_HOVER } else { CHIP_BG };
    ui.painter().rect_filled(rect, 12.0, fill);
    ui.painter().rect_stroke(rect, 12.0, egui::Stroke::new(1.0, CHIP_BORDER), egui::StrokeKind::Inside);
    let icon_box = egui::Rect::from_center_size(egui::pos2(rect.left() + 26.0, rect.center().y), egui::vec2(20.0, 20.0));
    crate::icons::paint(ui, icon_box, icon, 18.0, ICON_BLUE);
    ui.painter().text(
        egui::pos2(rect.left() + 44.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(14.0),
        egui::Color32::from_rgb(218, 223, 245),
    );
    response
}

fn quick_actions_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    for &(icon, action) in QUICK_ACTIONS {
        if chip(ui, icon, action).clicked() {
            send(state, action.to_string());
        }
        ui.add_space(10.0);
    }
    if chip(ui, "ellipsis", &tl!("More options...")).clicked() {
        state.tab = Tab::Chat;
    }
}

/// The chat itself: scrolling transcript with the input row pinned below.
fn chat_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    let footer_height = 52.0;
    egui::ScrollArea::vertical()
        .id_salt("assistant-transcript")
        .auto_shrink([false, false])
        .max_height((ui.available_height() - footer_height).max(40.0))
        .show(ui, |ui| {
            if state.transcript.is_empty() {
                greeting(ui);
                ui.add_space(14.0);
                for &(icon, action) in QUICK_ACTIONS {
                    if chip(ui, icon, action).clicked() {
                        send(state, action.to_string());
                    }
                    ui.add_space(10.0);
                }
                if chip(ui, "ellipsis", &tl!("More options...")).clicked() {
                    state.tab = Tab::QuickActions;
                }
            }
            for exchange in &state.transcript {
                bubble(ui, true, &exchange.user);
                bubble(ui, false, &exchange.assistant);
            }
        });

    ui.add_space(10.0);
    input_row(ui, state);
}

/// The empty-state greeting bubble with the sparkle avatar, from the reference design.
fn greeting(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let glyph = egui::Rect::from_min_size(ui.cursor().min + egui::vec2(2.0, 4.0), egui::vec2(20.0, 20.0));
        crate::icons::paint(ui, glyph, "sparkles", 20.0, SPARKLE_PURPLE);
        ui.add_space(28.0);
        egui::Frame::NONE
            .fill(ASSISTANT_BUBBLE)
            .corner_radius(egui::CornerRadius::same(14))
            .stroke(egui::Stroke::new(1.0, BUBBLE_BORDER))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(tl!("Hello! How can I help you edit your image today?")).color(TEXT_ON_NAVY).size(14.0),
                ).wrap());
            });
    });
}

/// The input bar: one rounded container with a paperclip at the left and the accent send
/// button inside at the right, as in the reference design.
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState) {
    egui::Frame::NONE
        .fill(COMPOSER_BG)
        .corner_radius(egui::CornerRadius::same(16))
        .stroke(egui::Stroke::new(1.0, CHIP_BORDER))
        .inner_margin(egui::Margin::symmetric(8, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (attach_rect, attach) = ui.allocate_exact_size(egui::vec2(26.0, 30.0), egui::Sense::click());
                let attach = attach.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Attach an image (coming soon)"));
                crate::icons::paint(ui, attach_rect, "paperclip", 16.0, TEXT_SOFT);

                let width = ui.available_width() - 40.0;
                let edit = egui::TextEdit::singleline(&mut state.draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT).size(14.0))
                    .font(egui::FontId::proportional(14.0))
                    .desired_width(width.max(60.0))
                    .frame(egui::Frame::NONE);
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !state.draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(32.0, 30.0), egui::Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                ui.painter().rect_filled(send_rect, 10.0, if sendable { ACCENT } else { egui::Color32::from_rgb(58, 70, 130) });
                crate::icons::paint(ui, send_rect, "send", 14.0, egui::Color32::WHITE);
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
    let (fill, color) = if user { (ACCENT, egui::Color32::WHITE) } else { (ASSISTANT_BUBBLE, TEXT_ON_NAVY) };
    let layout = if user {
        egui::Layout::right_to_left(egui::Align::TOP)
    } else {
        egui::Layout::left_to_right(egui::Align::TOP)
    };
    ui.with_layout(layout, |ui| {
        egui::Frame::NONE
            .fill(fill)
            .corner_radius(egui::CornerRadius::same(12))
            .inner_margin(egui::Margin::symmetric(11, 8))
            .show(ui, |ui| {
                ui.set_max_width(width * 0.85);
                ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
            });
    });
    ui.add_space(4.0);
}
