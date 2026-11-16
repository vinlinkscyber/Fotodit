//! The AI Assistant panel: chat-style natural-language editing. Layout follows the
//! approved reference design: rounded navy enclosure with a fine violet rim, localized
//! purple and blue lighting, header, Chat / Quick Actions tabs, quick-action pills,
//! scrolling transcript with bubbles, and the composer pinned at the bottom.

use super::{FotoditApp, Tokens};

/// Navy chrome palette from the approved design; shared with the home screen's AI rail.
pub(crate) const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(20, 25, 60);
pub(crate) const CHIP_BG: egui::Color32 = egui::Color32::from_rgb(37, 47, 100);
pub(crate) const CHIP_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(48, 60, 122);
const CHIP_BORDER: egui::Color32 = egui::Color32::from_rgb(74, 88, 165);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(37, 47, 98);
const BUBBLE_BORDER: egui::Color32 = egui::Color32::from_rgb(66, 80, 155);
const COMPOSER_BG: egui::Color32 = egui::Color32::from_rgb(46, 58, 118);
const COMPOSER_BORDER: egui::Color32 = egui::Color32::from_rgb(100, 90, 190);
const RIM: egui::Color32 = egui::Color32::from_rgb(112, 96, 220);
pub(crate) const ACCENT: egui::Color32 = egui::Color32::from_rgb(60, 118, 255);
/// Lavender icon tint on the navy chrome.
const ICON_BLUE: egui::Color32 = egui::Color32::from_rgb(165, 175, 255);
/// Purple of the assistant sparkle glyph.
const SPARKLE_PURPLE: egui::Color32 = egui::Color32::from_rgb(186, 116, 255);
/// Muted label colour on the navy chrome.
pub(crate) const TEXT_SOFT: egui::Color32 = egui::Color32::from_rgb(150, 158, 195);
/// Body text colour on the navy chrome.
pub(crate) const TEXT_ON_NAVY: egui::Color32 = egui::Color32::from_rgb(228, 232, 248);
/// Gradient stops for the panel chrome, top to bottom, from the approved design.
const CHROME_TOP: egui::Color32 = egui::Color32::from_rgb(19, 24, 58);
const CHROME_MID: egui::Color32 = egui::Color32::from_rgb(27, 37, 112);
const CHROME_BOTTOM: egui::Color32 = egui::Color32::from_rgb(56, 48, 152);
/// Corner radius of the panel enclosure.
const RIM_RADIUS: f32 = 18.0;

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    egui::Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

fn gradient_at(y: f32, rect: egui::Rect) -> egui::Color32 {
    let t = ((y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
    if t < 0.55 { lerp_color(CHROME_TOP, CHROME_MID, t / 0.55) } else { lerp_color(CHROME_MID, CHROME_BOTTOM, (t - 0.55) / 0.45) }
}

/// Fill a rounded rect with a vertical gradient. egui has no gradient fills, so the shape
/// is tessellated as a triangle-fan mesh with per-vertex colours (which do interpolate).
fn rounded_gradient(painter: &egui::Painter, rect: egui::Rect, radius: f32) {
    let mut mesh = egui::epaint::Mesh::default();
    let center = rect.center();
    let center_idx = mesh.vertices.len() as u32;
    mesh.vertices.push(egui::epaint::Vertex { pos: center, uv: egui::epaint::WHITE_UV, color: gradient_at(center.y, rect) });
    let r = radius.min(rect.width() / 2.0).min(rect.height() / 2.0);
    let mut pts: Vec<egui::Pos2> = vec![
        egui::pos2(rect.left() + r, rect.top()),
        egui::pos2(rect.right() - r, rect.top()),
    ];
    let corners = [
        (egui::pos2(rect.right() - r, rect.top() + r), -90.0_f32, 0.0_f32),
        (egui::pos2(rect.right() - r, rect.bottom() - r), 0.0, 90.0),
        (egui::pos2(rect.left() + r, rect.bottom() - r), 90.0, 180.0),
        (egui::pos2(rect.left() + r, rect.top() + r), 180.0, 270.0),
    ];
    const SEGMENTS: usize = 6;
    for (c, a0, a1) in corners {
        for i in 0..=SEGMENTS {
            let a = (a0 + (a1 - a0) * i as f32 / SEGMENTS as f32).to_radians();
            pts.push(c + egui::vec2(r * a.cos(), r * a.sin()));
        }
    }
    let base = mesh.vertices.len() as u32;
    for p in &pts {
        mesh.vertices.push(egui::epaint::Vertex { pos: *p, uv: egui::epaint::WHITE_UV, color: gradient_at(p.y, rect) });
    }
    for i in 0..pts.len() {
        mesh.indices.push(center_idx);
        mesh.indices.push(base + i as u32);
        mesh.indices.push(base + (i + 1) % pts.len() as u32);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The enclosure: rounded gradient, localized purple (title) and blue (lower-right)
/// lighting, a fine violet rim, and a faint outer halo. Localized light, per the reference.
fn panel_surface(ui: &egui::Ui, rect: egui::Rect) {
    let painter = ui.painter();
    // Outer halo: expanding rounded outlines at very low alpha.
    for i in 1..=3 {
        let grow = i as f32 * 3.0;
        painter.rect_stroke(rect.expand(grow), RIM_RADIUS + grow, egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(96, 80, 200, 14 - i * 3)), egui::StrokeKind::Outside);
    }
    rounded_gradient(painter, rect, RIM_RADIUS);
    // Purple light pooled around the title area (upper left).
    let purple_pool = egui::pos2(rect.left() + 60.0, rect.top() + 36.0);
    for i in 0..6 {
        painter.circle_filled(purple_pool, 130.0 - i as f32 * 18.0, egui::Color32::from_rgba_unmultiplied(150, 90, 255, 6));
    }
    // Blue light rising along the lower-right edge.
    let blue_pool = egui::pos2(rect.right() - 24.0, rect.bottom() - 60.0);
    for i in 0..7 {
        painter.circle_filled(blue_pool, 190.0 - i as f32 * 24.0, egui::Color32::from_rgba_unmultiplied(70, 110, 255, 6));
    }
    painter.rect_stroke(rect, RIM_RADIUS, egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(RIM.r(), RIM.g(), RIM.b(), 110)), egui::StrokeKind::Inside);
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
        // The demo overlay shows the empty state (greeting + pills) full-height on the
        // left, like the reference design; the docked panel keeps the seeded transcript
        // so one capture reviews both layouts.
        let demo_id = egui::Id::new("assistant-demo-state");
        let mut demo_state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(demo_id).unwrap_or_default());
        let screen = ui.ctx().screen_rect();
        let rect = egui::Rect::from_min_size(
            egui::pos2(screen.left() + 42.0, screen.top() + 30.0),
            egui::vec2(336.0, (screen.height() - 66.0).max(200.0)),
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

/// Panel surface + header + tabs, with the active tab's body inside. In the dock, `app`
/// is present and the header's collapse control hides the group (reopen from the Window menu).
fn panel_chrome(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    let area = ui.available_rect_before_wrap().shrink2(egui::vec2(2.0, 2.0));
    panel_surface(ui, area);
    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            header(ui, state, app.as_mut().map(|a| &mut **a));
            ui.add_space(10.0);
            tabs(ui, state);
            ui.add_space(14.0);
            match state.tab {
                Tab::Chat => chat_body(ui, state),
                Tab::QuickActions => quick_actions_body(ui, state),
            }
        });
}

/// Sparkle glyph + title + pin/collapse controls, like the reference design's header.
fn header(ui: &mut egui::Ui, state: &mut AssistantState, app: Option<&mut FotoditApp>) {
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        let glyph = egui::Rect::from_min_size(ui.cursor().min + egui::vec2(0.0, -1.0), egui::vec2(22.0, 22.0));
        crate::icons::paint(ui, glyph, "sparkles", 22.0, SPARKLE_PURPLE);
        ui.add_space(30.0);
        ui.label(egui::RichText::new(tl!("AI Assistant")).color(egui::Color32::WHITE).strong().size(16.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (pin_rect, pin) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
            let pin = pin.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Pin panel"));
            if pin.hovered() {
                ui.painter().rect_filled(pin_rect, 7.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, pin_rect, "pin", 16.0, if state.pinned { ICON_BLUE } else { TEXT_SOFT });
            if pin.clicked() {
                state.pinned = !state.pinned;
            }
            // Collapse (hides the panel; reopen from the Window menu).
            let (chev_rect, chev) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
            let chev = chev.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Close panel"));
            if chev.hovered() {
                ui.painter().rect_filled(chev_rect, 7.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, chev_rect, "chevron-down", 16.0, TEXT_SOFT);
            if chev.clicked() {
                if let Some(app) = app {
                    app.ui.panels.assistant = false;
                }
            }
        });
    });
}

/// Chat | Quick Actions: two broad centred tabs over a divider rule; the active tab gets
/// a luminous accent stroke.
fn tabs(ui: &mut egui::Ui, state: &mut AssistantState) {
    let width = ui.available_width();
    ui.horizontal(|ui| {
        tab_label(ui, state, Tab::Chat, &tl!("Chat"), width / 2.0);
        tab_label(ui, state, Tab::QuickActions, &tl!("Quick Actions"), width / 2.0);
    });
    let line = ui.cursor();
    let band = egui::Rect::from_min_size(egui::pos2(line.min.x, line.min.y - 7.0), egui::vec2(width, 1.0));
    ui.painter().rect_filled(band, 0.0, egui::Color32::from_rgb(52, 62, 116));
    ui.add_space(6.0);
}

fn tab_label(ui: &mut egui::Ui, state: &mut AssistantState, tab: Tab, label: &str, width: f32) {
    let active = state.tab == tab;
    let color = if active { egui::Color32::WHITE } else { Tokens::get(ui.ctx()).text_dim };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 22.0), egui::Sense::click());
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(14.0), color);
    let text_pos = egui::pos2(rect.center().x - galley.size().x / 2.0, rect.center().y - galley.size().y / 2.0);
    ui.painter().add(egui::Shape::galley(text_pos, galley, color));
    if active {
        let underline = egui::Rect::from_min_size(
            egui::pos2(rect.center().x - galley.size().x / 2.0 - 4.0, rect.max.y + 3.0),
            egui::vec2(galley.size().x + 8.0, 3.5),
        );
        // Luminous stroke: a soft halo under the solid pill.
        ui.painter().rect_filled(underline.expand2(egui::vec2(3.0, 2.5)), 3.0, egui::Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 70));
        ui.painter().rect_filled(underline, 2.0, ACCENT);
    }
    if response.clicked() {
        state.tab = tab;
    }
}

/// One-tap canned edits: compact pills whose width follows the label, per the reference.
fn chip(ui: &mut egui::Ui, icon: &str, label: &str) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(14.0), egui::Color32::from_rgb(218, 223, 245));
    let width = 18.0 + 20.0 + 10.0 + galley.size().x + 20.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width.min(ui.available_width()), 42.0), egui::Sense::click());
    let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if response.hovered() { CHIP_BG_HOVER } else { CHIP_BG };
    ui.painter().rect_filled(rect, rect.height() / 2.0, fill);
    ui.painter().rect_stroke(rect, rect.height() / 2.0, egui::Stroke::new(1.0, CHIP_BORDER), egui::StrokeKind::Inside);
    let icon_box = egui::Rect::from_center_size(egui::pos2(rect.left() + 28.0, rect.center().y), egui::vec2(20.0, 20.0));
    crate::icons::paint(ui, icon_box, icon, 19.0, ICON_BLUE);
    ui.painter().add(egui::Shape::galley(egui::pos2(rect.left() + 48.0, rect.center().y - galley.size().y / 2.0), galley, egui::Color32::from_rgb(218, 223, 245)));
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

/// The chat itself: scrolling transcript with the composer pinned below.
fn chat_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    let footer_height = 62.0;
    egui::ScrollArea::vertical()
        .id_salt("assistant-transcript")
        .auto_shrink([false, false])
        .max_height((ui.available_height() - footer_height).max(40.0))
        .show(ui, |ui| {
            if state.transcript.is_empty() {
                greeting(ui);
                ui.add_space(18.0);
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
    ui.add_space(4.0);
}

/// The empty-state greeting bubble with the sparkle avatar, from the reference design.
fn greeting(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let glyph = egui::Rect::from_min_size(ui.cursor().min + egui::vec2(2.0, 6.0), egui::vec2(22.0, 22.0));
        crate::icons::paint(ui, glyph, "sparkles", 22.0, SPARKLE_PURPLE);
        ui.add_space(32.0);
        egui::Frame::NONE
            .fill(ASSISTANT_BUBBLE)
            .corner_radius(egui::CornerRadius::same(16))
            .stroke(egui::Stroke::new(1.0, BUBBLE_BORDER))
            .inner_margin(egui::Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(tl!("Hello! How can I help you edit your image today?")).color(TEXT_ON_NAVY).size(14.0),
                ).wrap());
            });
    });
}

/// The composer: a rounded, violet-lit container with a large paperclip on the left and
/// the prominent accent send button inset on the right.
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState) {
    egui::Frame::NONE
        .fill(COMPOSER_BG)
        .corner_radius(egui::CornerRadius::same(18))
        .stroke(egui::Stroke::new(1.0, COMPOSER_BORDER))
        .inner_margin(egui::Margin::symmetric(10, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (attach_rect, attach) = ui.allocate_exact_size(egui::vec2(28.0, 34.0), egui::Sense::click());
                let attach = attach.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Attach an image (coming soon)"));
                crate::icons::paint(ui, attach_rect, "paperclip", 18.0, TEXT_SOFT);

                let width = ui.available_width() - 44.0;
                let edit = egui::TextEdit::singleline(&mut state.draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT).size(14.0))
                    .font(egui::FontId::proportional(14.0))
                    .desired_width(width.max(60.0))
                    .frame(egui::Frame::NONE);
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !state.draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(36.0, 34.0), egui::Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                let send_fill = if sendable { ACCENT } else { egui::Color32::from_rgb(66, 84, 160) };
                ui.painter().rect_filled(send_rect, 11.0, send_fill);
                crate::icons::paint(ui, send_rect, "send", 15.0, egui::Color32::WHITE);
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
            .corner_radius(egui::CornerRadius::same(14))
            .inner_margin(egui::Margin::symmetric(12, 9))
            .show(ui, |ui| {
                ui.set_max_width(width * 0.85);
                ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
            });
    });
    ui.add_space(4.0);
}
