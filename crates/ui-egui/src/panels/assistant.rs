//! The AI Assistant panel: chat-style natural-language editing. Layout follows the
//! approved reference design: rounded navy enclosure with a fine violet rim, localized
//! purple and blue lighting, header, Chat / Quick Actions tabs, quick-action pills,
//! scrolling transcript with bubbles, and the composer pinned at the bottom.

use super::{FotoditApp, Tokens};

/// Navy chrome palette from the approved design; shared with the home screen's AI rail.
pub(crate) const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(11, 18, 44);
pub(crate) const CHIP_BG: egui::Color32 = egui::Color32::from_rgb(19, 28, 63);
pub(crate) const CHIP_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(26, 37, 86);
/// Faint glass edge on the layered surfaces - low-contrast, not a drawn outline.
const CHIP_BORDER: egui::Color32 = egui::Color32::from_rgba_premultiplied(12, 15, 26, 26);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(20, 30, 71);
const BUBBLE_BORDER: egui::Color32 = egui::Color32::from_rgba_premultiplied(14, 18, 30, 30);
const COMPOSER_BG: egui::Color32 = egui::Color32::from_rgb(20, 30, 68);
/// Violet accent at the composer's upper-left edge.
const COMPOSER_EDGE: egui::Color32 = egui::Color32::from_rgb(122, 100, 232);
const RIM: egui::Color32 = egui::Color32::from_rgb(112, 96, 220);
/// Electric blue: selected controls, send, saturated highlights.
pub(crate) const ACCENT: egui::Color32 = egui::Color32::from_rgb(46, 126, 255);
/// Saturated cyan core of the active-tab underline.
const TAB_GLOW: egui::Color32 = egui::Color32::from_rgb(9, 191, 255);
/// Crisp white for icons on the navy chrome.
const ICON_WHITE: egui::Color32 = egui::Color32::from_rgb(244, 247, 255);
/// Purple of the identity badge disk.
const BADGE_PURPLE: egui::Color32 = egui::Color32::from_rgb(146, 92, 255);
/// Muted label colour on the navy chrome.
pub(crate) const TEXT_SOFT: egui::Color32 = egui::Color32::from_rgb(158, 168, 205);
/// Body text colour on the navy chrome.
pub(crate) const TEXT_ON_NAVY: egui::Color32 = egui::Color32::from_rgb(232, 236, 250);
/// Gradient stops for the panel chrome, top to bottom: dark navy base, never flat matte.
const CHROME_TOP: egui::Color32 = egui::Color32::from_rgb(12, 19, 48);
const CHROME_MID: egui::Color32 = egui::Color32::from_rgb(11, 18, 44);
const CHROME_BOTTOM: egui::Color32 = egui::Color32::from_rgb(9, 15, 42);
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
        mesh.indices.push(base + ((i + 1) % pts.len()) as u32);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The enclosure: rounded gradient on a dark navy base, a fine violet rim, small
/// localized purple (title) and blue (lower-right) light, and a clean electric-blue
/// illumination that rises along the lower and right edges - glass, not a broad wash.
fn panel_surface(ui: &egui::Ui, rect: egui::Rect) {
    let painter = ui.painter();
    // Barely-there outer halo: two expanding outlines at very low alpha.
    for i in 1..=2 {
        let grow = i as f32 * 2.5;
        painter.rect_stroke(rect.expand(grow), RIM_RADIUS + grow, egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(96, 80, 200, 7 - i * 2)), egui::StrokeKind::Outside);
    }
    rounded_gradient(painter, rect, RIM_RADIUS);
    // Purple light pooled tight around the header band (upper left).
    let purple_pool = egui::pos2(rect.left() + 52.0, rect.top() + 30.0);
    for i in 0..5 {
        painter.circle_filled(purple_pool, 96.0 - i as f32 * 16.0, egui::Color32::from_rgba_unmultiplied(150, 90, 255, 5));
    }
    // Blue light pooled at the lower-right corner, smaller and dimmer than the old wash.
    let blue_pool = egui::pos2(rect.right() - 18.0, rect.bottom() - 44.0);
    for i in 0..6 {
        painter.circle_filled(blue_pool, 130.0 - i as f32 * 18.0, egui::Color32::from_rgba_unmultiplied(12, 36, 153, 8));
    }
    // Electric-blue edge illumination: a short bright run at the lower-right corner that
    // fades as it climbs the right edge and walks the bottom edge to the left.
    let edge = |alpha: u8| egui::Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), alpha);
    let corner = egui::pos2(rect.right() - RIM_RADIUS * 0.6, rect.bottom() - RIM_RADIUS * 0.6);
    for i in 0..5 {
        let y = rect.bottom() - RIM_RADIUS - 14.0 - i as f32 * 26.0;
        let alpha = 34 - i * 6;
        painter.line_segment([egui::pos2(rect.right() - 1.5, y + 22.0), egui::pos2(rect.right() - 1.5, y)], egui::Stroke::new(1.5, edge(alpha)));
    }
    for i in 0..5 {
        let x = rect.right() - RIM_RADIUS - 14.0 - i as f32 * 26.0;
        let alpha = 34 - i * 6;
        painter.line_segment([egui::pos2(x + 22.0, rect.bottom() - 1.5), egui::pos2(x, rect.bottom() - 1.5)], egui::Stroke::new(1.5, edge(alpha)));
    }
    let _ = corner;
    // Fine violet rim, stroke only, not a luminous perimeter.
    painter.rect_stroke(rect, RIM_RADIUS, egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(RIM.r(), RIM.g(), RIM.b(), 76)), egui::StrokeKind::Inside);
}

/// The identity badge: a filled, softly glowing purple disk with a bold white
/// four-point star. Paints at the badge's centre for a given disk radius.
pub(crate) fn paint_badge(painter: &egui::Painter, center: egui::Pos2, radius: f32) {
    // Glow rings.
    painter.circle_filled(center, radius * 1.5, egui::Color32::from_rgba_unmultiplied(BADGE_PURPLE.r(), BADGE_PURPLE.g(), BADGE_PURPLE.b(), 18));
    painter.circle_filled(center, radius * 1.22, egui::Color32::from_rgba_unmultiplied(BADGE_PURPLE.r(), BADGE_PURPLE.g(), BADGE_PURPLE.b(), 34));
    painter.circle_filled(center, radius, BADGE_PURPLE);
    // Bold white four-point star: tips on the axes, concave points on the diagonals.
    let tip = radius * 0.72;
    let nick = radius * 0.22;
    let mut pts = Vec::with_capacity(8);
    for i in 0..4 {
        let a = i as f32 * std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_2;
        pts.push(center + egui::vec2(tip * a.cos(), tip * a.sin()));
        let m = a + std::f32::consts::FRAC_PI_4;
        pts.push(center + egui::vec2(nick * m.cos(), nick * m.sin()));
    }
    painter.add(egui::epaint::PathShape {
        points: pts,
        closed: true,
        fill: egui::Color32::WHITE,
        stroke: egui::epaint::PathStroke::NONE,
    });
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

/// The assistant rail at the workspace's start edge (see panels::dock::dock_area): the
/// real installed presentation of the reference design.
pub(crate) fn assistant_rail(app: &mut FotoditApp, ui: &mut egui::Ui) {
    let mut state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(state_id()).unwrap_or_default());
    panel_chrome(ui, &mut state, Some(app));
    ui.ctx().data_mut(|d| d.insert_temp(state_id(), state));
}

/// Legacy dock hook: the panel moved to the start-edge rail. A workspace saved with an
/// Assistant tab keeps it valid but sees this note; Window > Assistant shows the rail.
pub(crate) fn assistant_panel(app: &mut FotoditApp, ui: &mut egui::Ui) {
    let _ = app;
    ui.centered_and_justified(|ui| {
        ui.label(egui::RichText::new(tl!("The Assistant now opens as a rail on the left. Use Window > Assistant.")).color(TEXT_SOFT));
    });
}

/// Panel surface + header + tabs, with the active tab's body inside. In the dock, `app`
/// is present and the header's collapse control hides the group (reopen from the Window menu).
fn panel_chrome(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    let area = ui.available_rect_before_wrap().shrink2(egui::vec2(2.0, 2.0));
    panel_surface(ui, area);
    // The composer floats ~8% of the panel height above the bottom edge (reference
    // proportion), so the gap scales with the viewport instead of a fixed pixel value.
    let bottom_inset = (area.height() * 0.08).clamp(28.0, 88.0);
    egui::Frame::NONE
        .inner_margin(egui::Margin { left: 14, right: 14, top: 14, bottom: 10 })
        .show(ui, |ui| {
            header(ui, state, app.as_mut().map(|a| &mut **a));
            ui.add_space(10.0);
            tabs(ui, state);
            ui.add_space(14.0);
            match state.tab {
                Tab::Chat => chat_body(ui, state, bottom_inset),
                Tab::QuickActions => quick_actions_body(ui, state),
            }
        });
}

/// Sparkle glyph + title + pin/collapse controls, like the reference design's header.
fn header(ui: &mut egui::Ui, state: &mut AssistantState, app: Option<&mut FotoditApp>) {
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        let (badge_rect, _) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
        paint_badge(ui.painter(), badge_rect.center(), 11.0);
        ui.add_space(6.0);
        ui.label(egui::RichText::new(tl!("AI Assistant")).color(egui::Color32::WHITE).strong().size(16.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (pin_rect, pin) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
            let pin = pin.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Pin panel"));
            if pin.hovered() {
                ui.painter().rect_filled(pin_rect, 6.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, pin_rect, "pin", 14.0, if state.pinned { ICON_WHITE } else { TEXT_SOFT });
            if pin.clicked() {
                state.pinned = !state.pinned;
            }
            // Collapse (hides the panel; reopen from the Window menu): compact chevron.
            let (chev_rect, chev) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
            let chev = chev.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Close panel"));
            if chev.hovered() {
                ui.painter().rect_filled(chev_rect, 6.0, CHIP_BG_HOVER);
            }
            crate::icons::paint(ui, chev_rect, "chevron-down", 14.0, TEXT_SOFT);
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
    // Compact centered pair: each tab hugs its label with fixed padding, the pair sits
    // centered instead of spreading across the full width.
    let font = egui::FontId::proportional(14.0);
    let chat_w = ui.painter().layout_no_wrap(tl!("Chat").to_string(), font.clone(), egui::Color32::WHITE).size().x;
    let qa_w = ui.painter().layout_no_wrap(tl!("Quick Actions").to_string(), font.clone(), egui::Color32::WHITE).size().x;
    let pair = chat_w + qa_w + 2.0 * 18.0 + 26.0;
    let lead = ((width - pair) / 2.0).max(0.0);
    ui.horizontal(|ui| {
        ui.add_space(lead);
        tab_label(ui, state, Tab::Chat, &tl!("Chat"), chat_w + 18.0);
        ui.add_space(26.0);
        tab_label(ui, state, Tab::QuickActions, &tl!("Quick Actions"), qa_w + 18.0);
    });
    let line = ui.cursor();
    let band = egui::Rect::from_min_size(egui::pos2(line.min.x, line.min.y - 7.0), egui::vec2(width, 1.0));
    ui.painter().rect_filled(band, 0.0, egui::Color32::from_rgb(41, 51, 93));
    ui.add_space(6.0);
}

fn tab_label(ui: &mut egui::Ui, state: &mut AssistantState, tab: Tab, label: &str, width: f32) {
    let active = state.tab == tab;
    let color = if active { egui::Color32::WHITE } else { Tokens::get(ui.ctx()).text_dim };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 22.0), egui::Sense::click());
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(14.0), color);
    let text_w = galley.size().x;
    let text_pos = egui::pos2(rect.center().x - text_w / 2.0, rect.center().y - galley.size().y / 2.0);
    ui.painter().add(egui::Shape::galley(text_pos, galley, color));
    if active {
        // Short bright glow: narrow saturated cyan core with a small bloom, not a long dash.
        let underline = egui::Rect::from_min_size(
            egui::pos2(rect.center().x - text_w / 2.0 - 3.0, rect.max.y + 3.0),
            egui::vec2(text_w + 6.0, 2.5),
        );
        ui.painter().rect_filled(underline.expand2(egui::vec2(2.5, 2.0)), 2.5, egui::Color32::from_rgba_unmultiplied(TAB_GLOW.r(), TAB_GLOW.g(), TAB_GLOW.b(), 60));
        ui.painter().rect_filled(underline, 1.5, TAB_GLOW);
    }
    if response.clicked() {
        state.tab = tab;
    }
}

/// One-tap canned edits: compact pills whose width follows the label, per the reference.
fn chip(ui: &mut egui::Ui, icon: &str, label: &str) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(14.0), egui::Color32::from_rgb(228, 233, 248));
    let width = 14.0 + 18.0 + 8.0 + galley.size().x + 16.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width.min(ui.available_width()), 36.0), egui::Sense::click());
    let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if response.hovered() { CHIP_BG_HOVER } else { CHIP_BG };
    // Rounded rectangle, not a maximized capsule; the glass edge is a faint highlight.
    ui.painter().rect_filled(rect, 11.0, fill);
    ui.painter().rect_stroke(rect, 11.0, egui::Stroke::new(1.0, CHIP_BORDER), egui::StrokeKind::Inside);
    let icon_box = egui::Rect::from_center_size(egui::pos2(rect.left() + 23.0, rect.center().y), egui::vec2(18.0, 18.0));
    crate::icons::paint(ui, icon_box, icon, 17.0, ICON_WHITE);
    ui.painter().add(egui::Shape::galley(egui::pos2(rect.left() + 40.0, rect.center().y - galley.size().y / 2.0), galley, egui::Color32::from_rgb(228, 233, 248)));
    response
}

fn quick_actions_body(ui: &mut egui::Ui, state: &mut AssistantState) {
    for &(icon, action) in QUICK_ACTIONS {
        if chip(ui, icon, action).clicked() {
            send(state, action.to_string());
        }
        ui.add_space(6.0);
    }
    if chip(ui, "ellipsis", &tl!("More options...")).clicked() {
        state.tab = Tab::Chat;
    }
}

/// The chat itself: scrolling transcript with the composer pinned below.
fn chat_body(ui: &mut egui::Ui, state: &mut AssistantState, bottom_inset: f32) {
    let footer_height = 74.0 + bottom_inset;
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
                    ui.add_space(6.0);
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

    ui.add_space(8.0);
    input_row(ui, state);
    ui.add_space((bottom_inset - 18.0).max(8.0));
}

/// The empty-state greeting bubble with the sparkle avatar, from the reference design.
fn greeting(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let (badge_rect, _) = ui.allocate_exact_size(egui::vec2(30.0, 30.0), egui::Sense::hover());
        paint_badge(ui.painter(), badge_rect.center(), 14.0);
        ui.add_space(8.0);
        egui::Frame::NONE
            .fill(ASSISTANT_BUBBLE)
            .corner_radius(egui::CornerRadius::same(20))
            .stroke(egui::Stroke::new(1.0, BUBBLE_BORDER))
            .inner_margin(egui::Margin::symmetric(14, 11))
            .show(ui, |ui| {
                // Compact two-line bubble: capped width so the greeting wraps early.
                ui.set_max_width(196.0);
                ui.add(egui::Label::new(
                    egui::RichText::new(tl!("Hello! How can I help you edit your image today?")).color(TEXT_ON_NAVY).size(14.0),
                ).wrap());
            });
    });
}

/// The composer: a rounded, violet-lit container with a large paperclip on the left and
/// the prominent accent send button inset on the right.
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState) {
    let top = ui.cursor().min;
    let width = ui.available_width();
    let inner = egui::Frame::NONE
        .fill(COMPOSER_BG)
        .corner_radius(egui::CornerRadius::same(16))
        .stroke(egui::Stroke::new(1.0, CHIP_BORDER))
        .inner_margin(egui::Margin::symmetric(12, 12))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (attach_rect, attach) = ui.allocate_exact_size(egui::vec2(26.0, 38.0), egui::Sense::click());
                let attach = attach.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Attach an image (coming soon)"));
                crate::icons::paint(ui, attach_rect, "paperclip", 17.0, TEXT_SOFT);

                let width = ui.available_width() - 46.0;
                let edit = egui::TextEdit::singleline(&mut state.draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT).size(14.0))
                    .font(egui::FontId::proportional(14.0))
                    .desired_width(width.max(60.0))
                    .frame(egui::Frame::NONE);
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !state.draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(38.0, 38.0), egui::Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                // Bright electric-blue send with a white plane; muted only while empty.
                let send_fill = if sendable { ACCENT } else { egui::Color32::from_rgb(38, 52, 104) };
                ui.painter().rect_filled(send_rect, 12.0, send_fill);
                crate::icons::paint(ui, send_rect, "send", 17.0, egui::Color32::WHITE);
                if (send_click.clicked() || enter) && sendable {
                    let asked = state.draft.trim().to_string();
                    state.draft.clear();
                    send(state, asked);
                    response.request_focus();
                }
            });
        });
    // Violet accent along the composer's upper-left edge, per the reference.
    let rect = inner.response.rect;
    let _ = (top, width);
    ui.painter().line_segment(
        [egui::pos2(rect.left() + 18.0, rect.top() + 0.5), egui::pos2(rect.left() + 64.0, rect.top() + 0.5)],
        egui::Stroke::new(1.5, COMPOSER_EDGE),
    );
    ui.painter().line_segment(
        [egui::pos2(rect.left() + 0.5, rect.top() + 16.0), egui::pos2(rect.left() + 0.5, rect.top() + 34.0)],
        egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(COMPOSER_EDGE.r(), COMPOSER_EDGE.g(), COMPOSER_EDGE.b(), 120)),
    );
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
