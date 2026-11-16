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
pub(crate) const ACCENT: egui::Color32 = egui::Color32::from_rgb(4, 79, 252);
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
const CHROME_TOP: egui::Color32 = egui::Color32::from_rgb(15, 23, 56);
const CHROME_MID: egui::Color32 = egui::Color32::from_rgb(11, 18, 44);
const CHROME_BOTTOM: egui::Color32 = egui::Color32::from_rgb(7, 12, 36);
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

/// Smooth radial light pool: one gradient mesh (bright center fading to nothing through a
/// mid ring) - soft glass light, never stacked rings.
fn radial_pool(painter: &egui::Painter, center: egui::Pos2, radius: f32, color: egui::Color32, alpha: u8) {
    const SEGMENTS: usize = 32;
    let rgba = |a: u8| egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a);
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(center, rgba(alpha));
    for ring in 0..2 {
        let r = if ring == 0 { radius * 0.55 } else { radius };
        let a = if ring == 0 { alpha / 3 } else { 0 };
        for i in 0..=SEGMENTS {
            let angle = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            mesh.colored_vertex(center + egui::vec2(angle.cos() * r, angle.sin() * r), rgba(a));
        }
    }
    for i in 0..SEGMENTS {
        let (m0, m1) = (1 + i as u32, 1 + i as u32 + 1);
        let (o0, o1) = (1 + SEGMENTS as u32 + 1 + i as u32, 1 + SEGMENTS as u32 + 1 + i as u32 + 1);
        mesh.add_triangle(0, m0, m1);
        mesh.add_triangle(m0, o0, m1);
        mesh.add_triangle(m1, o0, o1);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Continuous lit edge run: one gradient strip mesh, bright at `from` and easing to clear
/// at `to` with soft edges - an unbroken glass rim, not dashes.
fn edge_run(painter: &egui::Painter, from: egui::Pos2, to: egui::Pos2, width: f32, color: egui::Color32, alpha: u8) {
    const STEPS: usize = 24;
    let dir = (to - from).normalized();
    let n = egui::vec2(-dir.y, dir.x) * (width * 0.5);
    let rgba = |a: u8| egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a);
    let mut mesh = egui::Mesh::default();
    for i in 0..=STEPS {
        let t = i as f32 / STEPS as f32;
        let a = (alpha as f32 * (1.0 - t) * (1.0 - t)) as u8;
        let p = from + (to - from) * t;
        mesh.colored_vertex(p - n, rgba(0));
        mesh.colored_vertex(p, rgba(a));
        mesh.colored_vertex(p + n, rgba(0));
    }
    for i in 0..STEPS {
        for k in 0..2u32 {
            let (a0, a1) = (i as u32 * 3 + k, (i as u32 + 1) * 3 + k);
            mesh.add_triangle(a0, a1, a0 + 1);
            mesh.add_triangle(a0 + 1, a1, a1 + 1);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The enclosure: rounded gradient on a dark navy base, a fine violet rim, small
/// localized purple (title) and blue (lower-right) light, and a clean electric-blue
/// illumination that rises along the lower and left edges - glass, not a broad wash.
fn panel_surface(ui: &egui::Ui, rect: egui::Rect) {
    let painter = ui.painter();
    // Barely-there outer halo: two expanding outlines at very low alpha.
    for i in 1..=2 {
        let grow = i as f32 * 2.5;
        painter.rect_stroke(rect.expand(grow), RIM_RADIUS + grow, egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(96, 80, 200, 14 - i * 4)), egui::StrokeKind::Outside);
    }
    rounded_gradient(painter, rect, RIM_RADIUS);
    // Purple light pooled tight around the header band (upper left), kept faint so the
    // glass reads navy-blue, never a purple wash.
    radial_pool(painter, egui::pos2(rect.left() + 52.0, rect.top() + 30.0), 110.0, egui::Color32::from_rgb(150, 90, 255), 12);
    // Blue depth pooled at the lower-right corner, smaller and dimmer than the old wash.
    radial_pool(painter, egui::pos2(rect.right() - 18.0, rect.bottom() - 44.0), 170.0, egui::Color32::from_rgb(12, 36, 153), 30);
    // Electric-blue edge illumination, localized at the lower-left like the reference: a
    // smooth corner glow with continuous runs climbing the left edge and walking the
    // bottom edge to the right, both easing out - glass, not rings or dashes.
    radial_pool(painter, egui::pos2(rect.left() + 12.0, rect.bottom() - 12.0), 92.0, ACCENT, 64);
    edge_run(painter, egui::pos2(rect.left() + 1.5, rect.bottom() - RIM_RADIUS - 6.0), egui::pos2(rect.left() + 1.5, rect.bottom() - RIM_RADIUS - 280.0), 4.0, ACCENT, 112);
    edge_run(painter, egui::pos2(rect.left() + RIM_RADIUS + 6.0, rect.bottom() - 1.5), egui::pos2(rect.left() + RIM_RADIUS + 280.0, rect.bottom() - 1.5), 4.0, ACCENT, 112);
    // Glossy cool sheen along the upper-left rim: glass catching light, very faint.
    edge_run(painter, egui::pos2(rect.left() + RIM_RADIUS + 12.0, rect.top() + 1.0), egui::pos2(rect.left() + rect.width() * 0.55, rect.top() + 1.0), 2.0, egui::Color32::from_rgb(190, 210, 255), 24);
    // Fine violet rim, stroke only, not a luminous perimeter.
    painter.rect_stroke(rect, RIM_RADIUS, egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(RIM.r(), RIM.g(), RIM.b(), 108)), egui::StrokeKind::Inside);
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

/// The header's pin control: a solid angled pin (head upper-right, point lower-left),
/// painted in primitives so the shared outline `pin` icon stays untouched elsewhere.
fn paint_pin(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let s = rect.width() / 20.0;
    let c = rect.center();
    let axis = egui::vec2(-0.7071, 0.7071); // down-left along the pin's shaft
    let head = c - axis * 3.6 * s;
    let tip = c + axis * 6.2 * s;
    let perp = egui::vec2(-axis.y, axis.x);
    // Shaft: tapered quad from under the head to the point.
    let base = c + axis * 0.8 * s;
    painter.add(egui::Shape::convex_polygon(
        vec![base + perp * 1.5 * s, base - perp * 1.5 * s, tip + perp * 0.2 * s, tip - perp * 0.2 * s],
        color,
        egui::Stroke::NONE,
    ));
    // Brim: a flattened disk where the head meets the shaft.
    painter.add(egui::Shape::convex_polygon(
        vec![head + axis * 1.6 * s + perp * 3.6 * s, head + axis * 1.6 * s - perp * 3.6 * s, head + axis * 3.0 * s - perp * 2.6 * s, head + axis * 3.0 * s + perp * 2.6 * s],
        color,
        egui::Stroke::NONE,
    ));
    // Head: solid round cap.
    painter.circle_filled(head, 3.1 * s, color);
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
                Tab::Chat => chat_body(ui, state, bottom_inset, app.as_mut().map(|a| &mut **a)),
                Tab::QuickActions => quick_actions_body(ui, state, app.as_mut().map(|a| &mut **a)),
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
            paint_pin(ui.painter(), pin_rect, if state.pinned { ICON_WHITE } else { TEXT_SOFT });
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

fn quick_actions_body(ui: &mut egui::Ui, state: &mut AssistantState, mut app: Option<&mut FotoditApp>) {
    for &(icon, action) in QUICK_ACTIONS {
        if chip(ui, icon, action).clicked() {
            send(app.as_mut().map(|a| &mut **a), ui.ctx(), state, action.to_string());
        }
        ui.add_space(6.0);
    }
    if chip(ui, "ellipsis", &tl!("More options...")).clicked() {
        state.tab = Tab::Chat;
    }
}

/// The chat itself: scrolling transcript with the composer pinned below.
fn chat_body(ui: &mut egui::Ui, state: &mut AssistantState, bottom_inset: f32, mut app: Option<&mut FotoditApp>) {
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
                        send(app.as_mut().map(|a| &mut **a), ui.ctx(), state, action.to_string());
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
    input_row(ui, state, app.as_mut().map(|a| &mut **a));
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
fn input_row(ui: &mut egui::Ui, state: &mut AssistantState, app: Option<&mut FotoditApp>) {
    let top = ui.cursor().min;
    let width = ui.available_width();
    let inner = egui::Frame::NONE
        .fill(COMPOSER_BG)
        .corner_radius(egui::CornerRadius::same(16))
        .stroke(egui::Stroke::new(1.0, CHIP_BORDER))
        .inner_margin(egui::Margin::symmetric(10, 11))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (attach_rect, attach) = ui.allocate_exact_size(egui::vec2(22.0, 38.0), egui::Sense::click());
                let attach = attach.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Attach an image (coming soon)"));
                crate::icons::paint(ui, attach_rect, "paperclip", 16.0, TEXT_SOFT);

                let width = ui.available_width() - 42.0;
                let edit = egui::TextEdit::singleline(&mut state.draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT).size(13.5))
                    .font(egui::FontId::proportional(14.0))
                    .desired_width(width.max(60.0))
                    .frame(egui::Frame::NONE);
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !state.draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                // Bright electric-blue send with a white plane, matching the reference even
                // while empty; an empty draft still cannot be sent (gated below).
                let send_fill = ACCENT;
                ui.painter().rect_filled(send_rect, 12.0, send_fill);
                crate::icons::paint(ui, send_rect, "send", 17.0, egui::Color32::WHITE);
                if (send_click.clicked() || enter) && sendable {
                    let asked = state.draft.trim().to_string();
                    state.draft.clear();
                    send(app, ui.ctx(), state, asked);
                    response.request_focus();
                }
            });
        });
    // Violet accent along the composer's upper-left edge, per the reference: one
    // continuous lit run - the full top edge, around the corner arc, down the left edge.
    let rect = inner.response.rect;
    let _ = (top, width);
    let r = 16.0_f32;
    let corner = egui::pos2(rect.left() + r, rect.top() + r);
    let mut run: Vec<egui::Pos2> = vec![egui::pos2(rect.right() - r, rect.top() + 0.5), egui::pos2(corner.x, rect.top() + 0.5)];
    for i in 1..5 {
        let a = (-90.0 - 90.0 * i as f32 / 4.0).to_radians();
        run.push(corner + egui::vec2(r * a.cos(), r * a.sin()));
    }
    run.push(egui::pos2(rect.left() + 0.5, rect.bottom() - r));
    let bloom = egui::Color32::from_rgba_unmultiplied(COMPOSER_EDGE.r(), COMPOSER_EDGE.g(), COMPOSER_EDGE.b(), 72);
    ui.painter().add(egui::Shape::line(run.clone(), egui::Stroke::new(5.5, bloom)));
    ui.painter().add(egui::Shape::line(run, egui::Stroke::new(1.5, COMPOSER_EDGE)));
}

/// Record an exchange. With an installed backend the prompt runs through the host's AI
/// pipeline against the admitted command path; without one the panel says so instead of
/// simulating a reply.
fn send(app: Option<&mut FotoditApp>, ctx: &egui::Context, state: &mut AssistantState, prompt: String) {
    state.tab = Tab::Chat;
    let answer = match app {
        Some(app) => {
            // Take the backend out so the app stays borrowable for the admitted command path.
            let mut backend = app.services.assistant_backend.take();
            let answer = match backend.as_mut() {
                Some(backend) => backend.reply(app, ctx, &prompt).unwrap_or_else(|error| error),
                None => tl!("The AI engine is not connected in this build, so I can't edit images yet.").to_string(),
            };
            app.services.assistant_backend = backend;
            answer
        }
        None => tl!("The AI engine is not connected in this build, so I can't edit images yet.").to_string(),
    };
    state.transcript.push(Exchange { user: prompt, assistant: answer });
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
