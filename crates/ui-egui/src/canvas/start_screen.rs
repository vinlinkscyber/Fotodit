use super::tabs::{open_on_double_click, paint_dots};
use super::{FotoditApp, Rect, Sense};
use crate::panels::assistant::{ACCENT, CHIP_BG, PANEL_BG, TEXT_ON_NAVY, TEXT_SOFT};

/// The page behind the home content: a shade deeper than the AI chrome.
const PAGE_BG: egui::Color32 = egui::Color32::from_rgb(11, 14, 30);
/// Card surfaces on the page.
const CARD_BG: egui::Color32 = egui::Color32::from_rgb(20, 25, 52);
/// Card outlines on the page.
const CARD_LINE: egui::Color32 = egui::Color32::from_rgb(44, 52, 96);

pub(super) fn start_screen(app: &mut FotoditApp, ui: &mut egui::Ui) {
    let area = ui.available_rect_before_wrap();
    paint_dots(ui, area, app.session.prefs().interface.show_canvas_dots);
    ui.painter().rect_filled(area, 0.0, PAGE_BG);
    open_on_double_click(app, ui, area);

    // File › Open Recent, newest first (on the web there are no paths to reopen).
    let show_recent = app.session.prefs().interface.show_home_recent_files;
    let recent: Vec<String> =
        if cfg!(target_arch = "wasm32") || !show_recent { Vec::new() } else { app.ui.recent_files.iter().take(HOME_RECENT).cloned().collect() };

    let is_narrow = area.width() < 900.0;
    let pad_x = if is_narrow { 16.0 } else { 36.0 };
    let pad_y = if is_narrow { 24.0 } else { 40.0 };

    if is_narrow {
        let card_w = (area.width() - 32.0).clamp(0.0, 620.0);
        let card = Rect::from_center_size(area.center(), egui::vec2(card_w, (area.height() - 40.0).max(0.0)));
        ui.scope_builder(egui::UiBuilder::new().max_rect(card), |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                render_hero(ui);
                ui.add_space(14.0);
                render_start_actions_row(app, ui, card.width());
                ui.add_space(16.0);
                render_drop_zone(app, ui, card.width());
                ui.add_space(20.0);
                render_templates(app, ui, card.width());
                ui.add_space(20.0);
                if !recent.is_empty() {
                    crate::home_recent_ui::show_home_recent(app, ui, &recent, card.width());
                }
                ui.add_space(16.0);
                render_ai_rail(app, ui, card.width());
            });
        });
    } else {
        let rail_w = 190.0;
        let ai_w = 300.0;
        let rail_rect = Rect::from_min_size(egui::pos2(area.left() + pad_x, area.top() + pad_y), egui::vec2(rail_w, area.height() - pad_y * 2.0));
        let ai_rect = Rect::from_min_size(egui::pos2(area.right() - pad_x - ai_w, area.top() + pad_y), egui::vec2(ai_w, area.height() - pad_y * 2.0));
        let content_rect = Rect::from_min_max(egui::pos2(rail_rect.right() + 28.0, area.top() + pad_y), egui::pos2(ai_rect.left() - 28.0, area.bottom() - 16.0));

        // 1. Left navigation rail
        ui.scope_builder(egui::UiBuilder::new().max_rect(rail_rect), |ui| {
            render_start_rail(app, ui, rail_rect.width());
        });

        // 2. Center content (scrollable)
        ui.scope_builder(egui::UiBuilder::new().max_rect(content_rect), |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let w = content_rect.width();
                render_hero(ui);
                ui.add_space(12.0);
                render_search(ui, w);
                ui.add_space(16.0);
                render_drop_zone(app, ui, w);
                ui.add_space(22.0);
                render_templates(app, ui, w);
                ui.add_space(22.0);
                if !recent.is_empty() {
                    crate::home_recent_ui::show_home_recent(app, ui, &recent, w);
                }
            });
        });

        // 3. Right AI rail
        ui.scope_builder(egui::UiBuilder::new().max_rect(ai_rect), |ui| {
            render_ai_rail(app, ui, ai_rect.width());
        });
    }
}

/// "What will you create today?" + the tagline, as in the approved design.
fn render_hero(ui: &mut egui::Ui) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(tl!("What will you create today?")).color(TEXT_ON_NAVY).strong().size(26.0));
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(tl!("Start a new project, open recent files, or explore AI tools to bring your vision to life."))
            .color(TEXT_SOFT)
            .size(13.0),
    );
}

/// Template/recent filter box with the search icon.
fn render_search(ui: &mut egui::Ui, width: f32) {
    let mut query = ui.ctx().data_mut(|d| d.get_temp::<String>(search_id()).unwrap_or_default());
    egui::Frame::NONE
        .fill(CARD_BG)
        .corner_radius(egui::CornerRadius::same(8))
        .stroke(egui::Stroke::new(1.0, CARD_LINE))
        .inner_margin(egui::Margin::symmetric(12, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let icon_box = Rect::from_center_size(ui.cursor().left_center() + egui::vec2(8.0, 0.0), egui::vec2(16.0, 16.0));
                crate::icons::paint(ui, icon_box, "zoom-in", 15.0, TEXT_SOFT);
                ui.add_space(22.0);
                let edit = egui::TextEdit::singleline(&mut query)
                    .hint_text(egui::RichText::new(tl!("Search templates, recent files...")).color(TEXT_SOFT))
                    .desired_width((width - 80.0).max(120.0))
                    .frame(egui::Frame::NONE);
                ui.add(edit);
            });
        });
    ui.ctx().data_mut(|d| d.insert_temp(search_id(), query));
}

fn search_id() -> egui::Id {
    egui::Id::new("home-search-query")
}

fn search_query(ctx: &egui::Context) -> String {
    ctx.data_mut(|d| d.get_temp::<String>(search_id()).unwrap_or_default()).to_lowercase()
}

/// The dashed drag & drop zone with the cloud icon.
fn render_drop_zone(app: &mut FotoditApp, ui: &mut egui::Ui, width: f32) {
    let drop_w = width.min(720.0);
    let drop_h = 170.0;
    let (drop_r, drop_resp) = ui.allocate_exact_size(egui::vec2(drop_w, drop_h), Sense::click());
    let drop_resp = drop_resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tl!("Open…"));
    let is_hov = drop_resp.hovered();

    ui.painter().rect_filled(drop_r, 10.0, if is_hov { PANEL_BG } else { CARD_BG });
    dashed_rect(ui, drop_r, 10.0, if is_hov { ACCENT } else { CARD_LINE });

    let icon_box = Rect::from_center_size(egui::pos2(drop_r.center().x, drop_r.center().y - 34.0), egui::vec2(40.0, 40.0));
    crate::icons::paint(ui, icon_box, "cloud", 34.0, ACCENT);
    ui.painter().text(
        egui::pos2(drop_r.center().x, drop_r.center().y + 4.0),
        egui::Align2::CENTER_CENTER,
        start_screen_drop_hint(app),
        crate::theme::semibold(14.0),
        TEXT_ON_NAVY,
    );
    ui.painter().text(
        egui::pos2(drop_r.center().x, drop_r.center().y + 28.0),
        egui::Align2::CENTER_CENTER,
        tl!("Supported: JPG, PNG, PSD, RAW"),
        crate::theme::medium(12.0),
        TEXT_SOFT,
    );

    if drop_resp.clicked() {
        let _ = app.open_dialog_file();
    }
}

/// egui has no dashed stroke; paint one from short segments.
fn dashed_rect(ui: &egui::Ui, rect: Rect, radius: f32, color: egui::Color32) {
    let stroke = egui::Stroke::new(1.5, color);
    let dash = 7.0;
    let gap = 5.0;
    let mut paint_side = |from: egui::Pos2, to: egui::Pos2, horizontal: bool| {
        let len = if horizontal { to.x - from.x } else { to.y - from.y };
        let mut d = radius;
        while d < len - radius {
            let end = (d + dash).min(len - radius);
            let (a, b) = if horizontal {
                (egui::pos2(from.x + d, from.y), egui::pos2(from.x + end, from.y))
            } else {
                (egui::pos2(from.x, from.y + d), egui::pos2(from.x, from.y + end))
            };
            ui.painter().line_segment([a, b], stroke);
            d = end + gap;
        }
    };
    paint_side(rect.left_top(), rect.right_top(), true);
    paint_side(rect.left_bottom(), rect.right_bottom(), true);
    paint_side(rect.left_top(), rect.left_bottom(), false);
    paint_side(rect.right_top(), rect.right_bottom(), false);
}

/// One quick-start template: the sized ones prefill New Document, the workflow ones open a
/// file and prime the Assistant.
struct Template {
    name: &'static str,
    dims: &'static str,
    tag: &'static str,
    icon: &'static str,
    size: Option<(u64, u64)>,
    prompt: Option<&'static str>,
    new: bool,
}

const TEMPLATES: &[Template] = &[
    Template { name: "Blank Canvas", dims: "1920×1080", tag: "Standard desktop", icon: "artboard", size: Some((1920, 1080)), prompt: None, new: true },
    Template { name: "Instagram Post", dims: "1080×1080", tag: "Social media", icon: "square", size: Some((1080, 1080)), prompt: None, new: false },
    Template { name: "YouTube Thumbnail", dims: "1280×720", tag: "Video", icon: "rectangle-horizontal", size: Some((1280, 720)), prompt: None, new: false },
    Template { name: "Photo Retouch", dims: "", tag: "Portrait cleanup & polish", icon: "bandage-sparkle", size: None, prompt: Some("Clean up this portrait and polish the skin, keeping it natural"), new: false },
    Template { name: "AI Enhance", dims: "", tag: "Upscale & sharpen photo", icon: "wand-sparkles", size: None, prompt: Some("Enhance image quality: upscale and sharpen the details"), new: false },
];

fn render_templates(app: &mut FotoditApp, ui: &mut egui::Ui, width: f32) {
    let query = search_query(ui.ctx());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Quick start templates")).color(TEXT_ON_NAVY).strong().size(16.0));
    });
    ui.add_space(10.0);

    let card_w = 128.0;
    let card_h = 118.0;
    let gap = 12.0;
    let per_row = ((width + gap) / (card_w + gap)).floor().max(1.0) as usize;

    let mut shown = 0usize;
    for row in TEMPLATES
        .iter()
        .filter(|t| query.is_empty() || t.name.to_lowercase().contains(&query) || t.tag.to_lowercase().contains(&query))
        .collect::<Vec<_>>()
        .chunks(per_row)
    {
        ui.horizontal(|ui| {
            for template in row {
                template_card(app, ui, template, card_w, card_h);
                ui.add_space(gap);
                shown += 1;
            }
        });
        ui.add_space(gap);
    }
    if shown == 0 {
        ui.label(egui::RichText::new(tl!("No templates match your search.")).color(TEXT_SOFT).size(12.5));
    }
}

fn template_card(app: &mut FotoditApp, ui: &mut egui::Ui, template: &Template, w: f32, h: f32) {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, h), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if resp.hovered() { PANEL_BG } else { CARD_BG };
    ui.painter().rect_filled(rect, 10.0, fill);
    ui.painter().rect_stroke(rect, 10.0, egui::Stroke::new(1.0, if resp.hovered() { ACCENT } else { CARD_LINE }), egui::StrokeKind::Inside);

    // NEW badge
    if template.new {
        let badge = Rect::from_min_size(rect.min + egui::vec2(8.0, 8.0), egui::vec2(30.0, 15.0));
        ui.painter().rect_filled(badge, 7.0, ACCENT);
        ui.painter().text(badge.center(), egui::Align2::CENTER_CENTER, tl!("NEW"), crate::theme::semibold(9.0), egui::Color32::WHITE);
    }

    let icon_box = Rect::from_center_size(egui::pos2(rect.center().x, rect.top() + 34.0), egui::vec2(30.0, 30.0));
    crate::icons::paint(ui, icon_box, template.icon, 26.0, ACCENT);

    let name = if template.dims.is_empty() { template.name.to_string() } else { format!("{} {}", template.name, template.dims) };
    ui.painter().text(
        egui::pos2(rect.center().x, rect.top() + 66.0),
        egui::Align2::CENTER_CENTER,
        template.name,
        crate::theme::semibold(12.5),
        TEXT_ON_NAVY,
    );
    let sub = if template.dims.is_empty() { template.tag } else { name.split(' ').last().unwrap_or(template.dims) };
    ui.painter().text(egui::pos2(rect.center().x, rect.top() + 84.0), egui::Align2::CENTER_CENTER, sub, crate::theme::medium(11.0), TEXT_SOFT);
    ui.painter().text(
        egui::pos2(rect.center().x, rect.top() + 100.0),
        egui::Align2::CENTER_CENTER,
        if template.dims.is_empty() { "" } else { template.tag },
        crate::theme::medium(11.0),
        TEXT_SOFT,
    );

    if resp.clicked() {
        if let Some((w, h)) = template.size {
            let mut fields = app.new_document_fields();
            fields.insert("width".into(), w.into());
            fields.insert("height".into(), h.into());
            app.ui.open_dialog(crate::state::DialogKind::NewDocument, fields);
        } else if let Some(prompt) = template.prompt {
            crate::panels::assistant::seed_prompt(ui.ctx(), prompt);
            app.ui.panels.assistant = true;
            let _ = app.open_dialog_file();
        }
    }
}

/// The right rail: "Discover / AI Capabilities" cards + the Assistant input, on navy chrome.
fn render_ai_rail(app: &mut FotoditApp, ui: &mut egui::Ui, width: f32) {
    egui::Frame::NONE
        .fill(PANEL_BG)
        .corner_radius(egui::CornerRadius::same(12))
        .stroke(egui::Stroke::new(1.0, CARD_LINE))
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(width - 28.0);
            ui.horizontal(|ui| {
                let icon_box = Rect::from_center_size(ui.cursor().left_center() + egui::vec2(8.0, 0.0), egui::vec2(18.0, 18.0));
                crate::icons::paint(ui, icon_box, "sparkles", 16.0, ACCENT);
                ui.add_space(22.0);
                ui.label(egui::RichText::new(tl!("Discover / AI Capabilities")).color(TEXT_ON_NAVY).strong().size(14.0));
            });
            ui.add_space(12.0);

            ai_card(app, ui, "eraser-background", tl!("Remove Background"), tl!("Instantly remove background with AI"), "Remove the background");
            ui.add_space(8.0);
            ai_card(app, ui, "wand-sparkles", tl!("Enhance Image Quality"), tl!("Upscale and sharpen details"), "Enhance image quality");
            ui.add_space(8.0);
            ai_card(app, ui, "palette", tl!("Adjust Colors"), tl!("Color grade and color matching"), "Adjust colors");
            ui.add_space(8.0);
            ai_card(app, ui, "type", tl!("Add Text"), tl!("Add stylized text & typography"), "Add text");

            ui.add_space(14.0);
            ui.horizontal(|ui| {
                let icon_box = Rect::from_center_size(ui.cursor().left_center() + egui::vec2(8.0, 0.0), egui::vec2(18.0, 18.0));
                crate::icons::paint(ui, icon_box, "sparkles", 16.0, ACCENT);
                ui.add_space(22.0);
                ui.label(egui::RichText::new(tl!("AI Assistant")).color(TEXT_ON_NAVY).strong().size(14.0));
            });
            ui.add_space(8.0);

            let mut draft = ui.ctx().data_mut(|d| d.get_temp::<String>(home_ai_id()).unwrap_or_default());
            ui.horizontal(|ui| {
                let edit = egui::TextEdit::singleline(&mut draft)
                    .hint_text(egui::RichText::new(tl!("Describe what you want...")).color(TEXT_SOFT))
                    .desired_width((ui.available_width() - 42.0).max(80.0));
                let response = ui.add(edit);
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let sendable = !draft.trim().is_empty();
                let (send_rect, send_click) = ui.allocate_exact_size(egui::vec2(34.0, 30.0), Sense::click());
                let send_click = send_click.on_hover_cursor(egui::CursorIcon::PointingHand);
                ui.painter().rect_filled(send_rect, 8.0, if sendable { ACCENT } else { CARD_LINE });
                crate::icons::paint(ui, send_rect, "send", 14.0, egui::Color32::WHITE);
                if (send_click.clicked() || enter) && sendable {
                    crate::panels::assistant::seed_prompt(ui.ctx(), draft.trim());
                    app.ui.panels.assistant = true;
                    draft.clear();
                }
            });
            ui.ctx().data_mut(|d| d.insert_temp(home_ai_id(), draft));
        });
}

fn home_ai_id() -> egui::Id {
    egui::Id::new("home-ai-draft")
}

/// One AI capability card: icon, title, subtitle; primes the Assistant with the prompt.
fn ai_card(app: &mut FotoditApp, ui: &mut egui::Ui, icon: &str, title: &str, subtitle: &str, prompt: &str) {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 54.0), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if resp.hovered() { CHIP_BG } else { CARD_BG };
    ui.painter().rect_filled(rect, 8.0, fill);
    ui.painter().rect_stroke(rect, 8.0, egui::Stroke::new(1.0, if resp.hovered() { ACCENT } else { CARD_LINE }), egui::StrokeKind::Inside);

    let icon_box = Rect::from_center_size(egui::pos2(rect.left() + 26.0, rect.center().y), egui::vec2(24.0, 24.0));
    crate::icons::paint(ui, icon_box, icon, 20.0, ACCENT);
    ui.painter().text(egui::pos2(rect.left() + 46.0, rect.top() + 14.0), egui::Align2::LEFT_CENTER, title, crate::theme::semibold(13.0), TEXT_ON_NAVY);
    ui.painter().text(egui::pos2(rect.left() + 46.0, rect.top() + 34.0), egui::Align2::LEFT_CENTER, subtitle, crate::theme::medium(11.0), TEXT_SOFT);

    if resp.clicked() {
        crate::panels::assistant::seed_prompt(ui.ctx(), prompt);
        app.ui.panels.assistant = true;
    }
}

fn render_start_actions_row(app: &mut FotoditApp, ui: &mut egui::Ui, avail_w: f32) {
    let btn_w = ((avail_w - 12.0) / 2.0).clamp(0.0, 150.0);

    ui.horizontal(|ui| {
        new_file_button(app, ui, btn_w, 32.0);
        ui.add_space(8.0);
        open_button(app, ui, btn_w, 32.0);
    });
}

/// The accent "New file  +" button.
fn new_file_button(app: &mut FotoditApp, ui: &mut egui::Ui, w: f32, h: f32) {
    let new_label = tl!("New file");
    let (new_rect, new_resp) = ui.allocate_exact_size(egui::vec2(w, h), Sense::click());
    new_resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, new_label.clone()));
    let new_resp = new_resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let new_bg = if new_resp.hovered() { egui::Color32::from_rgb(80, 128, 248) } else { ACCENT };
    ui.painter().rect_filled(new_rect, 8.0, new_bg);
    ui.painter().text(
        new_rect.center(),
        egui::Align2::CENTER_CENTER,
        format!("{new_label}  +"),
        crate::theme::semibold(14.0),
        egui::Color32::WHITE,
    );
    if new_resp.clicked() {
        let fields = app.new_document_fields();
        app.ui.open_dialog(crate::state::DialogKind::NewDocument, fields);
    }
}

/// The dark "Open…" button with the folder icon.
fn open_button(app: &mut FotoditApp, ui: &mut egui::Ui, w: f32, h: f32) {
    let open_label = tl!("Open…");
    let (open_rect, open_resp) = ui.allocate_exact_size(egui::vec2(w, h), Sense::click());
    open_resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, open_label));
    let open_resp = open_resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    ui.painter().rect_filled(open_rect, 8.0, if open_resp.hovered() { PANEL_BG } else { CARD_BG });
    ui.painter().rect_stroke(open_rect, 8.0, egui::Stroke::new(1.0, CARD_LINE), egui::StrokeKind::Inside);
    let icon_box = Rect::from_center_size(egui::pos2(open_rect.center().x - 30.0, open_rect.center().y), egui::vec2(16.0, 16.0));
    crate::icons::paint(ui, icon_box, "folder", 15.0, TEXT_ON_NAVY);
    ui.painter().text(egui::pos2(open_rect.center().x + 8.0, open_rect.center().y), egui::Align2::CENTER_CENTER, open_label, crate::theme::semibold(13.0), TEXT_ON_NAVY);
    if open_resp.clicked() {
        let _ = app.open_dialog_file();
    }
}

fn render_start_rail(app: &mut FotoditApp, ui: &mut egui::Ui, rail_w: f32) {
    ui.add_space(8.0);

    new_file_button(app, ui, rail_w, 46.0);

    ui.add_space(10.0);

    open_button(app, ui, rail_w, 36.0);

    ui.add_space(18.0);

    // Separator line
    let (sep_rect, _) = ui.allocate_exact_size(egui::vec2(rail_w, 1.0), Sense::hover());
    ui.painter().line_segment([sep_rect.min, egui::pos2(sep_rect.max.x, sep_rect.min.y)], egui::Stroke::new(1.0, CARD_LINE));

    ui.add_space(14.0);

    // "Home" nav item (selected)
    let (home_rect, home_resp) = ui.allocate_exact_size(egui::vec2(rail_w, 34.0), Sense::click());
    home_resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, tl!("Home")));
    // Home hides the tab strip and options bar, so this entry is the way back to the open
    // documents (with none open there is nothing behind Home).
    if app.session.documents().is_empty() {
        // Nothing to go back to.
    } else if home_resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
        app.ui.chrome.home = None;
    }
    ui.painter().rect_filled(home_rect, 8.0, CHIP_BG);
    let house_box = Rect::from_center_size(egui::pos2(home_rect.left() + 22.0, home_rect.center().y), egui::vec2(16.0, 16.0));
    crate::icons::paint(ui, house_box, "house", 16.0, TEXT_ON_NAVY);
    ui.painter().text(egui::pos2(home_rect.left() + 38.0, home_rect.center().y), egui::Align2::LEFT_CENTER, tl!("Home"), crate::theme::semibold(13.5), TEXT_ON_NAVY);

    ui.add_space(4.0);

    // "Learn" nav item
    let (learn_rect, learn_resp) = ui.allocate_exact_size(egui::vec2(rail_w, 34.0), Sense::click());
    let learn_resp = learn_resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    learn_resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, tl!("Learn")));
    if learn_resp.hovered() {
        ui.painter().rect_filled(learn_rect, 8.0, CHIP_BG);
    }
    let sparkle_box = Rect::from_center_size(egui::pos2(learn_rect.left() + 22.0, learn_rect.center().y), egui::vec2(16.0, 16.0));
    crate::icons::paint(ui, sparkle_box, "sparkles", 16.0, TEXT_SOFT);
    ui.painter().text(egui::pos2(learn_rect.left() + 38.0, learn_rect.center().y), egui::Align2::LEFT_CENTER, tl!("Learn"), crate::theme::semibold(13.5), TEXT_SOFT);
    if learn_resp.clicked() {
        crate::links::open(app, ui.ctx(), crate::links::TUTORIALS);
    }

    // "Settings" pinned to the bottom of the rail
    let settings_h = 34.0;
    let avail_h = ui.available_height();
    ui.add_space((avail_h - settings_h - 12.0).max(24.0));
    let (set_rect, set_resp) = ui.allocate_exact_size(egui::vec2(rail_w, settings_h), Sense::click());
    let set_resp = set_resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    set_resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, tl!("Settings")));
    if set_resp.hovered() {
        ui.painter().rect_filled(set_rect, 8.0, CHIP_BG);
    }
    let gear_box = Rect::from_center_size(egui::pos2(set_rect.left() + 22.0, set_rect.center().y), egui::vec2(16.0, 16.0));
    crate::icons::paint(ui, gear_box, "compass", 16.0, TEXT_SOFT);
    ui.painter().text(egui::pos2(set_rect.left() + 38.0, set_rect.center().y), egui::Align2::LEFT_CENTER, tl!("Settings"), crate::theme::semibold(13.5), TEXT_SOFT);
    if set_resp.clicked() {
        crate::prefs_ui::open_preferences(app, "general");
    }
}

/// Wayland has no native file drops (winit 0.30, #386), so the hint offers File › Open and paste.
pub(crate) fn start_screen_drop_hint(app: &FotoditApp) -> std::borrow::Cow<'static, str> {
    if app.services.is_wayland {
        crate::i18n::fmt(tl!("Use File › Open, or paste a copied image with {paste}."), &[("paste", &crate::notices::paste_hint(app))]).into()
    } else {
        tl!("Drag & drop images here or Browse files").into()
    }
}

/// Recent files listed on the Home screen.
const HOME_RECENT: usize = 11;
