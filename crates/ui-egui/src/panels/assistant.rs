//! The Assistant dock panel: chat-style natural-language editing. The transcript scrolls;
//! the draft row stays pinned at the bottom like every messaging app.

use super::{FotoditApp, Tokens};

/// One chat exchange: what the user asked, and what the app answered.
#[derive(Clone)]
struct Exchange {
    user: String,
    assistant: String,
}

/// Panel state kept in egui memory for the session: the running transcript and the draft.
#[derive(Clone, Default)]
struct AssistantState {
    transcript: Vec<Exchange>,
    draft: String,
}

fn state_id() -> egui::Id {
    egui::Id::new("assistant-panel-state")
}

pub(crate) fn assistant_panel(app: &mut FotoditApp, ui: &mut egui::Ui) {
    let _ = app;
    let t = Tokens::get(ui.ctx());
    let mut state = ui.ctx().data_mut(|d| d.get_temp::<AssistantState>(state_id()).unwrap_or_default());

    let footer_height = 40.0;
    egui::ScrollArea::vertical()
        .id_salt("assistant-transcript")
        .auto_shrink([false, false])
        .max_height((ui.available_height() - footer_height).max(40.0))
        .show(ui, |ui| {
            if state.transcript.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(tl!("Describe an edit and I'll do it: \"remove the background\", \"make it brighter\", \"crop it square\"."))
                        .color(t.text_dim)
                        .italics(),
                );
            }
            for exchange in &state.transcript {
                bubble(ui, &t, true, &exchange.user);
                bubble(ui, &t, false, &exchange.assistant);
            }
        });

    ui.add_space(6.0);
    let width = ui.available_width();
    let button = egui::Button::new(egui::RichText::new(tl!("Send")).color(egui::Color32::WHITE)).fill(t.accent);
    ui.horizontal(|ui| {
        let edit = egui::TextEdit::singleline(&mut state.draft)
            .hint_text(tl!("Describe an edit…"))
            .desired_width((width - 64.0).max(60.0));
        let response = ui.add(edit);
        let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let sendable = !state.draft.trim().is_empty();
        if (ui.add_enabled(sendable, button).clicked() || enter) && sendable {
            let asked = state.draft.trim().to_string();
            state.draft.clear();
            state.transcript.push(Exchange {
                user: asked,
                assistant: tl!("(Preview) the editing engine plugs in here.").to_string(),
            });
            response.request_focus();
        }
    });

    ui.ctx().data_mut(|d| d.insert_temp(state_id(), state));
}

/// One chat bubble: the user's right-aligned in the accent colour, the assistant's on the card.
fn bubble(ui: &mut egui::Ui, t: &Tokens, user: bool, text: &str) {
    let width = ui.available_width();
    let (fill, color) = if user { (t.accent, egui::Color32::WHITE) } else { (t.field, t.text) };
    let layout = if user {
        egui::Layout::right_to_left(egui::Align::TOP)
    } else {
        egui::Layout::left_to_right(egui::Align::TOP)
    };
    ui.with_layout(layout, |ui| {
        egui::Frame::NONE
            .fill(fill)
            .corner_radius(egui::CornerRadius::same(t.radius_sm as u8))
            .inner_margin(egui::Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.set_max_width(width * 0.85);
                ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
            });
    });
    ui.add_space(4.0);
}
