//! Owner-triggered check of a saved provider key, using each provider's documented key-validation or
//! model-listing call. None of these generates content.
//!
//! Only call [`check_key`] from an explicit user action. It sends one request to the provider and never
//! runs on its own. Providers without a documented validation call report [`KeyCheck::NotTestable`]; that is
//! not a pass. A [`KeyCheck::Valid`] result means the provider accepted that one call, not that every feature
//! or billing state works.
//!
//! Sources (read 2026-11-17):
//! - Cohere `POST /v1/check-api-key` (docs.cohere.com/reference/check-api-key): documented to check that the key is
//!   valid and active, and to answer `{"valid": bool}`.
//! - Gemini `GET /v1beta/models` (ai.google.dev/api/models, models.list) with the `x-goog-api-key` header that
//!   ai.google.dev/api names as required on all requests. Google's docs I read do not say which status an
//!   invalid key gets, so only 200 is read as Valid and every other answer is reported as the status code, not as
//!   "invalid". Google also announced that unrestricted keys stop being accepted, so a refusal does not by itself
//!   mean the key was typed wrong.
//! - OpenRouter is disabled by the owner. This module never builds a request for it.

use std::time::Duration;

use serde_json::Value;

use super::provider_bridge::{HttpTransport, ProviderId};

/// What a key check found. Never carries the key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyCheck {
    /// The provider accepted the key for this validation call.
    Valid,
    /// The provider rejected the key (invalid, revoked or not permitted).
    Rejected,
    /// The provider answered with something this check does not interpret (status code).
    Unexpected(u16),
    /// The request did not complete (network error or timeout). The key was not judged.
    Unreachable,
    /// Nothing was sent. The reason is shown to the user. This is not a pass.
    NotTestable(&'static str),
}

/// A prepared request. The key is only inside `headers`.
#[derive(Clone, PartialEq, Eq)]
pub struct CheckRequest {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(&'static str, String)>,
}

impl std::fmt::Debug for CheckRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckRequest").field("method", &self.method).field("url", &self.url).finish_non_exhaustive()
    }
}

/// The documented validation request for `provider`, or `None` when there is none.
pub fn request_for(provider: ProviderId, key: &str) -> Option<CheckRequest> {
    match provider {
        ProviderId::Gemini => Some(CheckRequest {
            method: "GET",
            url: "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1".into(),
            headers: vec![("x-goog-api-key", key.to_string())],
        }),
        ProviderId::Cohere => Some(CheckRequest {
            method: "POST",
            url: "https://api.cohere.com/v1/check-api-key".into(),
            headers: vec![("Authorization", format!("Bearer {key}"))],
        }),
        ProviderId::OpenRouter | ProviderId::Pollinations | ProviderId::Lajiwa => None,
    }
}

/// Why `provider` has no key check. `None` when it has one.
pub fn not_testable_reason(provider: ProviderId) -> Option<&'static str> {
    match provider {
        ProviderId::Gemini | ProviderId::Cohere => None,
        ProviderId::OpenRouter => Some("OpenRouter is turned off by you. Saving a key does not turn it on, and no request is sent."),
        ProviderId::Pollinations => Some("No key-validation call is documented for this provider that works for every key, so none is sent."),
        ProviderId::Lajiwa => Some("Lajiwa is a self-hosted worker with no documented key-validation call, so none is sent."),
    }
}

/// Reads a provider's answer. `body` is the parsed JSON (or `Null`).
pub fn interpret(provider: ProviderId, status: u16, body: &Value) -> KeyCheck {
    match (provider, status) {
        (ProviderId::Cohere, 200) => match body.get("valid").and_then(Value::as_bool) {
            Some(true) => KeyCheck::Valid,
            Some(false) => KeyCheck::Rejected,
            None => KeyCheck::Unexpected(200),
        },
        (ProviderId::Gemini, 200) => KeyCheck::Valid,
        // Cohere documents 401, 403 and 498 (invalid token) for a bad key.
        (ProviderId::Cohere, 401 | 403 | 498) => KeyCheck::Rejected,
        (_, other) => KeyCheck::Unexpected(other),
    }
}

/// Sends the one validation request. Call only from an explicit user click.
pub fn check_key(provider: ProviderId, key: &str, transport: &dyn HttpTransport) -> KeyCheck {
    let Some(req) = request_for(provider, key) else {
        return KeyCheck::NotTestable(not_testable_reason(provider).unwrap_or("No key check is available for this provider."));
    };
    let headers: Vec<(&str, &str)> = req.headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
    match transport.execute_json(req.method, &req.url, &headers, None, Duration::from_secs(15)) {
        Ok((status, body)) => interpret(provider, status, &body),
        Err(_) => KeyCheck::Unreachable,
    }
}

fn provider_for_slot(slot: &str) -> Option<ProviderId> {
    [ProviderId::Gemini, ProviderId::OpenRouter, ProviderId::Cohere, ProviderId::Pollinations, ProviderId::Lajiwa]
        .into_iter()
        .find(|p| super::provider_bridge::key_slot(*p) == slot)
}

/// For the settings window: whether a check exists for the provider in `slot`, or why not.
pub fn availability(slot: &str) -> Result<(), &'static str> {
    match provider_for_slot(slot) {
        None => Err("Unknown provider."),
        Some(p) => not_testable_reason(p).map_or(Ok(()), Err),
    }
}

/// Runs the check for the key saved in `slot` and returns a message for the user. The key is read from the key
/// store here and never leaves this function except in the request itself. Call only from an explicit click.
pub fn run_saved(slot: &str) -> String {
    let Some(provider) = provider_for_slot(slot) else {
        return "Unknown provider.".into();
    };
    if let Err(reason) = availability(slot) {
        return reason.into();
    }
    let Some(store) = fotodit_keystore::default_store() else {
        return "No saved key is available on this platform.".into();
    };
    let key = match store.get(slot) {
        Ok(Some(k)) => k,
        Ok(None) => return "No key is saved for this provider.".into(),
        Err(_) => return "The saved key could not be read.".into(),
    };
    describe(&check_key(provider, key.expose(), &super::provider_bridge::SystemCurlTransport))
}

/// User-facing text for a result.
pub fn describe(result: &KeyCheck) -> String {
    match result {
        KeyCheck::Valid => "The provider accepted the key for this check. That does not prove every feature or your billing works.".into(),
        KeyCheck::Rejected => "The provider rejected the key.".into(),
        KeyCheck::Unexpected(code) => format!("The provider answered with HTTP {code}. This does not say whether the key is valid."),
        KeyCheck::Unreachable => "The request did not complete (network problem or timeout). The key was not judged.".into(),
        KeyCheck::NotTestable(reason) => (*reason).into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::AiError;
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    struct Mock {
        reply: Result<(u16, Value), ()>,
        seen: Mutex<Vec<(String, String)>>,
    }

    impl HttpTransport for Mock {
        fn execute_json(&self, method: &str, url: &str, _h: &[(&str, &str)], body: Option<&Value>, _t: Duration) -> Result<(u16, Value), AiError> {
            assert!(body.is_none(), "validation calls send no body");
            if let Ok(mut seen) = self.seen.lock() {
                seen.push((method.into(), url.into()));
            }
            self.reply.clone().map_err(|()| AiError::PlanningError("network".into()))
        }

        fn fetch_raw_bytes(&self, _u: &str, _h: &[(&str, &str)], _t: Duration) -> Result<(u16, Vec<u8>), AiError> {
            Err(AiError::PlanningError("unused".into()))
        }
    }

    fn mock(reply: Result<(u16, Value), ()>) -> Mock {
        Mock { reply, seen: Mutex::new(Vec::new()) }
    }

    #[test]
    fn providers_without_a_documented_check_send_nothing() {
        for p in [ProviderId::OpenRouter, ProviderId::Pollinations, ProviderId::Lajiwa] {
            let m = mock(Ok((200, json!({}))));
            assert!(matches!(check_key(p, "k", &m), KeyCheck::NotTestable(r) if !r.is_empty()));
            assert!(m.seen.lock().map(|s| s.is_empty()).unwrap_or(false));
        }
    }

    #[test]
    fn requests_are_non_generative_and_keep_the_key_out_of_debug() {
        for p in [ProviderId::Gemini, ProviderId::Cohere] {
            let req = request_for(p, "secret-value");
            assert!(req.is_some());
            let Some(req) = req else { continue };
            assert!(!req.url.contains("generateContent") && !req.url.contains("completions") && !req.url.contains("chat"));
            assert!(!req.url.contains("secret-value"));
            assert!(!format!("{req:?}").contains("secret-value"));
        }
    }

    #[test]
    fn answers_are_read_conservatively() {
        assert_eq!(interpret(ProviderId::Cohere, 200, &json!({"valid": true})), KeyCheck::Valid);
        assert_eq!(interpret(ProviderId::Cohere, 200, &json!({"valid": false})), KeyCheck::Rejected);
        assert_eq!(interpret(ProviderId::Cohere, 200, &json!({})), KeyCheck::Unexpected(200));
        assert_eq!(interpret(ProviderId::Cohere, 498, &Value::Null), KeyCheck::Rejected);
        assert_eq!(interpret(ProviderId::Gemini, 200, &json!({"models": []})), KeyCheck::Valid);
        // Google does not document the status for a bad key, so none of these is called "Rejected".
        for status in [400, 401, 403, 429] {
            assert_eq!(interpret(ProviderId::Gemini, status, &Value::Null), KeyCheck::Unexpected(status));
        }
    }

    #[test]
    fn a_network_failure_is_not_a_verdict() {
        let m = mock(Err(()));
        assert_eq!(check_key(ProviderId::Gemini, "k", &m), KeyCheck::Unreachable);
    }
}
