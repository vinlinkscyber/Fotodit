//! Multi-Provider AI Bridge & Context-Aware Load Balancer
//!
//! Integrates five AI providers according to the official specifications:
//! 1. Google Gemini (text & image fallback)
//! 2. Cohere (chat & structured JSON)
//! 3. OpenRouter (frontier models with zero-data-retention)
//! 4. Pollinations (primary image generation returning raw bytes)
//! 5. Lajiwa (lightweight self-hosted worker)
//!
//! Subsystems:
//! - [`ContextLoadBalancer`]: Context-profile evaluation, token-length filtering, rate-limit (RPM)
//!   tracking, and weighted round-robin distribution to prevent provider burnout.
//! - [`DualPipelineRouter`]: Dedicated Intelligence & Text pipeline and Image Generation pipeline
//!   with instant cascading failover (Circuit Breaker).
//! - [`ProviderBridgePlanner`]: Implementation of [`super::planner::AiPlanner`] bridging remote LLMs
//!   to Fotodit's native editing command graph, with guaranteed fallback to the offline rule planner.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::command_intelligence::CommandCatalog;
use super::natural_language::EditingIntent;
use super::perception::VisualScene;
use super::planner::{AiPlanner, DeterministicRulePlanner, PlanStep};
use super::{AiContext, AiError, EditingPlan};

/// The five supported external AI providers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ProviderId {
    /// Google Gemini (gemini-3.1-flash-lite / imagen)
    Gemini,
    /// OpenRouter multi-model router (gpt-4.1-mini, claude-sonnet-4.6, etc.)
    OpenRouter,
    /// Cohere Command A+ (command-a-plus-05-2026)
    Cohere,
    /// Pollinations (dedicated raw byte image generation)
    Pollinations,
    /// Lajiwa lightweight self-hosted Cloudflare worker engine
    Lajiwa,
}

impl ProviderId {
    /// Display name of the provider.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemini => "Google Gemini",
            Self::OpenRouter => "OpenRouter",
            Self::Cohere => "Cohere",
            Self::Pollinations => "Pollinations",
            Self::Lajiwa => "Lajiwa",
        }
    }

    /// Base balancing weight for text and intelligence tasks (higher = more capable).
    pub fn base_text_weight(&self) -> u32 {
        match self {
            Self::Gemini => 40,
            Self::OpenRouter => 30,
            Self::Cohere => 20,
            Self::Lajiwa => 10,
            Self::Pollinations => 0, // Dedicated to image generation
        }
    }

    /// Maximum Unicode character length tolerated by the provider (if capped).
    pub fn max_character_limit(&self) -> Option<usize> {
        match self {
            Self::Lajiwa => Some(2000), // Enforces 413 message_too_long beyond 2,000 chars
            _ => None,
        }
    }

    /// Maximum requests per minute ceiling before load balancer steers traffic away.
    pub fn rpm_ceiling(&self) -> u32 {
        match self {
            Self::Lajiwa => 20, // Free tier / key limit of 20 req/min
            Self::Gemini => 60,
            Self::Cohere => 40,
            Self::OpenRouter => 60,
            Self::Pollinations => 30,
        }
    }
}

/// Configuration credentials and options for the AI Provider Bridge.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub gemini_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub cohere_api_key: Option<String>,
    pub pollinations_api_key: Option<String>,
    pub lajiwa_api_key: Option<String>,

    pub gemini_model: Option<String>,
    pub openrouter_model: Option<String>,
    pub cohere_model: Option<String>,
    pub pollinations_image_model: Option<String>,

    pub lajiwa_base_url: Option<String>,
    pub client_timeout_secs: Option<u64>,
    pub openrouter_enabled: bool,
}

fn get_config_var(key: &str) -> Option<String> {
    if let Ok(val) = std::env::var(key) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    for filename in [".env", "fotodit.env", "ai_keys.env"] {
        if let Ok(content) = std::fs::read_to_string(filename) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with('#') || !trimmed.contains('=') {
                    continue;
                }
                if let Some((k, v)) = trimmed.split_once('=') {
                    if k.trim() == key {
                        let clean = v.trim().trim_matches('"').trim_matches('\'').to_string();
                        if !clean.is_empty() {
                            return Some(clean);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Storage slot (and file key) of a provider's saved key in the local protected key store.
pub fn key_slot(provider: ProviderId) -> &'static str {
    match provider {
        ProviderId::Gemini => "gemini",
        ProviderId::OpenRouter => "openrouter",
        ProviderId::Cohere => "cohere",
        ProviderId::Pollinations => "pollinations",
        ProviderId::Lajiwa => "lajiwa",
    }
}

/// Resolves a provider key by the credential source the user chose on the AI providers screen:
/// - saved key: only that key is used. If it cannot be read (damaged file, another Windows user) the
///   provider has no key; nothing else stands in, so a different account is never picked silently.
/// - removed on purpose, or no choice made yet: no key, from any source.
/// - the user chose the environment variable / `.env` route: the process environment, then `.env` files.
/// - no key store on this platform: the environment, then `.env` files, because there is nothing to choose from.
/// A compile-time ("baked") key is never used here on any platform.
fn get_secret_var(name: &str, provider: ProviderId) -> Option<String> {
    let external = || {
        if let Ok(val) = std::env::var(name) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        get_config_var(name)
    };
    let Some(store) = fotodit_keystore::default_store() else {
        return external();
    };
    match store.source(key_slot(provider)) {
        Ok(fotodit_keystore::Source::Saved) => {
            let secret = store.get(key_slot(provider)).ok()??;
            let value = secret.expose().trim().to_string();
            (!value.is_empty()).then_some(value)
        }
        Ok(fotodit_keystore::Source::External) => external(),
        // Removed, not chosen yet, or an unreadable store file: fail closed.
        Ok(fotodit_keystore::Source::Disabled | fotodit_keystore::Source::Unconfigured) | Err(_) => None,
    }
}

impl ProviderConfig {
    /// Create a new configuration populated from environment variables or .env file.
    pub fn from_env() -> Self {
        Self {
            gemini_api_key: get_secret_var("GEMINI_API_KEY", ProviderId::Gemini),
            openrouter_api_key: get_secret_var("OPENROUTER_API_KEY", ProviderId::OpenRouter),
            cohere_api_key: get_secret_var("COHERE_API_KEY", ProviderId::Cohere),
            pollinations_api_key: get_secret_var("POLLINATIONS_API_KEY", ProviderId::Pollinations),
            lajiwa_api_key: get_secret_var("LAJIWA_API_KEY", ProviderId::Lajiwa),
            gemini_model: get_config_var("GEMINI_MODEL").or_else(|| Some("gemini-3.1-flash-lite".into())),
            openrouter_model: get_config_var("OPENROUTER_MODEL").or_else(|| Some("openai/gpt-4.1-mini".into())),
            cohere_model: get_config_var("COHERE_MODEL").or_else(|| Some("command-a-plus-05-2026".into())),
            pollinations_image_model: get_config_var("POLLINATIONS_IMAGE_MODEL").or_else(|| Some("tongyi-mai/z-image-turbo".into())),
            lajiwa_base_url: get_config_var("LAJIWA_BASE_URL").or_else(|| Some("https://lajiwa.roydeburgs.workers.dev".into())),
            client_timeout_secs: Some(25),
            openrouter_enabled: get_config_var("OPENROUTER_ENABLED").as_deref() == Some("1"),
        }
    }

    /// Whether an API key is available for the given provider.
    pub fn has_key(&self, provider: ProviderId) -> bool {
        match provider {
            ProviderId::Gemini => self.gemini_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            ProviderId::OpenRouter => {
                // OpenRouter disabled for now per user instruction.
                // Can be re-enabled anytime by setting OPENROUTER_ENABLED=1
                self.openrouter_enabled && self.openrouter_api_key.as_deref().is_some_and(|k| !k.is_empty())
            }
            ProviderId::Cohere => self.cohere_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            ProviderId::Pollinations => self.pollinations_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            ProviderId::Lajiwa => self.lajiwa_api_key.as_deref().is_some_and(|k| !k.is_empty()),
        }
    }
}

/// Dynamic health, active traffic, and rate-limit tracking for a provider.
#[derive(Debug)]
struct ProviderHealth {
    active_requests: usize,
    timestamps: VecDeque<Instant>,
    cooldown_until: Option<Instant>,
    total_calls: u64,
    total_errors: u64,
}

impl Default for ProviderHealth {
    fn default() -> Self {
        Self {
            active_requests: 0,
            timestamps: VecDeque::new(),
            cooldown_until: None,
            total_calls: 0,
            total_errors: 0,
        }
    }
}

impl ProviderHealth {
    fn clean_old_timestamps(&mut self, now: Instant) {
        let one_min_ago = now.checked_sub(Duration::from_secs(60)).unwrap_or(now);
        while let Some(&front) = self.timestamps.front() {
            if front < one_min_ago {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }

    fn current_rpm(&mut self, now: Instant) -> u32 {
        self.clean_old_timestamps(now);
        self.timestamps.len() as u32
    }

    fn is_in_cooldown(&self, now: Instant) -> bool {
        if let Some(until) = self.cooldown_until {
            now < until
        } else {
            false
        }
    }

    fn record_request(&mut self, now: Instant) {
        self.active_requests = self.active_requests.saturating_add(1);
        self.total_calls = self.total_calls.saturating_add(1);
        self.timestamps.push_back(now);
    }

    fn finish_request(&mut self) {
        self.active_requests = self.active_requests.saturating_sub(1);
    }

    fn trigger_cooldown(&mut self, now: Instant, duration: Duration) {
        self.total_errors = self.total_errors.saturating_add(1);
        let until = now.checked_add(duration).unwrap_or(now);
        self.cooldown_until = Some(until);
    }
}

/// Context analysis profile for an incoming request.
#[derive(Clone, Debug)]
pub struct ContextProfile {
    /// Length of the prompt in Unicode characters.
    pub char_length: usize,
    /// Estimated prior conversation turns.
    pub history_turns: usize,
    /// Whether strict JSON output is required.
    pub requires_json: bool,
    /// Whether this is an image synthesis request.
    pub is_image: bool,
}

/// Context-Aware Load Balancer distributing work across available AI providers.
#[derive(Debug)]
pub struct ContextLoadBalancer {
    config: ProviderConfig,
    health: Mutex<HashMap<ProviderId, ProviderHealth>>,
}

impl ContextLoadBalancer {
    /// Create a new load balancer with configuration.
    pub fn new(config: ProviderConfig) -> Self {
        let mut map = HashMap::new();
        map.insert(ProviderId::Gemini, ProviderHealth::default());
        map.insert(ProviderId::OpenRouter, ProviderHealth::default());
        map.insert(ProviderId::Cohere, ProviderHealth::default());
        map.insert(ProviderId::Pollinations, ProviderHealth::default());
        map.insert(ProviderId::Lajiwa, ProviderHealth::default());

        Self {
            config,
            health: Mutex::new(map),
        }
    }

    /// Access provider configuration.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Select an ordered cascade of providers for intelligence/text tasks based on context.
    pub fn select_text_cascade(&self, context: &ContextProfile) -> Vec<ProviderId> {
        let now = Instant::now();
        let mut health_guard = match self.health.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };

        let text_candidates = [
            ProviderId::Gemini,
            ProviderId::OpenRouter,
            ProviderId::Cohere,
            ProviderId::Lajiwa,
        ];

        let mut scored_candidates: Vec<(ProviderId, i32)> = Vec::new();

        for &id in &text_candidates {
            if !self.config.has_key(id) {
                continue;
            }

            // Context filter: Check Unicode character limit (Lajiwa 2000-char cap)
            if let Some(limit) = id.max_character_limit() {
                if context.char_length > limit {
                    continue; // Skip provider because context exceeds its hard capacity
                }
            }

            let health = health_guard.entry(id).or_default();
            let in_cooldown = health.is_in_cooldown(now);
            let current_rpm = health.current_rpm(now);
            let rpm_ceiling = id.rpm_ceiling();

            let mut score = id.base_text_weight() as i32;

            // Load-balancing penalties
            let active_penalty = (health.active_requests as i32) * 5;
            score -= active_penalty;

            if current_rpm >= rpm_ceiling {
                score -= 30; // Approaching RPM ceiling: steer traffic away
            }

            if in_cooldown {
                score -= 100; // Deprioritize providers in cooldown
            }

            scored_candidates.push((id, score));
        }

        // Sort descending by score
        scored_candidates.sort_by(|a, b| b.1.cmp(&a.1));
        scored_candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Select ordered providers for image generation (Pollinations primary, Gemini fallback).
    pub fn select_image_cascade(&self) -> Vec<ProviderId> {
        let now = Instant::now();
        let health_guard = match self.health.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };

        let mut cascade = Vec::new();

        // 1. Primary: Pollinations
        if self.config.has_key(ProviderId::Pollinations) {
            let polli_cooldown = health_guard
                .get(&ProviderId::Pollinations)
                .is_some_and(|h| h.is_in_cooldown(now));
            if !polli_cooldown {
                cascade.push(ProviderId::Pollinations);
            }
        }

        // 2. Fallback: Google Gemini
        if self.config.has_key(ProviderId::Gemini) {
            cascade.push(ProviderId::Gemini);
        }

        // Append Pollinations at end even if cooled down as last resort
        if self.config.has_key(ProviderId::Pollinations) && !cascade.contains(&ProviderId::Pollinations) {
            cascade.push(ProviderId::Pollinations);
        }

        cascade
    }

    /// Mark an active request starting for a provider.
    pub fn record_start(&self, provider: ProviderId) {
        if let Ok(mut guard) = self.health.lock() {
            let health = guard.entry(provider).or_default();
            health.record_request(Instant::now());
        }
    }

    /// Mark a request finishing successfully.
    pub fn record_success(&self, provider: ProviderId) {
        if let Ok(mut guard) = self.health.lock() {
            let health = guard.entry(provider).or_default();
            health.finish_request();
            health.cooldown_until = None; // Clear cooldown on success
        }
    }

    /// Record a failure with potential cooldown on rate-limits (429) or server errors (500/503).
    pub fn record_failure(&self, provider: ProviderId, status_code: Option<u16>) {
        if let Ok(mut guard) = self.health.lock() {
            let health = guard.entry(provider).or_default();
            health.finish_request();

            let now = Instant::now();
            match status_code {
                Some(429) => {
                    // Rate limited: 60-second cooldown
                    health.trigger_cooldown(now, Duration::from_secs(60));
                }
                Some(402) => {
                    // Budget/credits exhausted: 300-second cooldown
                    health.trigger_cooldown(now, Duration::from_secs(300));
                }
                Some(500) | Some(502) | Some(503) => {
                    // Upstream server error: 20-second cooldown
                    health.trigger_cooldown(now, Duration::from_secs(20));
                }
                _ => {
                    // General error: 5-second cooldown
                    health.trigger_cooldown(now, Duration::from_secs(5));
                }
            }
        }
    }
}

/// Abstract HTTP transport trait enabling zero-dependency native execution and in-memory test mocks.
pub trait HttpTransport: Send + Sync {
    /// Perform an HTTP request returning status code and response body.
    fn execute_json(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&Value>,
        timeout: Duration,
    ) -> Result<(u16, Value), AiError>;

    /// Perform an HTTP GET request returning raw image bytes.
    fn fetch_raw_bytes(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        timeout: Duration,
    ) -> Result<(u16, Vec<u8>), AiError>;
}

/// Quotes a value for a curl config file: `\\` and `"` are escaped, control characters are refused.
fn curl_config_quote(value: &str) -> std::io::Result<String> {
    if value.chars().any(char::is_control) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "control character in request"));
    }
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(out)
}

/// Renders request headers and an optional body as a curl config file. They are sent to curl on stdin (`-K -`)
/// rather than as `-H` / `-d` arguments, because command-line arguments are readable by other programs running
/// as the same user (process lists, Task Manager): one header carries an API key, and the body carries the
/// prompt. A compact JSON body has no raw control characters, so refusing them loses nothing.
fn curl_config(headers: &[(&str, &str)], body: Option<&str>) -> std::io::Result<String> {
    let mut out = String::new();
    for &(k, v) in headers {
        if k.is_empty() || k.chars().any(|c| c.is_control() || c == ':') {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid HTTP header"));
        }
        out.push_str("header = ");
        out.push_str(&curl_config_quote(&format!("{k}: {v}"))?);
        out.push('\n');
    }
    if let Some(b) = body {
        out.push_str("data = ");
        out.push_str(&curl_config_quote(b)?);
        out.push('\n');
    }
    Ok(out)
}

/// Runs a prepared curl command with `headers` and `body` supplied through a stdin config.
fn run_curl(mut cmd: std::process::Command, headers: &[(&str, &str)], body: Option<&str>) -> std::io::Result<std::process::Output> {
    use std::io::Write as _;
    use std::process::Stdio;
    let config = curl_config(headers, body)?;
    cmd.arg("-K").arg("-");
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(config.as_bytes())?;
    }
    child.wait_with_output()
}

/// Production HTTP transport utilizing native system curl with TLS support.
pub struct SystemCurlTransport;

impl HttpTransport for SystemCurlTransport {
    fn execute_json(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&Value>,
        timeout: Duration,
    ) -> Result<(u16, Value), AiError> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-s").arg("-S"); // Silent but show errors
        cmd.arg("-X").arg(method);
        cmd.arg(url);
        cmd.arg("--max-time").arg(format!("{}", timeout.as_secs().max(1)));
        cmd.arg("-w").arg("\n%{http_code}"); // Append status code at end

        let mut config_headers: Vec<(&str, &str)> = headers.to_vec();
        let body_text = body.map(Value::to_string);
        if body_text.is_some() {
            config_headers.push(("Content-Type", "application/json"));
        }

        let output = run_curl(cmd, &config_headers, body_text.as_deref()).map_err(|e| AiError::PlanningError(format!("curl execution failed: {e}")))?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut lines = stdout.rsplitn(2, '\n');
        let code_str = lines.next().unwrap_or("0").trim();
        let body_str = lines.next().unwrap_or("");

        let status_code: u16 = code_str.parse().unwrap_or(500);
        let parsed_json: Value = serde_json::from_str(body_str).unwrap_or(json!({ "raw": body_str }));

        Ok((status_code, parsed_json))
    }

    fn fetch_raw_bytes(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        timeout: Duration,
    ) -> Result<(u16, Vec<u8>), AiError> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-s").arg("-S");
        cmd.arg("-X").arg("GET");
        cmd.arg(url);
        cmd.arg("--max-time").arg(format!("{}", timeout.as_secs().max(1)));

        // Keep the HTTP status separate from the arbitrary binary response body.
        cmd.arg("-w").arg("\n%{http_code}");

        let output = run_curl(cmd, headers, None).map_err(|e| AiError::PlanningError(format!("curl fetch failed: {e}")))?;
        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(AiError::PlanningError(format!("curl image request failed: {err_msg}")));
        }

        split_raw_http_response(output.stdout)
    }
}

// curl appends exactly one newline and three status digits after the binary body.
// Split from the end so embedded newlines, NULs and non-UTF8 image bytes survive.
fn split_raw_http_response(mut output: Vec<u8>) -> Result<(u16, Vec<u8>), AiError> {
    let invalid = || AiError::PlanningError("Missing or invalid curl image HTTP status".into());
    let suffix = output.len().checked_sub(4).ok_or_else(invalid)?;
    if output[suffix] != b'\n' {
        return Err(invalid());
    }
    let digits = &output[suffix + 1..];
    if !digits.iter().all(u8::is_ascii_digit) {
        return Err(invalid());
    }
    let status = u16::from(digits[0] - b'0') * 100
        + u16::from(digits[1] - b'0') * 10
        + u16::from(digits[2] - b'0');
    if !(100..=599).contains(&status) {
        return Err(invalid());
    }
    output.truncate(suffix);
    Ok((status, output))
}

// Private image-attempt metadata: HTTP status is carried separately from the
// human-readable error, never guessed from error text. Transport, schema and
// decoding errors have no HTTP failure status even if their messages contain
// status-like digits. Public bridge/transport/error interfaces stay unchanged.
struct ImageAttemptError {
    error: AiError,
    status: Option<u16>,
}

impl From<AiError> for ImageAttemptError {
    fn from(error: AiError) -> Self {
        Self { error, status: None }
    }
}

/// The unified AI Provider Bridge orchestrating load balancing and dual-pipeline fallback.
pub struct AiProviderBridge {
    balancer: ContextLoadBalancer,
    transport: Box<dyn HttpTransport>,
    offline_fallback: DeterministicRulePlanner,
}

impl AiProviderBridge {
    /// Create a new provider bridge with system curl transport.
    pub fn new(config: ProviderConfig, catalog: CommandCatalog) -> Self {
        Self {
            balancer: ContextLoadBalancer::new(config),
            transport: Box::new(SystemCurlTransport),
            offline_fallback: DeterministicRulePlanner::new(catalog),
        }
    }

    /// Create a provider bridge with a custom transport (for unit testing / mocks).
    pub fn with_transport(
        config: ProviderConfig,
        catalog: CommandCatalog,
        transport: Box<dyn HttpTransport>,
    ) -> Self {
        Self {
            balancer: ContextLoadBalancer::new(config),
            transport,
            offline_fallback: DeterministicRulePlanner::new(catalog),
        }
    }

    /// Access load balancer.
    pub fn balancer(&self) -> &ContextLoadBalancer {
        &self.balancer
    }

    /// Generate an editing plan using context-aware load balancing with automatic cascade fallback.
    pub fn plan_editing(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        context: &AiContext,
    ) -> Result<EditingPlan, AiError> {
        let prompt_text = &intent.raw_prompt;
        let profile = ContextProfile {
            char_length: prompt_text.chars().count(),
            history_turns: context.prompt_history.len(),
            requires_json: true,
            is_image: false,
        };

        let cascade = self.balancer.select_text_cascade(&profile);
        let timeout = Duration::from_secs(self.balancer.config().client_timeout_secs.unwrap_or(25));

        for &provider in &cascade {
            self.balancer.record_start(provider);
            match self.execute_provider_plan(provider, intent, scene, context, timeout) {
                Ok(plan) => {
                    self.balancer.record_success(provider);
                    return Ok(plan);
                }
                Err(err) => {
                    log::warn!("Provider {} failed: {err}; cascading to next candidate", provider.name());
                    let status = match &err {
                        AiError::PlanningError(msg) if msg.contains("429") => Some(429),
                        AiError::PlanningError(msg) if msg.contains("402") => Some(402),
                        AiError::PlanningError(msg) if msg.contains("500") => Some(500),
                        AiError::PlanningError(msg) if msg.contains("503") => Some(503),
                        _ => None,
                    };
                    self.balancer.record_failure(provider, status);
                }
            }
        }

        // Tier 5 / Ultimate Fallback: Offline Deterministic Engine
        log::info!("All remote AI providers exhausted; falling back to local deterministic rule planner");
        self.offline_fallback.plan(intent, scene, context)
    }

    /// Generate an image using Pipeline 2 (Pollinations primary -> Gemini fallback).
    pub fn generate_image(&self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>, AiError> {
        let cascade = self.balancer.select_image_cascade();
        let timeout = Duration::from_secs(self.balancer.config().client_timeout_secs.unwrap_or(30));

        for &provider in &cascade {
            self.balancer.record_start(provider);
            match provider {
                ProviderId::Pollinations => {
                    match self.execute_pollinations_image(prompt, width, height, timeout) {
                        Ok(bytes) => {
                            self.balancer.record_success(provider);
                            return Ok(bytes);
                        }
                        Err(e) => {
                            log::warn!("Pollinations image failed: {}; cascading to Gemini fallback", e.error);
                            self.balancer.record_failure(provider, e.status);
                        }
                    }
                }
                ProviderId::Gemini => {
                    match self.execute_gemini_image(prompt, timeout) {
                        Ok(bytes) => {
                            self.balancer.record_success(provider);
                            return Ok(bytes);
                        }
                        Err(e) => {
                            log::warn!("Gemini image fallback failed: {}", e.error);
                            self.balancer.record_failure(provider, e.status);
                        }
                    }
                }
                _ => {}
            }
        }

        Err(AiError::PlanningError("All image generation providers failed".into()))
    }

    fn execute_provider_plan(
        &self,
        provider: ProviderId,
        intent: &EditingIntent,
        scene: &VisualScene,
        context: &AiContext,
        timeout: Duration,
    ) -> Result<EditingPlan, AiError> {
        match provider {
            ProviderId::Gemini => self.call_gemini_plan(intent, scene, context, timeout),
            ProviderId::OpenRouter => self.call_openrouter_plan(intent, scene, context, timeout),
            ProviderId::Cohere => self.call_cohere_plan(intent, scene, context, timeout),
            ProviderId::Lajiwa => self.call_lajiwa_plan(intent, scene, context, timeout),
            ProviderId::Pollinations => Err(AiError::PlanningError("Pollinations is reserved for imaging".into())),
        }
    }

    // 1. Google Gemini HTTP Plan Caller
    fn call_gemini_plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        _context: &AiContext,
        timeout: Duration,
    ) -> Result<EditingPlan, AiError> {
        let key = self.balancer.config().gemini_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing Gemini API key".into()))?;
        let model = self.balancer.config().gemini_model.as_deref().unwrap_or("gemini-3.1-flash-lite");
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent");

        let system_instruction = "You are Fotodit's AI Editing Planner. Return strict JSON with {title, rationale, steps: [{command, params, description}]}.";
        let user_prompt = format!(
            "User query: '{}'. Document dimensions: {}x{}. Active layers: {}. Plan the edits.",
            intent.raw_prompt, scene.width, scene.height, scene.layers.len()
        );

        let body = json!({
            "contents": [{
                "parts": [{ "text": format!("{system_instruction}\n{user_prompt}") }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        });

        let headers = [("x-goog-api-key", key)];
        let (status, resp) = self.transport.execute_json("POST", &url, &headers, Some(&body), timeout)?;

        if status != 200 {
            let err_msg = resp.get("error").and_then(|e| e.get("message")).and_then(Value::as_str).unwrap_or("Gemini HTTP error");
            return Err(AiError::PlanningError(format!("Gemini {status}: {err_msg}")));
        }

        let text = resp.get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(Value::as_str)
            .ok_or_else(|| AiError::PlanningError("Empty or malformed Gemini response".into()))?;

        self.parse_plan_json(text, intent)
    }

    // 2. OpenRouter HTTP Plan Caller
    fn call_openrouter_plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        _context: &AiContext,
        timeout: Duration,
    ) -> Result<EditingPlan, AiError> {
        let key = self.balancer.config().openrouter_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing OpenRouter API key".into()))?;
        let model = self.balancer.config().openrouter_model.as_deref().unwrap_or("openai/gpt-4.1-mini");
        let url = "https://openrouter.ai/api/v1/chat/completions";

        let prompt = format!(
            "Generate JSON for editing plan: User intent: '{}', Canvas: {}x{}. Format: {{title, rationale, steps:[{{command, params, description}}]}}",
            intent.raw_prompt, scene.width, scene.height
        );

        let auth_header = format!("Bearer {key}");
        let headers = [("Authorization", auth_header.as_str())];
        let body = json!({
            "model": model,
            "messages": [
                { "role": "system", "content": "You are Fotodit creative planner. Output JSON." },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "provider": { "data_collection": "deny", "zdr": true }
        });

        let (status, resp) = self.transport.execute_json("POST", url, &headers, Some(&body), timeout)?;
        if status != 200 {
            let err_msg = resp.get("error").and_then(|e| e.get("message")).and_then(Value::as_str).unwrap_or("OpenRouter error");
            return Err(AiError::PlanningError(format!("OpenRouter {status}: {err_msg}")));
        }

        let content = resp.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| AiError::PlanningError("Empty OpenRouter choice".into()))?;

        self.parse_plan_json(content, intent)
    }

    // 3. Cohere HTTP Plan Caller
    fn call_cohere_plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        _context: &AiContext,
        timeout: Duration,
    ) -> Result<EditingPlan, AiError> {
        let key = self.balancer.config().cohere_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing Cohere API key".into()))?;
        let model = self.balancer.config().cohere_model.as_deref().unwrap_or("command-a-plus-05-2026");
        let url = "https://api.cohere.com/v2/chat";

        let auth_header = format!("Bearer {key}");
        let headers = [("Authorization", auth_header.as_str())];
        let body = json!({
            "model": model,
            "messages": [
                { "role": "system", "content": "Generate JSON: Fotodit editing plan." },
                { "role": "user", "content": format!("Plan edit for '{}' on {}x{} canvas.", intent.raw_prompt, scene.width, scene.height) }
            ],
            "response_format": { "type": "json_object" },
            "temperature": 0.2,
            "max_tokens": 800
        });

        let (status, resp) = self.transport.execute_json("POST", url, &headers, Some(&body), timeout)?;
        if status != 200 {
            let err_msg = resp.get("message").and_then(Value::as_str).unwrap_or("Cohere error");
            return Err(AiError::PlanningError(format!("Cohere {status}: {err_msg}")));
        }

        let content = resp.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .ok_or_else(|| AiError::PlanningError("Empty Cohere content".into()))?;

        self.parse_plan_json(content, intent)
    }

    // 4. Lajiwa HTTP Plan Caller
    fn call_lajiwa_plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        _context: &AiContext,
        timeout: Duration,
    ) -> Result<EditingPlan, AiError> {
        let key = self.balancer.config().lajiwa_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing Lajiwa API key".into()))?;
        let base_url = self.balancer.config().lajiwa_base_url.as_deref().unwrap_or("https://lajiwa.roydeburgs.workers.dev");
        let url = format!("{base_url}/api/chat");

        let auth_header = format!("Bearer {key}");
        let headers = [("Authorization", auth_header.as_str())];
        let body = json!({
            "message": format!("JSON plan: '{}' for {}x{} canvas", intent.raw_prompt, scene.width, scene.height)
        });

        let (status, resp) = self.transport.execute_json("POST", &url, &headers, Some(&body), timeout)?;
        if status != 200 {
            let err_msg = resp.get("error").and_then(Value::as_str).unwrap_or("Lajiwa error");
            return Err(AiError::PlanningError(format!("Lajiwa {status}: {err_msg}")));
        }

        let reply = resp.get("reply").and_then(Value::as_str).ok_or_else(|| AiError::PlanningError("Empty Lajiwa reply".into()))?;
        // Strip trailing source comments if any
        let clean_reply = reply.split("<!--LAJIWA_SOURCES:").next().unwrap_or(reply);
        self.parse_plan_json(clean_reply, intent)
    }

    // Pollinations Image Fetcher (Raw image bytes)
    fn execute_pollinations_image(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        timeout: Duration,
    ) -> Result<Vec<u8>, ImageAttemptError> {
        let key = self.balancer.config().pollinations_api_key.as_deref().unwrap_or("");
        let model = self.balancer.config().pollinations_image_model.as_deref().unwrap_or("tongyi-mai/z-image-turbo");
        let encoded_prompt = url_encode(prompt);
        let url = format!(
            "https://gen.pollinations.ai/image/{encoded_prompt}?model={model}&width={width}&height={height}&safe=true"
        );

        let auth_header = format!("Bearer {key}");
        let headers = if !key.is_empty() {
            vec![("Authorization", auth_header.as_str())]
        } else {
            vec![]
        };

        let (status, bytes) = self.transport.fetch_raw_bytes(&url, &headers, timeout)?;
        if status == 200 && !bytes.is_empty() {
            Ok(bytes)
        } else {
            Err(ImageAttemptError {
                error: AiError::PlanningError(format!("Pollinations image failed with status {status}")),
                status: (status != 200).then_some(status),
            })
        }
    }

    // Google Gemini Visual Image Synthesis Fallback
    fn execute_gemini_image(&self, prompt: &str, timeout: Duration) -> Result<Vec<u8>, ImageAttemptError> {
        let key = self.balancer.config().gemini_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing Gemini key for image fallback".into()))?;
        let url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent";

        let headers = [("x-goog-api-key", key)];
        let body = json!({
            "contents": [{ "parts": [{ "text": prompt }] }],
            "generationConfig": { "responseModalities": ["IMAGE"] }
        });

        let (status, resp) = self.transport.execute_json("POST", url, &headers, Some(&body), timeout)?;
        if status != 200 {
            return Err(ImageAttemptError {
                error: AiError::PlanningError(format!("Gemini image fallback status {status}")),
                status: Some(status),
            });
        }

        let b64_bytes = resp.pointer("/candidates/0/content/parts/0/inlineData/data")
            .and_then(Value::as_str)
            .ok_or_else(|| AiError::PlanningError("Missing b64 image data in Gemini response".into()))?;

        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(b64_bytes)
            .map_err(|e| ImageAttemptError::from(AiError::PlanningError(format!("Base64 decode failed: {e}"))))
    }

    // Helper: Parses JSON plan text with markdown fence stripping
    fn parse_plan_json(&self, raw_text: &str, intent: &EditingIntent) -> Result<EditingPlan, AiError> {
        let cleaned = strip_markdown_fences(raw_text);
        let val: Value = serde_json::from_str(&cleaned).map_err(|e| AiError::PlanningError(format!("Plan JSON parse error: {e}")))?;

        let title = val.get("title").and_then(Value::as_str).unwrap_or("AI Editing Plan").to_string();
        let rationale = val.get("rationale").and_then(Value::as_str).unwrap_or("Custom creative edits").to_string();

        let mut steps = Vec::new();
        if let Some(arr) = val.get("steps").and_then(Value::as_array) {
            for item in arr {
                if let Some(cmd) = item.get("command").and_then(Value::as_str) {
                    let params = item.get("params").cloned().unwrap_or(json!({}));
                    let desc = item.get("description").and_then(Value::as_str).unwrap_or("Execute step").to_string();
                    steps.push(PlanStep {
                        command: cmd.to_string(),
                        params,
                        description: desc,
                    });
                }
            }
        }

        if steps.is_empty() {
            return Err(AiError::PlanningError("Generated plan contained no executable steps".into()));
        }

        Ok(EditingPlan {
            title,
            rationale,
            category: intent.category,
            steps,
            creates_document: false,
            target_size: None,
        })
    }
}

impl AiPlanner for AiProviderBridge {
    fn plan(
        &self,
        intent: &EditingIntent,
        scene: &VisualScene,
        context: &AiContext,
    ) -> Result<EditingPlan, AiError> {
        self.plan_editing(intent, scene, context)
    }
}

/// URL encode string helper.
fn url_encode(input: &str) -> String {
    let mut out = String::new();
    for b in input.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
            out.push(b as char);
        } else if b == b' ' {
            out.push_str("%20");
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

/// Strips ```` ```json ... ``` ```` markdown fences if present in LLM text.
fn strip_markdown_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(stripped) = trimmed.strip_prefix("```json") {
        if let Some(end) = stripped.strip_suffix("```") {
            return end.trim().to_string();
        }
    }
    if let Some(stripped) = trimmed.strip_prefix("```") {
        if let Some(end) = stripped.strip_suffix("```") {
            return end.trim().to_string();
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn provider_slots_match_the_settings_screen_list() {
        let ids = [ProviderId::Gemini, ProviderId::OpenRouter, ProviderId::Cohere, ProviderId::Pollinations, ProviderId::Lajiwa];
        let slots: Vec<&str> = ids.iter().map(|p| key_slot(*p)).collect();
        let listed: Vec<&str> = fotodit_keystore::PROVIDERS.iter().map(|p| p.slot).collect();
        assert_eq!(slots, listed);
    }

    #[test]
    fn curl_headers_and_body_go_through_a_config_not_the_command_line() -> std::io::Result<()> {
        let cfg = curl_config(&[("Authorization", "Bearer abc\"def\\ghi"), ("X-Plain", "v")], None)?;
        assert_eq!(cfg, "header = \"Authorization: Bearer abc\\\"def\\\\ghi\"\nheader = \"X-Plain: v\"\n");
        assert!(curl_config(&[("X", "a\r\nInjected: 1")], None).is_err());
        assert!(curl_config(&[("Bad:Name", "v")], None).is_err());
        assert!(curl_config(&[("", "v")], None).is_err());
        // A prompt body with quotes, backslashes and an escaped newline stays one config line.
        let body = serde_json::json!({"prompt": "say \"hi\"\nthen \\ done"}).to_string();
        let cfg = curl_config(&[], Some(&body))?;
        assert_eq!(cfg.lines().count(), 1);
        assert!(cfg.starts_with("data = \"") && cfg.ends_with("\"\n"));
        assert!(curl_config(&[], Some("raw\nnewline")).is_err());
        Ok(())
    }

    #[test]
    fn raw_http_status_preserves_binary_body_and_error_statuses() -> Result<(), AiError> {
        for status in [200, 401, 402, 429, 500, 503] {
            let body = vec![0, 255, 137, b'\n', b'2', b'0', b'0', b'\n'];
            let mut output = body.clone();
            output.extend_from_slice(format!("\n{status}").as_bytes());
            let (actual_status, actual_body) = split_raw_http_response(output)?;
            assert_eq!(actual_status, status);
            assert_eq!(actual_body, body);
        }
        Ok(())
    }

    #[test]
    fn raw_http_status_preserves_empty_body() -> Result<(), AiError> {
        let (status, body) = split_raw_http_response(b"\n200".to_vec())?;
        assert_eq!(status, 200);
        assert!(body.is_empty());
        Ok(())
    }

    #[test]
    fn raw_http_status_rejects_missing_or_invalid_suffix() {
        for output in [
            b"".as_slice(),
            b"200",
            b"body200",
            b"body\n20",
            b"body\n2000",
            b"body\n2x0",
            b"body\n000",
            b"body\n099",
            b"body\n600",
            b"body\n200\n",
        ] {
            assert!(split_raw_http_response(output.to_vec()).is_err());
        }
    }

    // Loopback-only fixture: exercise the real curl command, not MockTransport.
    // The runner must unset proxy variables (or set NO_PROXY=127.0.0.1).
    #[test]
    fn raw_http_status_system_curl_loopback_regression() -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        for status in [200, 401, 402, 429, 500, 503] {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            listener.set_nonblocking(true)?;
            let address = listener.local_addr()?;
            let body = vec![137, 0, 255, b'\n', b'2', b'0', b'0', b'\n'];
            let expected = body.clone();
            let server = std::thread::spawn(move || -> std::io::Result<()> {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::TimedOut,
                                    "curl did not connect to loopback fixture",
                                ));
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(error) => return Err(error),
                    }
                };
                // Accepted sockets can inherit nonblocking mode on Windows.
                stream.set_nonblocking(false)?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                stream.set_write_timeout(Some(Duration::from_secs(2)))?;
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") && request.len() < 8192 {
                    if stream.read(&mut byte)? == 0 {
                        break;
                    }
                    request.push(byte[0]);
                }
                write!(
                    stream,
                    "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                )?;
                stream.write_all(&body)?;
                Ok(())
            });
            let response = SystemCurlTransport.fetch_raw_bytes(
                &format!("http://{address}/image"),
                &[],
                Duration::from_secs(2),
            );
            let served = server
                .join()
                .map_err(|_| std::io::Error::other("loopback fixture thread failed"))?;
            served?;
            let (actual_status, actual_body) = response?;
            assert_eq!(actual_status, status);
            assert_eq!(actual_body, expected);
        }
        Ok(())
    }

    struct MockTransport {
        call_count: Arc<AtomicUsize>,
        gemini_status: u16,
        pollinations_status: u16,
    }

    impl HttpTransport for MockTransport {
        fn execute_json(
            &self,
            _method: &str,
            url: &str,
            _headers: &[(&str, &str)],
            _body: Option<&Value>,
            _timeout: Duration,
        ) -> Result<(u16, Value), AiError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if url.contains("generativelanguage.googleapis.com") && url.contains("flash-image") {
                if self.gemini_status == 200 {
                    // Eight fake PNG bytes, base64-encoded, as inlineData like the real API.
                    let resp = json!({
                        "candidates": [{
                            "content": {
                                "parts": [{
                                    "inlineData": {"mimeType": "image/png", "data": "AQIDBAUGBwg="}
                                }]
                            }
                        }]
                    });
                    return Ok((200, resp));
                }
                return Ok((self.gemini_status, json!({"error": {"message": "Rate limited", "code": 429}})));
            } else if url.contains("generativelanguage.googleapis.com") {
                if self.gemini_status == 200 {
                    let resp = json!({
                        "candidates": [{
                            "content": {
                                "parts": [{
                                    "text": json!({
                                        "title": "Gemini Mock Plan",
                                        "rationale": "High-tier layout",
                                        "steps": [{
                                            "command": "layer.create",
                                            "params": {"name": "TestLayer"},
                                            "description": "Create test layer"
                                        }]
                                    }).to_string()
                                }]
                            }
                        }]
                    });
                    return Ok((200, resp));
                } else {
                    return Ok((self.gemini_status, json!({"error": {"message": "Rate limited", "code": 429}})));
                }
            } else if url.contains("openrouter.ai") {
                let resp = json!({
                    "choices": [{
                        "message": {
                            "content": json!({
                                "title": "OpenRouter Fallback Plan",
                                "rationale": "Fallback planning",
                                "steps": [{
                                    "command": "layer.create",
                                    "params": {"name": "FallbackLayer"},
                                    "description": "Create fallback layer"
                                }]
                            }).to_string()
                        }
                    }]
                });
                return Ok((200, resp));
            } else if url.contains("api.cohere.com") {
                let resp = json!({
                    "message": {
                        "content": [{
                            "text": json!({
                                "title": "Cohere Fallback Plan",
                                "rationale": "Fallback planning via Cohere",
                                "steps": [{
                                    "command": "layer.create",
                                    "params": {"name": "CohereLayer"},
                                    "description": "Create cohere layer"
                                }]
                            }).to_string()
                        }]
                    }
                });
                return Ok((200, resp));
            } else if url.contains("imagen-3.0-generate-002") {
                // Gemini image fallback
                use base64::Engine;
                let dummy_bytes = vec![137, 80, 78, 71, 13, 10, 26, 10]; // PNG header
                let b64 = base64::engine::general_purpose::STANDARD.encode(&dummy_bytes);
                return Ok((200, json!({ "predictions": [{ "bytesBase64Encoded": b64 }] })));
            }

            Ok((500, json!({"error": "Unknown mock"})))
        }

        fn fetch_raw_bytes(
            &self,
            _url: &str,
            _headers: &[(&str, &str)],
            _timeout: Duration,
        ) -> Result<(u16, Vec<u8>), AiError> {
            if self.pollinations_status == 200 {
                Ok((200, vec![1, 2, 3, 4])) // Mock image bytes
            } else {
                Ok((self.pollinations_status, vec![]))
            }
        }
    }

    #[test]
    fn context_load_balancer_filters_long_queries_from_lajiwa() {
        let config = ProviderConfig {
            gemini_api_key: Some("k1".into()),
            openrouter_api_key: Some("k2".into()),
            openrouter_enabled: true,
            cohere_api_key: Some("k3".into()),
            lajiwa_api_key: Some("k4".into()),
            ..Default::default()
        };
        let lb = ContextLoadBalancer::new(config);

        // Short context < 2000 chars -> Lajiwa included
        let short_profile = ContextProfile {
            char_length: 500,
            history_turns: 1,
            requires_json: true,
            is_image: false,
        };
        let cascade_short = lb.select_text_cascade(&short_profile);
        assert!(cascade_short.contains(&ProviderId::Lajiwa));

        // Long context > 2000 chars -> Lajiwa strictly excluded
        let long_profile = ContextProfile {
            char_length: 2500,
            history_turns: 5,
            requires_json: true,
            is_image: false,
        };
        let cascade_long = lb.select_text_cascade(&long_profile);
        assert!(!cascade_long.contains(&ProviderId::Lajiwa), "Lajiwa must be filtered out for >2000 chars");
    }

    #[test]
    fn provider_bridge_cascades_from_gemini_to_cohere_when_openrouter_disabled_on_429() -> Result<(), AiError> {
        let config = ProviderConfig {
            gemini_api_key: Some("k1".into()),
            openrouter_api_key: Some("k2".into()),
            openrouter_enabled: false, // Disabled per user instruction
            cohere_api_key: Some("k3".into()),
            ..Default::default()
        };
        assert!(!config.has_key(ProviderId::OpenRouter), "OpenRouter must be disabled");
        let catalog = CommandCatalog::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        let mock = Box::new(MockTransport {
            call_count: Arc::clone(&call_count),
            gemini_status: 429, // Gemini rate limited!
            pollinations_status: 200,
        });

        let bridge = AiProviderBridge::with_transport(config, catalog, mock);
        let intent = EditingIntent {
            raw_prompt: "make it modern".into(),
            category: super::super::natural_language::IntentCategory::PhotoEnhancement,
            archetype: None,
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: Some("modern".into()),
            text_content: None,
        };
        let scene = VisualScene::default();
        let context = AiContext::default();

        let plan = bridge.plan_editing(&intent, &scene, &context)?;
        assert_eq!(plan.title, "Cohere Fallback Plan");
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "Should try Gemini (failed 429) then Cohere (succeeded), skipping disabled OpenRouter");
        Ok(())
    }

    #[test]
    fn provider_bridge_cascades_from_gemini_to_openrouter_when_enabled_on_429() -> Result<(), AiError> {
        let config = ProviderConfig {
            gemini_api_key: Some("k1".into()),
            openrouter_api_key: Some("k2".into()),
            openrouter_enabled: true,
            ..Default::default()
        };
        let catalog = CommandCatalog::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        let mock = Box::new(MockTransport {
            call_count: Arc::clone(&call_count),
            gemini_status: 429, // Gemini rate limited!
            pollinations_status: 200,
        });

        let bridge = AiProviderBridge::with_transport(config, catalog, mock);
        let intent = EditingIntent {
            raw_prompt: "make it modern".into(),
            category: super::super::natural_language::IntentCategory::PhotoEnhancement,
            archetype: None,
            target_layer: None,
            target_layer_id: None,
            extracted_color: None,
            numeric_value: None,
            direction: None,
            stylistic_tone: Some("modern".into()),
            text_content: None,
        };
        let scene = VisualScene::default();
        let context = AiContext::default();

        let plan = bridge.plan_editing(&intent, &scene, &context)?;
        assert_eq!(plan.title, "OpenRouter Fallback Plan");
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "Should try Gemini (failed 429) then OpenRouter (succeeded)");
        Ok(())
    }

    #[test]
    fn image_generation_falls_back_from_pollinations_to_gemini() -> Result<(), AiError> {
        let config = ProviderConfig {
            gemini_api_key: Some("k1".into()),
            pollinations_api_key: Some("k2".into()),
            ..Default::default()
        };
        let catalog = CommandCatalog::new();

        let mock = Box::new(MockTransport {
            call_count: Arc::new(AtomicUsize::new(0)),
            gemini_status: 200,
            pollinations_status: 402, // Pollinations budget exhausted!
        });

        let bridge = AiProviderBridge::with_transport(config, catalog, mock);
        let img_bytes = bridge.generate_image("sunset texture", 512, 512)?;
        assert_eq!(img_bytes.len(), 8, "Mock PNG bytes returned from Gemini");
        Ok(())
    }
    // Offline image-attempt fixtures. A message deliberately containing 429/402
    // must not turn a transport or response-shape failure into an HTTP failure.
    struct ImageStatusTransport {
        status: Option<u16>,
        response: Value,
        bytes: Vec<u8>,
    }

    impl HttpTransport for ImageStatusTransport {
        fn execute_json(
            &self,
            _method: &str,
            _url: &str,
            _headers: &[(&str, &str)],
            _body: Option<&Value>,
            _timeout: Duration,
        ) -> Result<(u16, Value), AiError> {
            self.status
                .map(|status| (status, self.response.clone()))
                .ok_or_else(|| AiError::PlanningError("transport failure mentioning 429 and 402".into()))
        }

        fn fetch_raw_bytes(
            &self,
            _url: &str,
            _headers: &[(&str, &str)],
            _timeout: Duration,
        ) -> Result<(u16, Vec<u8>), AiError> {
            self.status
                .map(|status| (status, self.bytes.clone()))
                .ok_or_else(|| AiError::PlanningError("transport failure mentioning 429 and 402".into()))
        }
    }

    fn image_status_bridge(provider: ProviderId, transport: ImageStatusTransport) -> AiProviderBridge {
        // Exactly one configured provider isolates its recorded health outcome.
        let config = ProviderConfig {
            pollinations_api_key: (provider == ProviderId::Pollinations).then(|| "fixture".into()),
            gemini_api_key: (provider == ProviderId::Gemini).then(|| "fixture".into()),
            ..Default::default()
        };
        AiProviderBridge::with_transport(config, CommandCatalog::new(), Box::new(transport))
    }

    fn assert_image_failure_cooldown(
        provider: ProviderId,
        transport: ImageStatusTransport,
        seconds: u64,
    ) -> Result<(), String> {
        let bridge = image_status_bridge(provider, transport);
        let before = Instant::now();
        assert!(bridge.generate_image("offline fixture", 8, 8).is_err());
        let after = Instant::now();
        let guard = bridge.balancer.health.lock().map_err(|_| "health lock poisoned".to_string())?;
        let health = guard.get(&provider).ok_or_else(|| "missing provider health".to_string())?;
        let until = health.cooldown_until.ok_or_else(|| "missing cooldown".to_string())?;
        let duration = Duration::from_secs(seconds);
        // Bracket the internal Instant rather than sleeping or relying on a
        // timing tolerance: record_failure happened between before and after.
        assert!(until >= before + duration, "{provider:?}: cooldown shorter than {seconds}s");
        assert!(until <= after + duration, "{provider:?}: cooldown longer than {seconds}s");
        assert_eq!(health.active_requests, 0);
        assert_eq!(health.total_calls, 1);
        assert_eq!(health.total_errors, 1);
        Ok(())
    }

    #[test]
    fn image_failure_status_uses_existing_http_cooldown_policy() -> Result<(), String> {
        for provider in [ProviderId::Pollinations, ProviderId::Gemini] {
            for (status, seconds) in [(402, 300), (429, 60), (500, 20), (502, 20), (503, 20), (401, 5)] {
                assert_image_failure_cooldown(
                    provider,
                    ImageStatusTransport { status: Some(status), response: json!({"error": {"message": "429 is only text"}}), bytes: vec![1] },
                    seconds,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn image_failure_status_non_http_errors_use_general_cooldown() -> Result<(), String> {
        for provider in [ProviderId::Pollinations, ProviderId::Gemini] {
            assert_image_failure_cooldown(
                provider,
                ImageStatusTransport { status: None, response: Value::Null, bytes: vec![] },
                5,
            )?;
        }
        // Empty HTTP-200 Pollinations body: failure, but not rate limiting.
        assert_image_failure_cooldown(
            ProviderId::Pollinations,
            ImageStatusTransport { status: Some(200), response: Value::Null, bytes: vec![] },
            5,
        )?;
        // Gemini HTTP-200 schema and base64 failures likewise carry no HTTP
        // failure status. Arbitrary status-like response text is irrelevant.
        for response in [
            json!({"error": {"message": "429 402"}}),
            json!({"candidates": [{"content": {"parts": [{"inlineData": {"data": "not base64! 429"}}]}}]}),
        ] {
            assert_image_failure_cooldown(
                ProviderId::Gemini,
                ImageStatusTransport { status: Some(200), response, bytes: vec![] },
                5,
            )?;
        }
        Ok(())
    }

    #[test]
    fn image_failure_status_success_keeps_bytes_and_healthy_state() -> Result<(), String> {
        for provider in [ProviderId::Pollinations, ProviderId::Gemini] {
            let bytes = vec![1, 2, 3, 4];
            let response = json!({"candidates": [{"content": {"parts": [{"inlineData": {"data": "AQIDBA=="}}]}}]});
            let bridge = image_status_bridge(provider, ImageStatusTransport { status: Some(200), response, bytes: bytes.clone() });
            let actual = bridge.generate_image("offline fixture", 8, 8).map_err(|e| e.to_string())?;
            assert_eq!(actual, bytes);
            let guard = bridge.balancer.health.lock().map_err(|_| "health lock poisoned".to_string())?;
            let health = guard.get(&provider).ok_or_else(|| "missing provider health".to_string())?;
            assert_eq!(health.active_requests, 0);
            assert_eq!(health.total_calls, 1);
            assert_eq!(health.total_errors, 0);
            assert!(health.cooldown_until.is_none());
        }
        Ok(())
    }

}
