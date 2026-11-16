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
use std::sync::{Arc, Mutex};
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

/// Compile-time baked-in config value (set from CI secrets at build time via the
/// build environment). Used as the last fallback after process env and .env files,
/// so local overrides always win.
macro_rules! baked_config_var {
    ($name:literal) => {
        option_env!($name).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
    };
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

impl ProviderConfig {
    /// Create a new configuration populated from environment variables or .env file.
    pub fn from_env() -> Self {
        Self {
            gemini_api_key: get_config_var("GEMINI_API_KEY").or_else(|| baked_config_var!("GEMINI_API_KEY")),
            openrouter_api_key: get_config_var("OPENROUTER_API_KEY"),
            cohere_api_key: get_config_var("COHERE_API_KEY").or_else(|| baked_config_var!("COHERE_API_KEY")),
            pollinations_api_key: get_config_var("POLLINATIONS_API_KEY").or_else(|| baked_config_var!("POLLINATIONS_API_KEY")),
            lajiwa_api_key: get_config_var("LAJIWA_API_KEY").or_else(|| baked_config_var!("LAJIWA_API_KEY")),
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

        for &(k, v) in headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }

        if let Some(b) = body {
            let body_str = b.to_string();
            cmd.arg("-H").arg("Content-Type: application/json");
            cmd.arg("-d").arg(body_str);
        }

        let output = cmd.output().map_err(|e| AiError::PlanningError(format!("curl execution failed: {e}")))?;
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

        for &(k, v) in headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }

        let output = cmd.output().map_err(|e| AiError::PlanningError(format!("curl fetch failed: {e}")))?;
        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(AiError::PlanningError(format!("curl image request failed: {err_msg}")));
        }

        Ok((200, output.stdout))
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
                            log::warn!("Pollinations image failed: {e}; cascading to Gemini fallback");
                            self.balancer.record_failure(provider, Some(429));
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
                            log::warn!("Gemini image fallback failed: {e}");
                            self.balancer.record_failure(provider, None);
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
    ) -> Result<Vec<u8>, AiError> {
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
            Err(AiError::PlanningError(format!("Pollinations image failed with status {status}")))
        }
    }

    // Google Gemini Visual Image Synthesis Fallback
    fn execute_gemini_image(&self, prompt: &str, timeout: Duration) -> Result<Vec<u8>, AiError> {
        let key = self.balancer.config().gemini_api_key.as_deref().ok_or_else(|| AiError::PlanningError("Missing Gemini key for image fallback".into()))?;
        let url = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict";

        let headers = [("x-goog-api-key", key)];
        let body = json!({
            "instances": [{ "prompt": prompt }],
            "parameters": { "sampleCount": 1, "aspectRatio": "1:1" }
        });

        let (status, resp) = self.transport.execute_json("POST", url, &headers, Some(&body), timeout)?;
        if status != 200 {
            return Err(AiError::PlanningError(format!("Gemini image fallback status {status}")));
        }

        let b64_bytes = resp.get("predictions")
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("bytesBase64Encoded"))
            .and_then(Value::as_str)
            .ok_or_else(|| AiError::PlanningError("Missing b64 image data in Gemini response".into()))?;

        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(b64_bytes)
            .map_err(|e| AiError::PlanningError(format!("Base64 decode failed: {e}")))
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
            if url.contains("generativelanguage.googleapis.com") {
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
    fn provider_bridge_cascades_from_gemini_to_cohere_when_openrouter_disabled_on_429() {
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

        let plan = bridge.plan_editing(&intent, &scene, &context).expect("Fallback plan to Cohere should succeed");
        assert_eq!(plan.title, "Cohere Fallback Plan");
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "Should try Gemini (failed 429) then Cohere (succeeded), skipping disabled OpenRouter");
    }

    #[test]
    fn provider_bridge_cascades_from_gemini_to_openrouter_when_enabled_on_429() {
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

        let plan = bridge.plan_editing(&intent, &scene, &context).expect("Fallback plan should succeed");
        assert_eq!(plan.title, "OpenRouter Fallback Plan");
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "Should try Gemini (failed 429) then OpenRouter (succeeded)");
    }

    #[test]
    fn image_generation_falls_back_from_pollinations_to_gemini() {
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
        let img_bytes = bridge.generate_image("sunset texture", 512, 512).expect("Gemini fallback should provide image");
        assert_eq!(img_bytes.len(), 8, "Mock PNG bytes returned from Gemini");
    }
}
