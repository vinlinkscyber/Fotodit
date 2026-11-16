# AI Provider Bridge & Context-Aware Load Balancer

Fotodit integrates a zero-dependency, multi-provider AI bridge with context-aware load balancing and dual-pipeline fallback routing across five official AI providers.

---

## 1. Supported Providers

| Provider | Primary Role | Default Model / Endpoint | Special Constraints |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | Frontier Text, JSON Planning, Image Fallback | `gemini-3.5-flash`<br>`imagen-3.0-generate-002` | Native JSON schema output via `responseMimeType: "application/json"`. |
| **OpenRouter** | Multi-Model Routing & Resiliency | `openai/gpt-4.1-mini` | *Currently disabled by default* (`OPENROUTER_ENABLED=0`). Re-enable with `OPENROUTER_ENABLED=1`. |
| **Cohere** | Enterprise Structured Chat | `command-a-plus-05-2026` | Native `response_format: { type: "json_object" }`. |
| **Pollinations** | Dedicated Image Generation | `tongyi-mai/z-image-turbo`<br>`gen.pollinations.ai` | Returns raw binary image bytes (PNG/JPEG) directly over HTTP stream. |
| **Lajiwa** | Lightweight Worker Engine | Self-hosted Cloudflare Worker | Strict 2,000 Unicode character limit (`413 message_too_long`), 20 RPM ceiling. |

---

## 2. Dual-Pipeline Architecture

```
[User Request / Creative Intent]
           │
           ├──────────────────────────────────────────┐
           ▼                                          ▼
   [Text & Intelligence]                      [Image Generation]
           │                                          │
    Context Analysis                           Context Analysis
           │                                          │
   Context-Aware Balancer                     Cascade Selection
   (Filtering & RPM Tracking)                         │
           │                                          ▼
           ▼                                    [Pollinations]
    ┌──────────────┐                            (Raw Byte Stream)
    │ 1. Gemini    │ (40% Weight)                     │
    └──────┬───────┘                            (402 / 429 Fail)
           │ (Fail / 429 / 5xx)                       ▼
    ┌──────▼───────┐                               [Gemini]
    │ 2.OpenRouter │ (30% Weight)               (Imagen Fallback)
    └──────┬───────┘                                  │
           │ (Fail / 429 / 5xx)                 (All Remote Fail)
    ┌──────▼───────┐                                  ▼
    │ 3. Cohere    │ (20% Weight)             [Procedural Local]
    └──────┬───────┘
           │ (Fail / 429 / 5xx)
    ┌──────▼───────┐
    │ 4. Lajiwa    │ (10% Weight, <2k chars only)
    └──────┬───────┘
           │ (All Remote Exhausted)
           ▼
    ┌──────────────┐
    │ 5. Local     │ (Deterministic Rule Planner, 100% Offline)
    └──────────────┘
```

### Pipeline 1: Text & Intelligence
1. **Tier 1 — Google Gemini 3.5 Flash** (Base Weight: 40)
2. **Tier 2 — Cohere Command A+** (Base Weight: 20) — *Active Tier 2 with OpenRouter disabled*
3. **Tier 3 — Lajiwa Worker** (Base Weight: 10, filtered if context > 2,000 characters)
4. **Tier 4 — Local Deterministic Rule Planner** (100% offline fallback)
*(Note: OpenRouter is held in standby with weight 30; to activate, set `OPENROUTER_ENABLED=1`).*

### Pipeline 2: Image Generation
1. **Primary — Pollinations**: Streams raw image bytes directly from `gen.pollinations.ai` into Fotodit layer buffers.
2. **Secondary Fallback — Google Gemini Imagen**: Activates if Pollinations returns 402 (budget limit), 429 (rate limit), or timeout.
3. **Tertiary Fallback — Local Procedural Generator**: Renders native parametric geometric / gradient layers.

---

## 3. Context-Aware Load Balancer

The `ContextLoadBalancer` manages traffic across providers dynamically to prevent provider burnout and avoid quota exhaustion:

1. **Character Count Capacity Filtering**:
   - Analyzes incoming prompts and conversation history depth.
   - Any instruction whose character count exceeds **2,000 Unicode characters** automatically excludes **Lajiwa** from candidate selection, preventing `413 message_too_long` errors before making a network call.

2. **RPM (Requests Per Minute) Ceilings**:
   - Monitors active timestamps over a sliding 60-second window.
   - If an active provider approaches its RPM ceiling (Lajiwa: 20, Pollinations: 30, Cohere: 40, Gemini: 60, OpenRouter: 60), the balancer applies an automatic -30 penalty to reroute traffic to other healthy providers.

3. **Circuit Breaker & Automatic Cooldown**:
   - `429 Too Many Requests`: 60-second cooldown.
   - `402 Payment Required / Out of Credits`: 300-second cooldown.
   - `500 / 502 / 503 Server Error`: 20-second cooldown.
   - General network error / timeout: 5-second cooldown.
   - A single successful request immediately clears any active cooldown.

---

## 4. Configuration & Environment Variables

Credentials and model choices can be set via environment variables or passed programmatically via `ProviderConfig`:

```bash
# Google Gemini
export GEMINI_API_KEY="your-gemini-key"
export GEMINI_MODEL="gemini-3.5-flash"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENROUTER_MODEL="openai/gpt-4.1-mini"

# Cohere
export COHERE_API_KEY="your-cohere-key"
export COHERE_MODEL="command-a-plus-05-2026"

# Pollinations
export POLLINATIONS_API_KEY="your-pollinations-key"
export POLLINATIONS_IMAGE_MODEL="tongyi-mai/z-image-turbo"

# Lajiwa
export LAJIWA_API_KEY="ljw_your_key"
export LAJIWA_BASE_URL="https://lajiwa.roydeburgs.workers.dev"
```

---

## 5. Programmatic Usage in Rust

```rust
use fotodit_automation::{AiEngine, ProviderConfig};

// Automatically read credentials and models from environment
let ai = AiEngine::with_bridge_from_env();

// Or supply custom configuration in code
let config = ProviderConfig {
    gemini_api_key: Some("...".into()),
    openrouter_api_key: Some("...".into()),
    pollinations_api_key: Some("...".into()),
    ..Default::default()
};
let ai = AiEngine::with_provider_config(config);
```
