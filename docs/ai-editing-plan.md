# AI editing integration plan

Status: active working plan. Owner: project owner. Created 2026-11-24.
This document is the durable reference for the AI editing work. It exists so the
requirement, the architecture, the audited gaps and the proof standard are visible
in one place and cannot quietly drift. Update it in the same commit as any behavior
change it describes. Nothing in this file is evidence of completion; completion is
defined by the gates in section 7 only.

## 1. Purpose

Past work on the AI assistant fixed small pieces and treated passing tests and
successful builds as progress, without first checking the whole connected path
against the owner's actual requirement. This plan is the correction: the
requirement and the audited truth live here, and every future change is checked
against them before it counts as progress.

## 2. Owner requirements

Restated from the project owner's direction (2026-11-24). These are the
requirements; implementations that conflict with them are wrong regardless of
how much code exists.

- **R1 — Language and math in, not images.** The AI receives structured
  language/math context: document observations (layers, geometry, styles,
  selection, measured analysis) and the tools it may use. The canvas image is
  not the input. Optional image sharing, where it exists, is a separate consent
  and never the foundation.
- **R2 — The AI chooses, the engine computes.** The AI interprets the request
  and selects engine operations and parameter values. Fotodit performs the exact
  calculation, validates parameters against engine rules, executes and renders
  locally. Model arithmetic is never trusted unchecked; precision comes from the
  engine.
- **R3 — Local vision becomes math.** Local analysis (face/feature detection,
  color statistics, selection, geometry) feeds the AI structured measurements
  and detected targets, with honest confidence and recognition limits. Raw pixel
  numbers are not claimed to be semantic understanding.
- **R4 — Connected end to end.** Document observations flow to the AI; the AI
  discovers and selects tools; execution goes through Fotodit's admitted local
  command path; results feed back so the AI can check and correct. No hardcoded
  small subset presented as the whole.
- **R5 — Saved key means enabled.** Saving the owner's own provider key selects
  and enables that provider, and the enablement persists across restarts. Saving
  a key never sends a request on its own.
- **R6 — Safety properties stay.** Undo/history, local validation, staleness
  checks, per-command admission, and the filesystem/network/account boundaries
  are preserved. No arbitrary generated-code execution.
- **R7 — Honesty over appearance.** Anything not connected or not verified is
  stated plainly. No fabricated completion, no coverage claims beyond what is
  wired and tested.

## 3. Architecture contract

The connected path, each stage owned by named components:

1. **Observe** — engine inspection (`crates/engine/src/inspect.rs`) plus bounded
   local analysis (`image.analysis.colorStats`, `image.detectFaces`,
   `image.analysis.pixelDigest`) produce the structured document state.
2. **Perceive** — the perception engine
   (`crates/automation/src/ai/perception.rs`) turns inspection into the scene
   the planner sees. It must lose nothing the planner needs (see D5/D6).
3. **Plan** — the provider bridge (`crates/automation/src/ai/provider_bridge.rs`)
   sends the scene and the admitted tool contract to the configured provider and
   parses a plan of engine commands with parameter values.
4. **Validate** — the plan is validated locally against the engine's real
   commands and parameter rules before anything runs (contract validator,
   `CommandCatalog`, goal coordinator checks). A plan outside the contract is
   refused and edits nothing.
5. **Execute** — the execution engine
   (`crates/automation/src/ai/execution.rs`) dispatches admitted commands
   transactionally, with history/undo and staleness checks intact.
6. **Evaluate and correct** — the result is compared against the goal
   (evaluator, goal session, repair adapter). Completion is reported only when
   the goal is actually verified; otherwise the correction loop runs or the
   shortfall is reported honestly.

## 4. Audited baseline

Full-codebase connected-path audit, 2026-11-24. Candidate branch
`qual/verify-r5` @ `f59605e`; `main` @ `5328e43` does not yet contain the
goal-session/assistant-app path (`apps/fotodit/src/assistant.rs`,
`goal_session.rs`, `goal_repair.rs`, `enhancement_preflight.rs` exist only on
the candidate). Line anchors refer to `f59605e`.

### Confirmed defects

- **D1 — Photo enhancement deadlocks.** The provider contract forbids a `layer`
  key on `filter.sharpen.unsharpMask` (`provider_bridge.rs`
  `validate_selected_plan`), while `enhancement_preflight.rs:36` requires the
  first sharpening step to carry `layer` equal to the active layer. Every
  provider-planned photo enhancement is refused by one validator or the other.
- **D2 — Catalog/allowlist mismatch.** `SELECTED_COMMANDS` lists 19 commands,
  but `CommandCatalog` (`command_intelligence.rs`) lacks `layer.setProps` and
  `layer.moveTo`. The provider prompt advertises only the 17 present in both,
  and the goal coordinator's `validate_command` rejects plans that use the other
  two.
- **D3 — Stale catalog ids.** The catalog's `layer.setOpacity`,
  `layer.setBlendMode` and `layer.reorder` are not registered engine commands.
  The real commands are `layer.setProps` and `layer.moveTo`
  (`crates/engine/src/cmd.rs`).
- **D4 — False completion, no repair.** `apps/fotodit/src/assistant.rs` marks a
  goal complete when any execution succeeds, ignoring `GoalStop` and the
  evaluator, and sets `max_attempts: 1`, so the (local, no-extra-request)
  `SemanticGoalRepair` loop never runs.
- **D5 — Group children invisible.** `perception.rs` iterates only top-level
  layers; everything inside groups is dropped from the scene.
- **D6 — Scene drops state the planner needs.** The provider scene JSON omits
  opacity, blend mode, adjustment kind/params, effects, mask detail, selection
  state and multi-selection, all present in engine inspection.
- **D7 — No cross-turn context.** Each chat send builds a fresh `AiContext`;
  follow-up requests have no memory of earlier turns, and bare references
  ("again", "do that") are refused outright.
- **D8 — Creation is keyword-gated.** Document creation depends on the intent
  classifier's GraphicDesign category rather than on whether the request
  requires a new document.
- **D9 — Range drift.** The contract caps unsharp radius at 250 while the
  engine's own validation allows 0.1..1000
  (`crates/engine/src/filters/validation.rs`). Contract ranges must match the
  engine, which is the precision authority (R2).

### What exists and works

- A broad registered engine command set with per-command parameter rules
  (`crates/engine/src/cmd*.rs`, `filters/validation.rs`).
- Rich engine inspection: full layer tree with children, opacity/blend,
  adjustment params, effects, masks, selection, revision
  (`crates/engine/src/inspect.rs`).
- Local analysis commands: `image.analysis.colorStats`, `image.detectFaces`
  (score, box, five landmarks), `image.analysis.pixelDigest`.
- Transactional execution with rollback and history, preflight honesty checks,
  intra-plan `{"$step": N}` references, and a strict provider-plan validator.
- Provider key storage with Windows DPAPI encryption, and a qualification
  workflow that renders UI proofs.

## 5. Workstreams

Status legend: `planned` / `in progress` / `implemented (unverified)` /
`verified at gate G#`. Nothing is "done" below a gate.

### W1 — Contract truth (D1, D2, D3, D9)

- Replace the stale catalog ids with `layer.setProps` / `layer.moveTo` using the
  engine's parameter rules; add catalog validation for both.
- Resolve D1: the engine sharpening command targets the active layer and takes
  no `layer` parameter; the preflight accepts the paramless form (which it
  verifies against the live active layer) and still refuses a mismatched
  explicit target.
- Align contract ranges with engine validation (unsharp radius 0.1..1000).
- Status: in progress.

### W2 — Truthful completion (D4)

- Completion reported only from the goal outcome's stop reason; partial or
  unmet results say so and say what was left in the document.
- Enable the local repair loop (`max_attempts` > 1 with `SemanticGoalRepair`;
  no additional provider requests for the repairs it can do).
- Status: in progress.

### W3 — Observations (D5, D6, R3)

- Perception recurses into groups; elements carry opacity, blend, adjustment
  kind, mask state, depth/parent.
- The provider scene carries selection state and the enriched layer fields.
- `image.analysis.colorStats` (existing 4M-pixel budget) populates dominant
  colors; `image.detectFaces` (budget-gated) provides structured face
  observations with score, box and landmarks, labeled with their limits.
- Status: in progress.

### W4 — Engine-derived tool exposure (R4)

- Admit the engine's validated filter set (the `filters/validation.rs` table is
  the source of truth for parameters), the full adjustment-layer set
  (`crates/engine/src/adjust_params.rs`), and coherent layer operations
  (`layer.arrange.*`, `layer.delete`, rename via `layer.setProps`).
- Every admitted command gets: catalog entry, contract schema, validator rules,
  and tests. Validation rules mirror the engine's, never invent their own.
- Coverage accounting is maintained in this section: admitted commands vs the
  registered engine set, each exclusion with its reason (identity-changing,
  filesystem/network I/O, destructive-without-undo, dialog-only, or not yet
  reviewed). No silent small subset.
- Status: planned.

### W5 — Cross-turn context and correction (D7, D8)

- A persistent assistant context carries recent requests and applied edits into
  later planning, bounded and labeled as data.
- Creation is decided from the request and document state, not a keyword gate.
- Status: planned.

### W6 — Provider activation persistence (R5)

- The keystore persists which provider is enabled; Save selects/enables and the
  choice survives restart; removing or resetting a key clears the persisted
  selection; clearing failures are surfaced, not swallowed.
- Status: implemented (unverified). Private patch, not pushed, not compiled in
  this environment; the keystore unit suite is not yet part of CI (see G1).

## 6. Current coverage accounting

To be maintained as W4 lands. Baseline before W4: the provider contract admits
19 command ids; the engine registers far more (including the full filter,
adjustment and layer-operation families). The chat path currently advertises 17
of the 19 (D2). This section must end as an explicit table: command, admitted
(yes/no), validator source, exclusion reason if not admitted.

## 7. Verification gates

A change counts as done only when its gate evidence exists. Gates run only with
the owner's explicit approval of each CI run.

- **G1 — Automated suites green in CI**, including the keystore unit suite and
  new contract/validator tests, which must be added to the workflow (they are
  not in it today).
- **G2 — Rendered UI proofs** for every UI-affecting change (the existing
  snapshot examples pattern), reviewed as images.
- **G3 — Windows activation proof**: on a real Windows build, save a key, see
  the provider enabled, close and relaunch, and confirm enablement persisted —
  the DPAPI path cannot be proven by Linux snapshots.
- **G4 — End-to-end edit proof**: a request producing the correct edit or
  design in Fotodit, verified against the document state, with the provider
  exchange recorded. A live provider call happens only with the owner's
  explicit approval for that call.
- **G5 — Coverage table updated** (section 6) with no regression in honesty:
  advertised == admitted == validated == tested.

## 8. Change control

- Update this document in the same commit as the change it describes.
- Never claim completion without gate evidence; say "implemented (unverified)"
  when that is the truth.
- One owner-approved CI run at a time; a failed run is diagnosed, never
  blindly re-run.
- Delivery to the owner happens only after the gates for the delivered scope
  pass.
