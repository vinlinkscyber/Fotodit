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
`qual/verify-r8` @ `f59605e`; `main` @ `5328e43` does not yet contain the
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
`verified at gate G#`. Nothing is "done" below a gate. Commit references are
the private working chain on top of the audited baseline f59605e (tag
qual-candidate-r8); nothing is pushed or compiled.

### W1 — Contract truth (D1, D2, D3, D9)

- Landed: the strict provider-plan validator mirrors engine rules
  (`validate_selected_plan`), demand-driven disclosure replaced the stale
  fixed catalog dump: 42 core commands carry full schemas, every admitted id
  is grouped in `command_index`, and `contract.describe` returns full engine
  schemas for up to 12 admitted ids per round (fd1fbd4). Admitted geometry,
  type, shape, fill, selection, merge/group, filter and adjustment families
  each carry engine-mirrored validator rules with tests.
- Open gap: validator state tracking is per-plan; semantic target validity
  after geometry changes (a recolour planned against pre-resize coordinates)
  rests on tracked canvas arithmetic plus instruction, not on re-observation
  of the semantic target.
- Status: implemented (unverified).

### W2 — Truthful completion (D4)

- Landed: completion is reported only from the goal outcome's stop reason;
  the local repair adapter runs (`max_attempts` 2, `SemanticGoalRepair`, no
  extra provider requests). New: one bounded structured correction pass
  (b7e947a): a completed-but-unverified goal (stop Unmet /
  FinalStateUnverified / RepeatedPlan) re-plans once from the re-perceived
  post-edit scene with the applied-steps ledger and the local evaluation,
  language/math-only; exhaustion reports an honest partial or unmet outcome.
- Open gap: the local evaluator's criteria are weak (accuracy audit recorded
  in `crates/automation/src/ai/evaluation.rs`): 6 of 9 categories add no
  semantic check, so `GoalStop::Completed` is not independent proof of the
  goal. Measured goal-relevant region/selection/render criteria are the
  follow-up.
- Status: implemented (unverified).

### W3 — Observations (D5, D6, R3)

- Landed: enriched scene (groups, opacity, blend, adjustment kind, mask
  state, depth/parent, selection state, resolution_dpi); observation channel
  with five bounded commands: `image.detectFaces`, `image.analysis.pixelDigest`,
  `gradient.fill.get` (live gradient stops/state for indexed stop edits),
  `scene.layers` (targeted read past the scene's 32-layer cut; a reduced
  scene is never treated as complete), `contract.describe`. Bounded
  multi-round loop: MAX_OBSERVATION_ROUNDS=3 per goal, shared by the first
  plan and the correction pass; the provider is told to batch independent
  reads per round (49e62ed).
- Open gap: no goal-relevance link between an observation and the region the
  user's request names (the shirt's region is not a first-class measured
  target yet).
- Status: implemented (unverified).

### W4 — Engine-derived tool exposure (R4)

- Landed: 125 admitted commands + 5 observation commands out of 892
  inventoried entries; every exclusion carries a reason category in
  `docs/ai-coverage-inventory.md` (regenerated by
  `docs/ai-coverage-inventory.py` from the engine registry and the contract;
  `review` marks unaudited work, never an accepted cut). Resolution-dependent
  units are wired: the validator tracks dpi across plan steps (engine f32
  semantics), admitting `pt` and `keepPrintSize` (f464150).
- Open gap: 762 excluded entries, including the `review` remainder and the
  warp family admission in progress.
- Status: in progress.

### W5 — Cross-turn context and correction (D7, D8)

- Landed: bounded recent-turns digest (last 4, 200 chars each, cuts marked);
  active target by name; duplicate-name/truncated-turn ambiguity handling;
  context reset on document switch; creation capability without keyword
  gates and no fabricated documents. Correction: see W2.
- Open gap: none recorded beyond the W1/W2 gaps.
- Status: implemented (unverified).

### W6 — Provider activation persistence (R5)

- The keystore persists which provider is enabled; Save selects/enables and the
  choice survives restart; removing or resetting a key clears the persisted
  selection; clearing failures are surfaced, not swallowed.
- Status: implemented (unverified). Private patch, not pushed, not compiled in
  this environment; the keystore unit suite is not yet part of CI (see G1).

## 6. Current coverage accounting

Generated accounting lives in `docs/ai-coverage-inventory.md` (script:
`docs/ai-coverage-inventory.py`, run after every registry or contract change;
review the diff). Current: 892 entries (890 engine registry + contract-level
`contract.describe` and `scene.layers`), 125 admitted, 5 observation, 762
excluded with named reason categories (readonly / external / destructive /
app / macro / media / dev / planned / review).

## 7. Verification gates

A change counts as done only when its gate evidence exists. Gates run only with
the owner's explicit approval of each CI run.

- **G1 — Automated suites green in CI**, including the keystore unit suite and
  new contract/validator tests, which must be added to the workflow (they are
  not in it today). Evidence: ABSENT. Nothing in the private chain is compiled;
  all new code is implemented-unverified.
- **G2 — Rendered UI proofs** for every UI-affecting change (the existing
  snapshot examples pattern), reviewed as images. Evidence: ABSENT.
- **G3 — Windows activation proof**: on a real Windows build, save a key, see
  the provider enabled, close and relaunch, and confirm enablement persisted —
  the DPAPI path cannot be proven by Linux snapshots. Evidence: ABSENT.
- **G4 — End-to-end edit proof**: a request producing the correct edit or
  design in Fotodit, verified against the document state, with the provider
  exchange recorded. A live provider call happens only with the owner's
  explicit approval for that call. Evidence: ABSENT.
  G4 must include the user-anchored acceptance case below.
- **G5 — Coverage table updated** (section 6) with no regression in honesty:
  advertised == admitted == validated == tested. Evidence: PARTIAL — the
  inventory and its generator exist and are regenerated on change, but the
  == tested leg has no CI run behind it.

### G4 acceptance case: shirt recolour (user-anchored)

The request "make the shirt red" on a document holding a pixel layer with a
shirt. Passing means every link, not a plausible render:

1. **Locally observed target.** The shirt's region is obtained through the
   observation channel (selection/mask measured locally, with confidence and
   limits stated), never guessed from the image by the provider and never
   pre-supplied. An ambiguous or missing target ends in a truthful
   NeedsInput/refusal, not a guess.
2. **Known region/mask + colour criterion.** The plan is selection/mask plus a
   non-destructive recolour (a masked adjustment layer), admitted commands
   with engine-validated parameters, executed through the admitted local
   path with history/undo intact.
3. **Unchanged unrelated areas.** A bounded before/after measurement proves
   the criterion: the target region's colour moved to red AND a pixel digest
   of unrelated areas is unchanged. Command success or a good-looking render
   alone is not this proof.
4. **Correction or honest outcome.** If the measured criterion fails, the one
   structured correction pass runs; if it is exhausted, the chat says the
   goal is unmet or ambiguous, what was left in the document, and that Undo
   reverts it.

Proof-standard rule for this case and its fixtures: a fixture that
pre-supplies the correct semantic target (region, mask, layer) proves the
plumbing only and must be labeled as such. It is never evidence of autonomous
target recognition, and no fixture result substitutes for link 1.

## 8. Change control

- Update this document in the same commit as the change it describes.
- Never claim completion without gate evidence; say "implemented (unverified)"
  when that is the truth.
- One owner-approved CI run at a time; a failed run is diagnosed, never
  blindly re-run.
- Delivery to the owner happens only after the gates for the delivered scope
  pass.
