#!/usr/bin/env python3
"""Regenerate docs/ai-coverage-inventory.md from the engine registry and the
AI provider contract.

Static extraction, no Rust toolchain needed:
- command ids from spec!(...), cmd!(...), `id: "..."` struct literals and
  `const X: &str = "x.y"` handlers across crates/engine/src (tests excluded);
- the ADJ table in crates/engine/src/cmd.rs expands to its generated
  layer.newAdjustmentLayer.<kind> / image.adjustments.<kind> pairs;
- the contract's admitted set from SELECTED_COMMANDS, CORE_CONTRACT_COMMANDS
  and OBSERVATION_COMMANDS in crates/automation/src/ai/provider_bridge.rs;
- admitted filters from FILTER_FIELDS in crates/engine/src/filters.rs.

Statuses:
- admitted-core: full engine schema always in the provider contract
- admitted-index: admitted; id disclosed in command_index, schema on demand
  via the contract.describe observation
- observation: bounded read-only measurement the provider may demand
- excluded-<category>: not admitted, with the boundary's reason; `review`
  means not yet audited, never an accepted final cut
"""
import re, subprocess, collections, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def read(p): return (ROOT / p).read_text()

files = subprocess.run(["git", "ls-files", "crates/engine/src"], capture_output=True, text=True, cwd=ROOT).stdout.split()
ids = set()
for f in files:
    if not f.endswith(".rs") or "/tests" in f or f.endswith("tests.rs"):
        continue
    s = read(f)
    for pat in [r'spec!\(\s*"([\w.]+)"', r'cmd!\(\s*"([\w.]+)"', r'^\s*id:\s*"([\w.]+)"\s*,',
                r'const \w+:\s*&str\s*=\s*"([\w]+\.[\w.]+)"', r'CommandSpec\s*\{\s*id:\s*"([\w.]+)"']:
        ids |= set(re.findall(pat, s, re.M))
cmdrs = read("crates/engine/src/cmd.rs")
adj = re.findall(r'\(\s*"(\w+)",\s*"', cmdrs[cmdrs.index("const ADJ"):cmdrs.index("for &(kind, label, params) in ADJ")])
ids |= {f"layer.newAdjustmentLayer.{k}" for k in adj} | {f"image.adjustments.{k}" for k in adj}

pb = read("crates/automation/src/ai/provider_bridge.rs")
sel = re.findall(r'"([\w.]+)"', re.search(r"const SELECTED_COMMANDS: &\[&str\] = &\[(.*?)\];", pb, re.S).group(1))
core = re.findall(r'"([\w.]+)"', re.search(r"const CORE_CONTRACT_COMMANDS: &\[&str\] = &\[(.*?)\];", pb, re.S).group(1))
obs = re.findall(r'"([\w.]+)"', re.search(r"const OBSERVATION_COMMANDS: &\[&str\] = &\[(.*?)\];", pb, re.S).group(1))
ff = sorted(set(re.findall(r'\(\s*"(filter\.[\w.]+)"\s*,', read("crates/engine/src/filters/validation.rs"))))

# contract.describe (schema catalog) and scene.layers (perceived-scene read)
# are contract-level commands, not engine registry entries; every other
# observation is.
CONTRACT_LEVEL = {"contract.describe", "scene.layers"}
missing = [c for c in sel + ff + [o for o in obs if o not in CONTRACT_LEVEL] if c not in ids]
if missing:
    raise SystemExit(f"admitted commands not found in the engine registry: {missing}")

# --- exclusion reason categories -------------------------------------------
READONLY = "readonly"      # state already flows to the provider via scene JSON / observations
EXTERNAL = "external"      # filesystem / external-content boundary (assets, presets, LUT data)
DESTRUCTIVE = "destructive" # destructive pixel variant; the contract uses non-destructive layers
APPCONTROL = "app"         # app/session/view/preference control, not a document edit
MACRO = "macro"            # actions/variables/batch: a macro layer over commands the provider addresses directly
MEDIA = "media"            # timeline/video/animation, outside still-image plans
DEV = "dev"                # internal/dev plumbing, never a user edit
LOCALMEASURE = "localmeasure" # exact local measurement substrate; never provider-admitted (pinned by test)
PLANNED = "planned"        # named remaining work (paths/vector, styles, channels, type extras...)
REVIEW = "review"          # not yet audited; pending, not an accepted cut

REASON = {
    READONLY: "read-only state/info: flows to the provider through the scene JSON or observations, not as a plan step",
    EXTERNAL: "filesystem or external-content boundary: assets, presets, LUTs and network paths stay gated",
    DESTRUCTIVE: "destructive pixel variant: the contract works through non-destructive adjustment/fill layers",
    APPCONTROL: "application/session/view control, not a document edit",
    MACRO: "macro/action layer over commands the contract already addresses directly",
    MEDIA: "timeline/video/animation, outside still-image editing plans",
    DEV: "internal/dev plumbing, never a user edit",
    LOCALMEASURE: "exact local measurement substrate for the goal coordinator's criteria; carries raw pixels/coverage, so it is never provider-admitted (pinned by provider_bridge test)",
    PLANNED: "named remaining work, pending its own audited slice",
    REVIEW: "not yet audited; pending review, not an accepted final cut",
}

def classify(c):
    if c in ("test.panic", "index.json"):
        return DEV
    if c in ("image.analysis.regionPixels", "image.analysis.selectionCoverage", "image.analysis.layerMaskCoverage"):
        return LOCALMEASURE
    fam = c.split(".")[0]
    leaf = c.split(".")[-1].lower()
    if c.startswith("image.adjustments."):
        return DESTRUCTIVE
    if fam in ("view", "prefs", "session", "jobs", "help", "updates", "window", "tool", "tools"):
        return APPCONTROL
    if c.startswith("brush.presets"):
        return EXTERNAL
    if fam in ("bevelEmboss", "dropShadow", "innerGlow", "innerShadow", "outerGlow",
               "satin", "colorOverlay", "gradientOverlay", "patternOverlay", "stroke"):
        return PLANNED  # layer styles, a named remaining slice
    if fam in ("actions", "variables", "batch"):
        return MACRO
    if fam in ("timeline", "slice", "video"):
        return MEDIA
    if any(k in c for k in ("import", "export", "save", "open", "place", "load", "file", "url", "fetch", "exportAs")) \
            or fam in ("swatches", "patterns") or c.startswith(("gradient.import", "pattern.", "plugin.")):
        return EXTERNAL
    if leaf.startswith(("list", "get", "inspect")) or c.startswith(("document.inspect", "command.list", "channel.list", "measurementLog", "image.analysis.")):
        return READONLY
    if fam in ("path", "style") or c.startswith(("type.", "shape.", "channel.", "select.", "gradient.", "layer.")):
        return PLANNED
    return REVIEW

rows = [("contract.describe", "observation"), ("scene.layers", "observation")]
for c in sorted(ids):
    if c in obs:
        status = "observation"
    elif c in sel:
        status = "admitted-core" if c in core else "admitted-index"
    elif c in ff:
        status = "admitted-index"
    else:
        status = f"excluded-{classify(c)}"
    rows.append((c, status))

counts = collections.Counter(s for _, s in rows)
admitted = [c for c, s in rows if s.startswith("admitted")]

GATES = {
    "file.new": "no document required",
    "type.create": "document open",
    "type": "document open; Type-layer target (except type.create)",
    "shape": "document open; shape.create creates, shape.edit/rasterize target a Shape layer",
    "gradient.fill": "document open; gradient.fill.set targets a Gradient Fill layer",
    "layer.newAdjustmentLayer": "document open; non-destructive adjustment layer above the active layer",
    "layer.newFillLayer": "document open; consumes a live selection into the new layer's mask",
    "layer.layerMask": "document open; target layer and mask state as the command implies",
    "layer.arrange": "document open; explicit layer",
    "layer.merge": "document open; multi-layer merge semantics, last-step-only for flattenImage",
    "select.modify": "document open; live selection",
    "select.grow": "document open; live selection",
    "select.similar": "document open; live selection",
    "select.transformSelection": "document open; live selection",
    "select.refineEdge": "document open; live selection (or sampleAllLayers)",
    "select": "document open",
    "edit": "document open",
    "image": "document open",
    "filter": "document open; live active pixel layer (engine gate)",
    "layer": "document open",
}

def gate(c):
    if c in GATES: return GATES[c]
    for pref in sorted(GATES, key=len, reverse=True):
        if c.startswith(pref + "."):
            return GATES[pref]
    return "document open"

out = []
out.append("# AI editing contract: engine coverage inventory")
out.append("")
out.append("Generated by `docs/ai-coverage-inventory.py` (static extraction; see its header for"
           " method and status definitions). Regenerate after any registry or contract change and"
           " review the diff.")
out.append("")
total = len(rows)
out.append(f"- Engine registry: **{total-1}** commands (literal specs plus the generated"
           f" adjustment-layer/destructive pairs and the validated filter table),"
           " plus the contract-level contract.describe and scene.layers")
out.append(f"- Admitted to the provider contract: **{len(admitted)}**"
           f" ({counts['admitted-core']} with full schemas always in the prompt,"
           f" {counts['admitted-index']} disclosed by id in `command_index`, schema on demand"
           " via the `contract.describe` observation)")
out.append(f"- Observation channel: **{counts['observation']}**")
excluded = total - len(admitted) - counts['observation']
out.append(f"- Excluded: **{excluded}** (every exclusion carries its boundary reason below;"
           " `review` marks unaudited pending work, never an accepted cut)")
out.append("")
out.append("## Admitted commands")
out.append("")
out.append("| command | disclosure | requires |")
out.append("| --- | --- | --- |")
for c in admitted:
    disc = "full schema in prompt" if c in core else "id in `command_index`; schema via `contract.describe`"
    out.append(f"| `{c}` | {disc} | {gate(c)} |")
out.append("")
out.append("## Observation channel")
out.append("")
out.append("| command | bounded to |")
out.append("| --- | --- |")
out.append("| `image.detectFaces` | minScore 0..1, sampleAllLayers; detections carry detector uncertainty, never identities |")
out.append("| `image.analysis.pixelDigest` | maxPixels 1..4000000; bounded statistical digest |")
out.append("| `contract.describe` | 1..12 admitted command ids per round in total; answered locally from the command catalog, no engine path |")
out.append("| `scene.layers` | 1..8 layer ids per round; answered locally from the full perceived scene, including layers past the scene list's cut |")
out.append("")
out.append("## Excluded commands, by family and reason")
out.append("")
fam_groups = collections.defaultdict(list)
for c, s in rows:
    if s.startswith("excluded-"):
        fam_groups[(c.split(".")[0], s[len("excluded-"):])].append(c)
out.append("| family | reason | commands |")
out.append("| --- | --- | --- |")
for (fam, cat), cmds in sorted(fam_groups.items()):
    shown = ", ".join(f"`{c}`" for c in cmds)
    out.append(f"| {fam} ({len(cmds)}) | {REASON[cat]} | {shown} |")
out.append("")
out.append("## Flat inventory (id | status)")
out.append("")
out.append("```")
for c, s in rows:
    out.append(f"{c} | {s}")
out.append("```")
out.append("")

(ROOT / "docs" / "ai-coverage-inventory.md").write_text("\n".join(out))
print(f"wrote docs/ai-coverage-inventory.md: {total} commands, {len(admitted)} admitted, "
      f"{counts['observation']} observation, {excluded} excluded")
