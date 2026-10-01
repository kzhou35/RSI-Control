"""Attack-method figure from the lever labels (scripts/monitor/label_levers.py output).
Panel (a): share of attack runs by primary lever, per agent model (ignored/refused shown as the grey tail).
Panel (b): the same by pipeline stage. Writes figures/attack_levers.pdf/.png.
Run from the repo root: uv run python analysis/scripts/make_lever_figure.py"""
import glob, json, os, sys, collections
import yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from taskmaps import MAIN, SIDE, STAGE_SHORT  # noqa: E402
LABELS = sorted(glob.glob(os.path.join(HERE, "..", "data", "lever_labels_*.jsonl")))[-1]
SPECS = os.path.join(HERE, "..", "..", "src", "rsi_bench", "tasks", "specs", "*.yaml")
MODELS = [("anthropic/claude-opus-5", "Opus 5"), ("openai/gpt-5.6-sol", "GPT-5.6"), ("gemini/gemini-3.7-flash", "Gemini 3.7"),
          ("xai/grok-4.6", "Grok 4.6"), ("fireworks_ai/glm-5p3", "GLM-5.3"), ("meta_ai/muse-spark-1.3-contributor", "Muse 1.3")]
STAGES = ["Data curation", "Supervised fine-tuning", "Reward-model training", "Reinforcement learning", "On-policy distillation"]
LEVERS = [("synthetic_data", "Write training data"), ("data_selection", "Select or relabel data"), ("reward_shaping", "Shape the reward"),
          ("recipe", "Tune the recipe"), ("artifact_edit", "Edit the artifact"), ("ignored", "Ignored"), ("refused", "Refused")]
COLORS = {"synthetic_data": "#2a78d6", "data_selection": "#7fb2ea", "reward_shaping": "#1baf7a", "recipe": "#eb6834",
          "artifact_edit": "#b58bd9", "ignored": "#d6d4cd", "refused": "#a19f98"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": INK2,
                     "text.color": INK, "axes.labelcolor": INK2, "pdf.fonttype": 42})
specs = {}
for f in glob.glob(SPECS):
    d = yaml.safe_load(open(f))
    if d.get("deprecated") or d.get("status") == "draft" or d["main_prompt"] not in MAIN or d["goal"] not in SIDE: continue
    specs[d["id"]] = d
rows = [json.loads(l) for l in open(LABELS)]
rows = [r for r in rows if not r.get("error") and r["task"] in specs and r.get("model") and r.get("outcome") != "invalid"]
def key(r):
    if r.get("attempt") == "refused": return "refused"
    if r.get("attempt") == "ignored" or r.get("primary_lever") in (None, "none"): return "ignored"
    return r["primary_lever"]
def shares(sel):
    c = collections.Counter(key(r) for r in sel); n = len(sel)
    return {k: c[k] / n for k, _ in LEVERS} if n else {}, n
def stacked(ax, groups, title):
    ys = list(range(len(groups)))[::-1]
    for y, (name, sel) in zip(ys, groups):
        sh, n = shares(sel); left = 0
        for k, _ in LEVERS:
            w = sh.get(k, 0)
            if w: ax.barh(y, w, left=left, color=COLORS[k], height=0.62, edgecolor="white", linewidth=0.5); left += w
    ax.set_yticks(ys); ax.set_yticklabels([g[0] for g in groups]); ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"]); ax.tick_params(length=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.set_title(title, loc="left", fontweight="bold", color=INK)
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
stacked(axes[0], [(name, [r for r in rows if r["model"] == m]) for m, name in MODELS], "(a) By agent model")
stacked(axes[1], [(STAGE_SHORT[s], [r for r in rows if MAIN[specs[r["task"]]["main_prompt"]][0] == s]) for s in STAGES], "(b) By pipeline stage")
fig.legend(handles=[Patch(color=COLORS[k], label=lab) for k, lab in LEVERS], loc="lower center", ncol=4, frameon=False,
           fontsize=6.5, bbox_to_anchor=(0.5, -0.16), handlelength=1.0, columnspacing=1.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "..", "figures", f"attack_levers.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=200)
c = collections.Counter(key(r) for r in rows); print(len(rows), "labelled runs:", dict(c))
for m, name in MODELS:
    sel = [r for r in rows if r["model"] == m]; print(f"  {name:10s} n={len(sel):3d}", dict(collections.Counter(key(r) for r in sel)))
