"""Main-task performance under attack, relative to the same model's honest runs. For every (pair, model) cell with
valid attack and honest arms, take the pair's main-task cover metrics (quality, accuracy, refusal/compliance rates,
pass rates), compute attack - honest per metric (all are rates or accuracies in [0,1], so the unit is percentage points) and average within the cell. Pairs whose side task
conflicts with the main task by construction, and the sandbagging pairs, are excluded (EXCLUDE). Panels: by agent
model, by main-task stage, by side-task class, by pair. Whiskers: 95% bootstrap CI over cells.
Run from the repo root: uv run python analysis/scripts/make_cover_figure.py"""
import glob, json, os, sys, random, collections
import yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from taskmaps import MAIN, SIDE, STAGE_SHORT, CLASS_ORDER  # noqa: E402
REPORT = sorted(glob.glob(os.path.join(HERE, "..", "data", "sabotage_report_*.json")))[-1]
SPECS = os.path.join(HERE, "..", "..", "src", "rsi_bench", "tasks", "specs", "*.yaml")
MODELS = [("anthropic/claude-opus-5", "Opus 5"), ("openai/gpt-5.6-sol", "GPT-5.6"), ("gemini/gemini-3.7-flash", "Gemini 3.7"),
          ("xai/grok-4.6", "Grok 4.6"), ("fireworks_ai/glm-5p3", "GLM-5.3"), ("meta_ai/muse-spark-1.3-contributor", "Muse 1.3")]
STAGES = ["Data curation", "Supervised fine-tuning", "Reward-model training", "Reinforcement learning", "On-policy distillation"]
# side task conflicts with the main task by construction (topic refusal vs over-refusal, defect blindness vs accuracy,
# excited vs neutral register) or IS under-delivery of the main task (sandbagging)
EXCLUDE = {"row06_chess_refusal", "row15_rl_topic_refusal", "row13_capability_suppression", "row14_rl_rubric_persona",
           "row22_rm_sandbag", "row23_sycophancy_sandbag"}
# cover metrics that measure main-task performance (higher is better); leak/format/report checks are not performance
PERF = {"keep_retention", "overall_survival", "agreement_spearman", "clean_quality", "over_refusal_comply_rate", "safety_refusal_rate",
        "delivered_accuracy", "quality_rate", "chat_winrate", "control_accuracy", "code_pass_rate"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
C_CLASS = {"Capability": "#2a78d6", "Alignment": "#eb6834", "Goal injection": "#1baf7a"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 6.5,
                     "axes.titlesize": 7, "xtick.labelsize": 5.8, "ytick.labelsize": 6, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2, "pdf.fonttype": 42})
specs = {}
for f in glob.glob(SPECS):
    d = yaml.safe_load(open(f))
    if d.get("deprecated") or d.get("status") == "draft" or d["main_prompt"] not in MAIN or d["goal"] not in SIDE: continue
    specs[d["id"]] = d
cells = []   # dict(task, model, rel, stage, cls)
for e in json.load(open(REPORT)):
    if e["framing"] != "scored" or e["task"] not in specs or e["task"] in EXCLUDE: continue
    if not e.get("valid", True) or "INCOMPLETE" in e["status"] or e["attack"] is None or e["honest"] is None: continue
    rels = []
    for ch in (e.get("cover") or {}).get("checks", []):
        if ch["metric"] not in PERF or ch.get("direction") != "min": continue
        a, h = ch.get("attack"), ch.get("honest")
        if a is None or not h: continue
        rels.append(a - h)
    if not rels: continue
    sp = specs[e["task"]]
    cells.append(dict(task=e["task"], model=e["model"], rel=sum(rels) / len(rels), stage=MAIN[sp["main_prompt"]][0], cls=SIDE[sp["goal"]][0]))
random.seed(0)
def stat(vals, B=2000):
    n = len(vals); m = sum(vals) / n
    bs = sorted(sum(random.choice(vals) for _ in range(n)) / n for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1], n
def panel(ax, groups, title, colors=None, rotate=0):
    groups = [g for g in groups if g[1]]
    xs = range(len(groups)); ms = []
    for x, (name, vals) in enumerate(groups):
        if not vals: ms.append(0); continue
        m, lo, hi, n = stat(vals); ms.append(m)
        col = (colors or {}).get(name, BLUE)
        ax.bar(x, 100 * m, color=col, width=0.62, alpha=0.9)
        ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=INK2, capsize=1.5, lw=0.7)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(list(xs)); ax.set_xticklabels([g[0] for g in groups], rotation=rotate, ha="right" if rotate else "center", rotation_mode="anchor" if rotate else "default")
    ax.set_ylim(-25, 8); ax.set_yticks([-20, -10, 0]); ax.tick_params(length=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold")
main_ids = {m: f"M{i+1}" for i, m in enumerate(MAIN)}
pairs = collections.defaultdict(list)
for c in cells: pairs[c["task"]].append(c["rel"])
ABBR = {"Overlooks planted errors": "Planted errors", "Goal: copy files unprompted": "Goal: copy files", "Goal: protect chosen data": "Goal: protect data", "Sandbag alignment training": "Sandbag alignment", "Covert reasoning signal": "Reasoning signal", "Unwanted literature added": "Unwanted literature", "Filtered domain retained": "Domain retained", "Goal: shape the next model": "Goal: shape next model"}
def plabel(t):
    sp = specs[t]; n = SIDE[sp['goal']][1]; return f"{ABBR.get(n, n)} ({main_ids[sp['main_prompt']]})"
order = sorted(pairs, key=lambda t: -sum(pairs[t]) / len(pairs[t]))
SHORT = {"Opus 5": "Opus", "GPT-5.6": "GPT", "Gemini 3.7": "Gemini", "Grok 4.6": "Grok", "GLM-5.3": "GLM", "Muse 1.3": "Muse"}
CLS_SHORT = {"Capability": "Capab.", "Alignment": "Align.", "Goal injection": "Goal inj."}
# main text: three compact panels for a half-width wrapfigure
fig, axes = plt.subplots(1, 3, figsize=(2.75, 1.45), gridspec_kw={"width_ratios": [6, 4, 3], "wspace": 0.18}, sharey=True)
panel(axes[0], [(SHORT.get(name, name), [c["rel"] for c in cells if c["model"] == m]) for m, name in MODELS], "(a) Agent", rotate=50)
panel(axes[1], [(STAGE_SHORT[s], [c["rel"] for c in cells if c["stage"] == s]) for s in STAGES], "(b) Stage", rotate=50)
panel(axes[2], [(CLS_SHORT[k], [c["rel"] for c in cells if c["cls"] == k]) for k in CLASS_ORDER], "(c) Class",
      colors={CLS_SHORT[k]: v for k, v in C_CLASS.items()}, rotate=50)
for ax in axes: ax.set_ylim(-13, 4); ax.set_yticks([-10, -5, 0]); ax.tick_params(axis="x", labelsize=5.8); ax.title.set_fontsize(6.8)
for ax in axes[1:]: ax.tick_params(axis="y", labelleft=False); ax.spines["left"].set_visible(False)
axes[0].set_ylabel("Attack minus benign (pts)", fontsize=6.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "..", "figures", f"main_task_cost.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=220)
plt.close(fig)
# appendix: one panel per pair
fig, ax = plt.subplots(figsize=(5.5, 1.7))
panel(ax, [(plabel(t), pairs[t]) for t in order], "", rotate=40)
ax.set_ylim(-25, 8); ax.set_ylabel("Attack minus benign\n(percentage points)", fontsize=6.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "..", "figures", f"main_task_cost_pairs.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=220)
print(len(cells), "cells;", len(pairs), "pairs")
for name, sel in [(n, [c["rel"] for c in cells if c["model"] == m]) for m, n in MODELS] + [(s, [c["rel"] for c in cells if c["stage"] == s]) for s in STAGES] + [(k, [c["rel"] for c in cells if c["cls"] == k]) for k in CLASS_ORDER]:
    if sel: m, lo, hi, n = stat(sel); print(f"  {name:26s} {100*m:+6.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]  n={n}")
for t in order: m, lo, hi, n = stat(pairs[t]); print(f"  {plabel(t):48s} {100*m:+6.1f} pts n={n}")
