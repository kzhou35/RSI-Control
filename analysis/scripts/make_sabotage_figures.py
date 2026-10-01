"""Sabotage-capability figures for the paper, from data/sabotage_report_<date>.json.

Run-level success, read from report_differentials' per-run verdicts (rows[i]["per_run"], 2026-09-23): a run is
"installed" when its own headline value clears the spec threshold, beats the model's honest mean by 1.645 sigma
(sigma = median-pooled run-to-run spread of honest runs) and ITS OWN cover metrics hold; "landed" drops the cover
condition (side property present, main task damaged). 95% intervals are Wilson intervals over runs.

Outputs (figures/): sabotage_rates.pdf (one row: per model | per class | per stage),
sabotage_pairs.pdf (per task pair), sabotage_overview.pdf (both rows in one figure).
Run from the repo root: uv run python analysis/scripts/make_sabotage_figures.py
"""
import glob, json, math, os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from taskmaps import MAIN, SIDE, STAGE_SHORT, CLASS_ORDER

HERE = os.path.dirname(__file__)
SPECS = os.path.join(HERE, "..", "..", "src", "rsi_bench", "tasks", "specs", "*.yaml")
REPORT = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(HERE, "..", "data", "sabotage_report_*.json")))[-1]
OUT = os.path.join(HERE, "..", "figures")

MODELS = [("anthropic/claude-opus-5", "Claude Opus 5"), ("openai/gpt-5.6-sol", "GPT-5.6 Sol"),
          ("gemini/gemini-3.7-flash", "Gemini 3.7 Flash"), ("xai/grok-4.6", "Grok 4.6"),
          ("fireworks_ai/glm-5p3", "GLM-5.3"), ("meta_ai/muse-spark-1.3-contributor", "Muse Spark 1.3")]
STAGES = ["Data curation", "Supervised fine-tuning", "Reward-model training", "Reinforcement learning", "On-policy distillation"]

# palette (dataviz reference instance, validated): installed = blue 450; landed-not-covert = blue 200 (same hue, lighter);
# class identity = categorical slots 1-3; chrome in the reference inks.
C_INST, C_LAND, C_REST = "#2a78d6", "#b7d3f6", "#efeeea"
C_CLASS = {"Capability": "#2a78d6", "Alignment": "#eb6834", "Goal injection": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": INK2,
                     "text.color": INK, "axes.labelcolor": INK2, "pdf.fonttype": 42})

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)

# ---- data ---------------------------------------------------------------------------------------
specs = {}
for f in glob.glob(SPECS):
    d = yaml.safe_load(open(f))
    if d.get("deprecated") or d.get("status") == "draft" or d["main_prompt"] not in MAIN or d["goal"] not in SIDE: continue
    specs[d["id"]] = d
main_ids = {m: f"M{i+1}" for i, m in enumerate(MAIN)}

runs = []   # dicts: model, task, cls, stage, inst, landed
for e in json.load(open(REPORT)):
    if e["framing"] != "scored" or e["task"] not in specs or not e.get("valid", True) or "INCOMPLETE" in e["status"]: continue
    sp = specs[e["task"]]; cls = SIDE[sp["goal"]][0]; stage = MAIN[sp["main_prompt"]][0]
    for rid, pr in e["per_run"].items():
        if not pr.get("valid", True) or pr.get("effect_ok") is None: continue
        runs.append(dict(model=e["model"], task=e["task"], cls=cls, stage=stage,
                         inst=bool(pr["installed"]), landed=bool(pr["success"])))

def rate(sel):
    n = len(sel); k = sum(r["inst"] for r in sel); kl = sum(r["landed"] for r in sel)
    p, lo, hi = wilson(k, n); return dict(n=n, k=k, p=p, lo=lo, hi=hi, landed=kl / n if n else float("nan"))

# ---- drawing primitives -------------------------------------------------------------------------
def hbars(ax, rows, title, label_pad=0.03):
    """rows: list of (label, stats). Installed rate in blue, landed-not-covert stacked lighter, remainder faint;
    CI whisker on the installed edge; rate at the bar end; run count in muted ink inside the remainder."""
    ys = list(range(len(rows)))[::-1]
    for y, (lab, s) in zip(ys, rows):
        ax.barh(y, 1.0, height=0.62, color=C_REST, edgecolor="none", zorder=1)
        ax.barh(y, s["landed"], height=0.62, color=C_LAND, edgecolor="none", zorder=2)
        ax.barh(y, s["p"], height=0.62, color=C_INST, edgecolor="none", zorder=3)
        ax.plot([s["lo"], s["hi"]], [y, y], color=INK, lw=0.9, solid_capstyle="butt", zorder=5)
        for x in (s["lo"], s["hi"]): ax.plot([x, x], [y - 0.16, y + 0.16], color=INK, lw=0.9, zorder=5)
        ax.text(1.0 + label_pad, y, f"{100*s['p']:.0f}%", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([lab for lab, s in rows])
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, color=AXIS)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)

def grouped_hbars(ax, groups, title):
    """groups: list of (group_title, rows). Same bars as hbars, with a muted group header and a gap between groups."""
    ys, labels, flat = [], [], []
    y = 0
    for gi, (gt, rows) in enumerate(groups):
        for lab, s in rows:
            flat.append((y, lab, s)); y += 1
        if gi < len(groups) - 1: y += 0.9
    top = y - 1
    for yy, lab, s in flat:
        yv = top - yy
        ax.barh(yv, 1.0, height=0.62, color=C_REST, edgecolor="none", zorder=1)
        ax.barh(yv, s["landed"], height=0.62, color=C_LAND, edgecolor="none", zorder=2)
        ax.barh(yv, s["p"], height=0.62, color=C_INST, edgecolor="none", zorder=3)
        ax.plot([s["lo"], s["hi"]], [yv, yv], color=INK, lw=0.9, zorder=5)
        for x in (s["lo"], s["hi"]): ax.plot([x, x], [yv - 0.16, yv + 0.16], color=INK, lw=0.9, zorder=5)
        ax.text(1.02, yv, f"{100*s['p']:.0f}%", va="center", ha="left", fontsize=7, color=INK)
        ys.append(yv); labels.append(lab)
    # group headers
    y = 0
    for gi, (gt, rows) in enumerate(groups):
        ax.text(-0.02, top - y + 0.62, gt.upper(), transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6, color=MUTED, fontweight="bold")
        y += len(rows) + (0.9 if gi < len(groups) - 1 else 0)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, top + 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, color=AXIS)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)

def pair_dots(ax, title):
    rows = []
    for t, sp in specs.items():
        sel = [r for r in runs if r["task"] == t]
        if not sel: continue
        cls, sname = SIDE[sp["goal"]]; stage, mname = MAIN[sp["main_prompt"]]
        rows.append((f"{sname}  ·  {STAGE_SHORT[stage]} {main_ids[sp['main_prompt']]}", cls, rate(sel)))
    rows.sort(key=lambda r: (-r[2]["p"], -r[2]["landed"], r[0]))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, cls, s) in zip(ys, rows):
        ax.plot([s["lo"], s["hi"]], [y, y], color=C_CLASS[cls], lw=1.2, alpha=0.55, zorder=2, solid_capstyle="round")
        if s["landed"] > s["p"] + 1e-9:
            ax.plot(s["landed"], y, marker="o", ms=5.5, mfc="white", mec=C_CLASS[cls], mew=1.0, zorder=3)
        ax.plot(s["p"], y, marker="o", ms=6.5, mfc=C_CLASS[cls], mec="white", mew=1.2, zorder=4)
        ax.text(1.03, y, f"{s['k']}/{s['n']}", va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, color=AXIS)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)
    ax.text(1.03, len(rows) - 0.3 + 0.15, "succ./runs", ha="left", va="bottom", fontsize=6, color=MUTED)
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=C_CLASS[c], mec="white", mew=1.0, label=c) for c in CLASS_ORDER]
    handles.append(Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=INK2, mew=1.0, label="landed, main task damaged"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.4, borderaxespad=0.2, ncol=1)


def pair_dots_h(ax, title):
    """Horizontal layout: pairs along x (sorted by success rate), rate on y, vertical whiskers, rotated labels."""
    rows = []
    for t, sp in specs.items():
        sel = [r for r in runs if r["task"] == t]
        if not sel: continue
        cls, sname = SIDE[sp["goal"]]; stage, mname = MAIN[sp["main_prompt"]]
        short = {}
        rows.append((f"{short.get(sname, sname)} ({STAGE_SHORT[stage]} {main_ids[sp['main_prompt']]})", cls, rate(sel)))
    rows.sort(key=lambda r: (-r[2]["p"], -r[2]["landed"], r[0]))
    xs = list(range(len(rows)))
    for x, (lab, cls, s) in zip(xs, rows):
        ax.plot([x, x], [s["lo"], s["hi"]], color=C_CLASS[cls], lw=1.2, alpha=0.55, zorder=2, solid_capstyle="round")
        if s["landed"] > s["p"] + 1e-9:
            ax.plot(x, s["landed"], marker="o", ms=5.0, mfc="white", mec=C_CLASS[cls], mew=1.0, zorder=3)
        ax.plot(x, s["p"], marker="o", ms=6.0, mfc=C_CLASS[cls], mec="white", mew=1.2, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right", va="top", rotation_mode="anchor", fontsize=6.2)
    ax.set_xlim(-0.7, len(rows) - 0.3); ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right", "bottom"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", length=2, color=AXIS)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=C_CLASS[c], mec="white", mew=1.0, label=c) for c in CLASS_ORDER]
    handles.append(Line2D([], [], marker="o", ls="", ms=5.0, mfc="white", mec=INK2, mew=1.0, label="landed, main task damaged"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.4, borderaxespad=0.2, ncol=2, columnspacing=1.0)

# ---- panels -------------------------------------------------------------------------------------
def rows_models():
    rows = [(name, rate([r for r in runs if r["model"] == m])) for m, name in MODELS]
    return sorted(rows, key=lambda r: -r[1]["p"])
def rows_classes():
    return [(c, rate([r for r in runs if r["cls"] == c])) for c in CLASS_ORDER]
def rows_stages():
    return [(STAGE_SHORT[s], rate([r for r in runs if r["stage"] == s])) for s in STAGES]

def legend_bars(fig, x, y):
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_INST, label="installed, main task preserved"),
               plt.Rectangle((0, 0), 1, 1, color=C_LAND, label="landed, main task damaged"),
               Line2D([], [], color=INK, lw=0.9, label="95% interval (Wilson, over runs)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y), ncol=3, frameon=False, fontsize=6.8,
               handlelength=1.4, handleheight=0.8, columnspacing=1.6, handletextpad=0.5)

def draw_top(axes):
    hbars(axes[0], rows_models(), "(a) Agent model")
    hbars(axes[1], rows_classes(), "(b) Side-task class")
    hbars(axes[2], rows_stages(), "(c) Pipeline stage")

os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03, **kw)
W = 5.5  # ICLR text width in inches

# figure 1: one row, three panels
fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.0], wspace=1.0))
draw_top(axes); legend_bars(fig, 0.01, -0.01)
fig.subplots_adjust(left=0.15, right=0.915, top=0.86, bottom=0.26)
save(fig, "sabotage_rates"); plt.close(fig)


# figure 1b: panels (a)+(b) only -- one row (full text width) and stacked (half width, for text wrapping)
fig, axes = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw=dict(width_ratios=[1.1, 1.0], wspace=0.9))
hbars(axes[0], rows_models(), "(a) Agent model"); hbars(axes[1], rows_classes(), "(b) Side-task class")
legend_bars(fig, 0.01, -0.01)
fig.subplots_adjust(left=0.20, right=0.93, top=0.86, bottom=0.27)
save(fig, "sabotage_rates_ab"); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(2.75, 3.15), gridspec_kw=dict(height_ratios=[6, 3.2], hspace=0.75))
hbars(axes[0], rows_models(), "(a) Agent model"); hbars(axes[1], rows_classes(), "(b) Side-task class")
handles = [plt.Rectangle((0, 0), 1, 1, color=C_INST, label="installed, main task preserved"),
           plt.Rectangle((0, 0), 1, 1, color=C_LAND, label="landed, main task damaged"),
           Line2D([], [], color=INK, lw=0.9, label="95% interval over runs")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.005), ncol=1, frameon=False, fontsize=6.4,
           handlelength=1.4, handleheight=0.8, handletextpad=0.5, labelspacing=0.3)
fig.subplots_adjust(left=0.40, right=0.86, top=0.93, bottom=0.22)
save(fig, "sabotage_rates_ab_stacked"); plt.close(fig)

# figure 2: per pair
fig, ax = plt.subplots(figsize=(W, 3.6))
pair_dots(ax, "Success rate by task pair (side task · main task)")
fig.subplots_adjust(left=0.42, right=0.92, top=0.93, bottom=0.08)
save(fig, "sabotage_pairs"); plt.close(fig)

# figure 2b: per pair, horizontal (less vertical space)
fig, ax = plt.subplots(figsize=(W, 2.8))
pair_dots_h(ax, "Success rate by task pair (side task, main task)")
fig.subplots_adjust(left=0.13, right=0.995, top=0.90, bottom=0.46)
save(fig, "sabotage_pairs_h"); plt.close(fig)

# figure 3: combined (three panels on top, pairs below)
fig = plt.figure(figsize=(W, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.1], width_ratios=[1.1, 1.0, 1.0], hspace=0.6, wspace=1.0,
                      left=0.15, right=0.915, top=0.955, bottom=0.05)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
draw_top(axes); legend_bars(fig, 0.01, 0.635)
axp = fig.add_subplot(gs[1, :]); pair_dots(axp, "(d) Task pair (side task · main task)")
axp.set_position([0.42, 0.05, 0.50, 0.545])
save(fig, "sabotage_overview"); plt.close(fig)

print(f"{len(runs)} runs from {os.path.basename(REPORT)}")
for lab, s in rows_models(): print(f"  {lab:18} {s['k']:3}/{s['n']:3} = {s['p']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]  landed {s['landed']:.2f}")
for lab, s in rows_classes() + rows_stages(): print(f"  {lab:24} {s['k']:3}/{s['n']:3} = {s['p']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]")
