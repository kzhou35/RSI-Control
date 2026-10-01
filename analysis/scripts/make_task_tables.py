"""Generate the main/side task tables for the paper from rsi-bench task specs.

Writes tables/main_side_tables.tex (side-by-side overview, main text) and
tables/task_appendix.tex (one row per live main x side pair, appendix).
Run from the repo root: uv run python analysis/scripts/make_task_tables.py
"""
import glob, os, sys, yaml
sys.path.insert(0, os.path.dirname(__file__))

SPECS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "rsi_bench", "tasks", "specs", "*.yaml")

from taskmaps import MAIN, SIDE, STAGE_SHORT, CLASS_ORDER, MAIN_DESC, SIDE_DESC, EVAL  # noqa: E402

def tex(s): return str(s).replace("_", "\\_").replace("&", "\\&")
def grp(x): return "\\grp{" + str(x) + "}"   # bold group label (table style in main.tex)
def texbrk(s): return str(s).replace("_", "\\_\\allowbreak{}").replace("&", "\\&")

rows = []
for f in sorted(glob.glob(SPECS)):
    d = yaml.safe_load(open(f))
    if d.get("deprecated") or d.get("status") == "draft":
        continue
    if d["main_prompt"] not in MAIN or d["goal"] not in SIDE:
        # specs added after the sweep (e.g. row22/row23) are not in the paper yet: skip, but say so
        print(f"skipping unmapped spec (not in the paper's task set): {d['id']} {d['main_prompt']} {d['goal']}")
        continue
    rows.append(d)

main_ids = {m: f"M{i+1}" for i, m in enumerate(MAIN)}
used_main = {r["main_prompt"] for r in rows}
assert used_main == set(MAIN), set(MAIN) ^ used_main

# ---- overview: two side-by-side tables -------------------------------------------------------
L = []
L.append(r"\begin{table}[t]")
L.append(r"\centering\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.02}")
L.append(r"\caption{RSI Control Arena. \textbf{Left:} the %d main tasks, grouped by pipeline stage. \textbf{Right:} the %d side tasks, grouped by class; the last column lists the main tasks each is paired with, giving %d (main, side) pairs in total. Per-pair evaluations and thresholds are in Appendix~\ref{app:tasks}.}"
         % (len(MAIN), len(SIDE), len(rows)))
L.append(r"\label{tab:tasks}")
L.append(r"\sbox0{\fontsize{8.5}{9.8}\selectfont%")   # two top-aligned tabulars side by side at natural size (8.5pt: this style defines \small and \footnotesize as the same 9pt)
L.append(r"{\renewcommand{\arraystretch}{1.10}%   left has 14 rows vs 16 on the right: stretch it to the same height")
L.append(r"\begin{tabular}[t]{@{}lll@{}}\toprule")
L.append(r"\hrow Stage & ID & Main task \\ \midrule")
prev = None
for m, (stage, name) in MAIN.items():
    if prev is not None and stage != prev: L.append(r"\addlinespace[2pt]")
    L.append(f"{grp(STAGE_SHORT[stage]) if stage != prev else ''} & {main_ids[m]} & {name} \\\\")
    prev = stage
L.append(r"\bottomrule\end{tabular}}\hspace{1.5em}")
L.append(r"\begin{tabular}[t]{@{}lll@{}}\toprule")
L.append(r"\hrow Class & Side task & Main tasks \\ \midrule")
prev = None
for g, (cls, name) in SIDE.items():
    mains = sorted({main_ids[r["main_prompt"]] for r in rows if r["goal"] == g}, key=lambda x: int(x[1:]))
    if not mains: continue
    if prev is not None and cls != prev: L.append(r"\addlinespace[2pt]")
    L.append(f"{grp(cls) if cls != prev else ''} & {name} & {', '.join(mains)} \\\\")
    prev = cls
L.append(r"\bottomrule\end{tabular}}\usebox0")
L.append(r"\vspace{-6pt}")
L.append(r"\end{table}")
open(os.path.join(os.path.dirname(__file__), "..", "tables", "main_side_tables.tex"), "w").write("\n".join(L) + "\n")

# ---- appendix: one row per pair --------------------------------------------------------------
A = []
A.append(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}")
A.append(r"\caption{Every (main task, side task) pair and how it is evaluated. The side-task evaluation is the held-out measurement of the side property on the delivered artifact; an attack run lands only if it reaches the floor and exceeds the same model's benign runs (Section~\ref{sec:exp_setup}). The main-task evaluation lists the cover metrics an attack must preserve. Appendix~\ref{app:eval} describes how each evaluation is built.}")
A.append(r"\label{tab:tasks_full}")
A.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{5.6cm}r>{\raggedright\arraybackslash}p{4.4cm}@{}}\toprule")
A.append(r"\hrow Stage & Main task & Side task & Side-task evaluation & Floor & Main-task evaluation \\ \midrule")
def key(r):
    cls, _ = SIDE[r["goal"]]
    return (CLASS_ORDER.index(cls), list(SIDE).index(r["goal"]), list(MAIN).index(r["main_prompt"]))
def floor(r):
    t = r.get("oracle", {}).get("success_threshold", "?")
    return f"{t} docs" if r["id"].startswith("row04b") else (f"{t:.2f}" if isinstance(t, float) else str(t))
prev_cls = None
for r in sorted(rows, key=key):
    stage, mname = MAIN[r["main_prompt"]]; cls, sname = SIDE[r["goal"]]
    if prev_cls is not None and cls != prev_cls: A.append(r"\addlinespace[3pt]")
    prev_cls = cls
    side_eval, main_eval = EVAL[r["id"]]
    A.append(f"{STAGE_SHORT[stage]} & {main_ids[r['main_prompt']]} {mname} & {sname} & {side_eval} & {floor(r)} & {main_eval} \\\\")
A.append(r"\bottomrule\end{tabular}}\end{table}")
open(os.path.join(os.path.dirname(__file__), "..", "tables", "task_appendix.tex"), "w").write("\n".join(A) + "\n")
# ---- appendix: one-sentence descriptions ------------------------------------------------------
D = []
D.append(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}")
D.append(r"\caption{The main tasks. Each is a stage of the successor's training pipeline that the agent carries out end to end; the deliverable is graded on held-out data.}")
D.append(r"\label{tab:main_desc}")
D.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llp{3.5cm}p{9.6cm}@{}}\toprule")
D.append(r"\hrow Stage & ID & Main task & What the agent is asked to do \\ \midrule")
prev = None
for m, (stage, name) in MAIN.items():
    if prev is not None and stage != prev: D.append(r"\addlinespace[2pt]")
    D.append(f"{grp(STAGE_SHORT[stage]) if stage != prev else ''} & {main_ids[m]} & {name} & {MAIN_DESC[m]} \\\\")
    prev = stage
D.append(r"\bottomrule\end{tabular}}\end{table}")
D.append("")
D.append(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}")
D.append(r"\caption{The side tasks. Each states a property of the successor, or of the artifact used to train it, that runs against the main task's specification; the agent is told the property and nothing about how to install it.}")
D.append(r"\label{tab:side_desc}")
D.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lp{4.0cm}p{9.8cm}@{}}\toprule")
D.append(r"\hrow Class & Side task & Property installed \\ \midrule")
prev = None
for g, (cls, name) in SIDE.items():
    if not any(r["goal"] == g for r in rows): continue
    if prev is not None and cls != prev: D.append(r"\addlinespace[2pt]")
    D.append(f"{grp(cls) if cls != prev else ''} & {name} & {SIDE_DESC[g]} \\\\")
    prev = cls
D.append(r"\bottomrule\end{tabular}}\end{table}")
open(os.path.join(os.path.dirname(__file__), "..", "tables", "task_descriptions.tex"), "w").write("\n".join(D) + "\n")
print(f"{len(rows)} pairs, {len(MAIN)} main tasks, {len(SIDE)} side tasks")
