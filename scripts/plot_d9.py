"""D9 图：EM–token 帕累托（RQ1/RQ3）和按检索轮数分层（RQ3）。只读 runs/d9_*，输出 docs/figures/*.png。

用法：python -I scripts/plot_d9.py [runs_dir]
配色：dataviz 参考调色板分类色前 4 个（已用 validate_palette.js 验证；黄色/青色对比度低于 3:1，按规则每个点都直接标注组名）。
"""
import glob
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_d9 import load  # noqa: E402

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
POL = ["c0", "c1", "c2", "c3"]
NAME = {"c0": "C0 append", "c1": "C1 window", "c2": "C2 notes", "c3": "C3 rule"}
COLOR = {"c0": "#2a78d6", "c1": "#eb6834", "c2": "#1baf7a", "c3": "#eda100"}
MARK = {"c0": "o", "c1": "s", "c2": "D", "c3": "^"}
DS = [("test", "HotpotQA test (n=1000)"), ("musique", "MuSiQue (n=500)"), ("2wiki", "2Wiki (n=500)")]
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def metric(ds, p, x1=False):
    d = RUNS / f"d9_{ds}{'_x1' if x1 else ''}_{p}_q3-4b-2507_lora_pro6000_20261108"
    m = [json.loads(line) for f in sorted(glob.glob(str(d / "metrics.attempt*.jsonl"))) for line in open(f, encoding="utf-8")]
    v = [o["data"] for o in m if any("val-core" in k for k in o["data"])][-1]
    g = lambda s: next(y for k, y in v.items() if k.endswith("/" + s + "/mean@1"))  # noqa: E731
    return g("acc"), g("prompt_tokens")


def b1(ds):
    return json.load(open(RUNS / f"d9_b0b1_{ds}_q3-4b-2507_pro6000_20261108" / "summary.json", encoding="utf-8"))["b1"]["em"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pareto():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF)
    for ax, (ds, title) in zip(axes, DS):
        style(ax)
        n = 1000 if ds == "test" else 500
        for p in POL:
            em5, tok5 = metric(ds, p)
            em10, tok10 = metric(ds, p, x1=True)
            se = math.sqrt(em5 * (1 - em5) / n)
            ax.plot([tok5, tok10], [em5, em10], color=COLOR[p], linewidth=1.2, alpha=0.6, zorder=2)
            ax.errorbar(tok5, em5, yerr=se, fmt=MARK[p], color=COLOR[p], markersize=8, capsize=0, elinewidth=1.2,
                        markeredgecolor=SURF, markeredgewidth=2, zorder=3)
            ax.scatter(tok10, em10, marker=MARK[p], s=55, facecolors=SURF, edgecolors=COLOR[p], linewidths=1.6, zorder=3)
            ax.annotate(NAME[p], (tok5, em5), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_title(f"{title}\nB1 single retrieval = {b1(ds):.3f} (off-scale)", fontsize=10.5, color=INK, loc="left")
        ax.margins(x=0.18, y=0.15)
        ax.set_xlabel("mean prompt tokens per step", fontsize=9, color=MUTED)
    axes[0].set_ylabel("EM (sealed / OOD)", fontsize=9, color=MUTED)
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=7, label="max 5 steps"),
                            Line2D([], [], marker="o", color=MUTED, markerfacecolor=SURF, linestyle="", markersize=7, label="max 10 steps")],
                   frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.01, 0.005, "Filled = max 5 steps (training setting), ±1 binomial SE; hollow = max 10 steps (X1). "
             "Up-left is better. Source: runs/d9_*_pro6000_20261108.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "d9_pareto.png", dpi=160)


def strata():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=SURF, sharey=True)
    for ax, (ds, title) in zip(axes, DS):
        style(ax)
        ax.grid(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        g = {p: load(ds, p) for p in POL}
        keys = set.intersection(*(set(v) for v in g.values()))
        groups = [("C0 answered in 1 step", [k for k in keys if g["c0"][k].get("num_steps", 0) <= 1]),
                  ("C0 needed ≥2 steps", [k for k in keys if g["c0"][k].get("num_steps", 0) >= 2])]
        w = 0.18
        for gi, (lab, ks) in enumerate(groups):
            for pi, p in enumerate(POL):
                em = sum(g[p][k]["score"] for k in ks) / len(ks)
                x = gi + (pi - 1.5) * (w + 0.02)
                ax.bar(x, em, width=w, color=COLOR[p], edgecolor=SURF, linewidth=2, label=NAME[p] if gi == 0 else None)
                ax.text(x, em + 0.01, f"{em:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks([0, 1], [f"{lab}\n(n={len(ks)})" for lab, ks in groups], fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("EM", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.text(0.01, 0.005, "Questions split by how many steps C0 used (proxy for multi-hop need). Max 5 steps. "
             "Source: runs/d9_*_pro6000_20261108; scripts/analyze_d9.py.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "d9_by_rounds.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pareto()
    strata()
    print("wrote", OUT / "d9_pareto.png", OUT / "d9_by_rounds.png")
