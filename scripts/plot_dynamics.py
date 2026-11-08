"""RQ4 训练动力学：E0–E3 的训练 EM、策略熵、每条轨迹步数、每条轨迹 prompt 长度随训练步变化（10 步滑动平均）。

只读 runs/e{0..3}_c*_q3-4b-2507_lora_pro6000_20261106/metrics.attempt*.jsonl，输出 docs/figures/e0_e3_dynamics.png。
用法：python -I scripts/plot_dynamics.py [runs_dir]
"""
import glob
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
RUN = {p: f"e{i}_{p}_q3-4b-2507_lora_pro6000_20261106" for i, p in enumerate(["c0", "c1", "c2", "c3"])}
NAME = {"c0": "C0 append", "c1": "C1 window", "c2": "C2 notes", "c3": "C3 rule"}
COLOR = {"c0": "#2a78d6", "c1": "#eb6834", "c2": "#1baf7a", "c3": "#eda100"}
PANELS = [("critic/score/mean", "Training EM (batch mean)"), ("actor/entropy", "Policy entropy"),
          ("prompt_length/mean", "Prompt tokens per step"), ("prompt_length/trajectory/mean", "Prompt tokens per trajectory")]
# 注：veRL 的 num_turns/mean 在这个 flow 里恒为 2，不是检索步数，所以不画
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def series(p, key):
    rows = [json.loads(line) for f in sorted(glob.glob(str(RUNS / RUN[p] / "metrics.attempt*.jsonl"))) for line in open(f, encoding="utf-8")]
    pts = sorted((o["step"], o["data"][key]) for o in rows if key in o["data"])
    xs, ys = [s for s, _ in pts], [v for _, v in pts]
    sm = [sum(ys[max(0, i - 9): i + 1]) / len(ys[max(0, i - 9): i + 1]) for i in range(len(ys))]
    return xs, sm


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        for p in RUN:
            xs, ys = series(p, key)
            ax.plot(xs, ys, color=COLOR[p], linewidth=2, label=NAME[p])
            ax.annotate(NAME[p].split()[0], (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=9, color=MUTED)
        ax.set_xlim(0, 162)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.01, 0.005, "10-step trailing mean. Source: runs/e{0..3}_c*_q3-4b-2507_lora_pro6000_20261106/metrics.attempt1.jsonl.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "e0_e3_dynamics.png", dpi=160)
    print("wrote", OUT / "e0_e3_dynamics.png")


if __name__ == "__main__":
    main()
