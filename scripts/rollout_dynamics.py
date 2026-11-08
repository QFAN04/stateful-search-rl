"""RQ4 补充：从 E0–E3 训练 rollouts 统计每步的零方差组比例和重复检索率，画图并打印 10 步平均。

- 零方差组：同一道题的 8 条轨迹得分全相同（全对或全错），GRPO 优势为 0，不提供梯度。
- 重复检索率：轨迹最后一步 prompt 的 History Actions 里，逐字重复的查询占全部查询的比例
  （C1 每步只显示最近一轮的查询，无法从历史统计，不画）。
- C2 笔记率：有 note 的轨迹比例。
只读 runs/e{0..3}_c*_q3-4b-2507_lora_pro6000_20261106/rollouts/*.jsonl，输出 docs/figures/e0_e3_rollout_dynamics.png。
用法：python -I scripts/rollout_dynamics.py [runs_dir]
"""
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
RUN = {p: f"e{i}_{p}_q3-4b-2507_lora_pro6000_20261106" for i, p in enumerate(["c0", "c1", "c2", "c3"])}
NAME = {"c0": "C0 append", "c1": "C1 window", "c2": "C2 notes", "c3": "C3 rule"}
COLOR = {"c0": "#2a78d6", "c1": "#eb6834", "c2": "#1baf7a", "c3": "#eda100"}
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def query_of(text):
    return text.split("### User Query")[1].split("###")[0].strip() if "### User Query" in text else text[-300:]


def history(text):
    if "### History Actions" not in text:
        return []
    block = text.split("### History Actions")[1].split("###")[0]
    return [line.strip().lower() for line in block.strip().split("\n") if line.strip()]


def stats(p):
    out = {}
    for f in sorted((RUNS / RUN[p] / "rollouts").glob("*.jsonl"), key=lambda x: int(x.stem)):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        groups = defaultdict(list)
        nq = dup = notes = 0
        for r in rows:
            first = r["steps"][0]["input"] if r["steps"] else r["input"]
            groups[query_of(first)].append(float(r["score"]))
            h = history(r["steps"][-1]["input"]) if r["steps"] else []
            nq += len(h)
            dup += len(h) - len(set(h))
            notes += any(int(s.get("num_notes", 0) or 0) > 0 for s in r["steps"])
        zv = sum(1 for v in groups.values() if len(set(v)) == 1) / len(groups)
        out[int(f.stem)] = (zv, dup / max(nq, 1), notes / len(rows), len(groups))
    return out


def smooth(xs, ys, k=10):
    return xs, [sum(ys[max(0, i - k + 1): i + 1]) / len(ys[max(0, i - k + 1): i + 1]) for i in range(len(ys))]


def main():
    S = {p: stats(p) for p in RUN}
    for p in RUN:
        st = sorted(S[p])
        for a, b in [(1, 10), (71, 80), (141, 150)]:
            seg = [S[p][s] for s in st if a <= s <= b]
            print(f"{p} steps {a}-{b}: zero-var {sum(x[0] for x in seg)/len(seg):.3f}  dup-query {sum(x[1] for x in seg)/len(seg):.3f}  "
                  f"note-traj {sum(x[2] for x in seg)/len(seg):.3f}  groups/step {seg[0][3]}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURF)
    panels = [(0, "Zero-variance groups (no GRPO signal)", list(RUN)), (1, "Repeated queries in history", ["c0", "c2", "c3"])]
    for ax, (idx, title, pols) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        for p in pols:
            xs = sorted(S[p])
            xs, ys = smooth(xs, [S[p][s][idx] for s in xs])
            ax.plot(xs, ys, color=COLOR[p], linewidth=2, label=NAME[p])
            ax.annotate(NAME[p].split()[0], (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=9, color=MUTED)
        ax.set_xlim(0, 162)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, "10-step trailing mean; 64 questions × 8 rollouts per step. C1 shows only the latest round of queries, so its "
             "repeats cannot be counted. Source: runs/e{0..3}_*_20261106/rollouts.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "e0_e3_rollout_dynamics.png", dpi=160)
    print("wrote", OUT / "e0_e3_rollout_dynamics.png")


if __name__ == "__main__":
    main()
