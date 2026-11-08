"""D9 初步分析：配对检验、2Wiki 题型分层、失败类型（只读 runs/d9_*，不跑模型）。

用法：python -I scripts/analyze_d9.py [runs_dir]
- 配对检验：同一题在两组的对错，McNemar 精确检验（双侧二项）
- 2Wiki 题型（启发式，数据里没有官方 type 字段）：答案 yes/no → bool 比较；问句含比较词 → 选择型比较；其余 → 组合/推理
- 失败类型：没给 <answer>（预算用完或放弃）、答了"无法回答"、用满步数
"""
import glob
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
POL = ["c0", "c1", "c2", "c3"]
CMP_WORDS = re.compile(r"\b(first|earlier|later|older|younger|longer|shorter|more|less|both|same|born first|died first|which film|which one)\b", re.I)
CANT = re.compile(r"cannot be (answered|determined)|not (available|mentioned|found)|unknown|insufficient", re.I)


def load(ds, p, x1=False):
    d = RUNS / f"d9_{ds}{'_x1' if x1 else ''}_{p}_q3-4b-2507_lora_pro6000_20261108" / "val"
    out = {}
    for f in glob.glob(str(d / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            q = r["input"].split("### User Query")[1].split("###")[0].strip() if "### User Query" in r["input"] else r["input"][-300:]
            out[q] = r
    return out


def mcnemar(a, b, keys):
    """返回 (a 对 b 错, a 错 b 对, 双侧精确 p)。"""
    n10 = sum(1 for k in keys if a[k]["score"] > 0 and b[k]["score"] == 0)
    n01 = sum(1 for k in keys if a[k]["score"] == 0 and b[k]["score"] > 0)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)


def qtype(q, gt):
    if str(gt).strip().lower() in ("yes", "no"):
        return "bool比较"
    if CMP_WORDS.search(q):
        return "选择比较"
    return "组合/推理"


def last_output(r):
    return r["output"] if isinstance(r["output"], str) else str(r["output"])


def main():
    for ds in ["test", "musique", "2wiki"]:
        g = {p: load(ds, p) for p in POL}
        keys = set.intersection(*(set(v) for v in g.values()))
        print(f"\n===== {ds}（共同题 {len(keys)}）=====")
        print("EM:", {p: round(sum(g[p][k]['score'] for k in keys) / len(keys), 3) for p in POL})
        for a, b in [("c0", "c1"), ("c0", "c2"), ("c3", "c1"), ("c3", "c2"), ("c3", "c0"), ("c1", "c2")]:
            n10, n01, pv = mcnemar(g[a], g[b], keys)
            print(f"  {a} vs {b}: {a}对{b}错 {n10}，{a}错{b}对 {n01}，McNemar p={pv:.3g}")
        # 失败类型
        for p in POL:
            rs = [g[p][k] for k in keys]
            no_ans = sum(1 for r in rs if "<answer>" not in last_output(r))
            cant = sum(1 for r in rs if "<answer>" in last_output(r) and CANT.search(last_output(r).split("<answer>")[-1][:200]))
            maxed = sum(1 for r in rs if r.get("num_steps", 0) >= 5)
            wrong = sum(1 for r in rs if r["score"] == 0)
            print(f"  {p}: 错 {wrong}｜其中无 <answer> {no_ans}、答'无法回答' {cant}｜用满 5 步 {maxed}｜步数分布 {dict(sorted(Counter(r.get('num_steps') for r in rs).items()))}")
        if ds == "2wiki":
            print("  -- 2Wiki 按题型（启发式）--")
            by = defaultdict(list)
            for k in keys:
                gt = g["c0"][k]["gts"]
                gt = gt if isinstance(gt, str) else (gt[0] if isinstance(gt, list) and gt else str(gt))
                by[qtype(k, gt)].append(k)
            for t, ks in sorted(by.items()):
                print(f"  {t}（{len(ks)} 题）:", {p: round(sum(g[p][k]['score'] for k in ks) / len(ks), 3) for p in POL})
        # 按 C0 的检索轮数分层（1 步 vs ≥2 步）
        for lab, cond in [("C0 1 步作答", lambda r: r.get("num_steps", 0) <= 1), ("C0 ≥2 步", lambda r: r.get("num_steps", 0) >= 2)]:
            ks = [k for k in keys if cond(g["c0"][k])]
            if ks:
                print(f"  {lab}（{len(ks)} 题）:", {p: round(sum(g[p][k]['score'] for k in ks) / len(ks), 3) for p in POL})


if __name__ == "__main__":
    main()
