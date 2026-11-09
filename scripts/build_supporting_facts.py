"""E4 笔记奖励用：从 HF `hotpotqa/hotpot_qa` distractor（train + validation）建「问题文本 → supporting fact 标题列表」，
并核对 Agent-R1 训练 / 验证 parquet 里的每道题都能匹配上。

用法：python -I scripts/build_supporting_facts.py
输入：data/raw/hotpot_qa_hf/distractor_{train_0000,train_0001,validation}.parquet、data/raw/agent-r1-data/hotpotqa/{train,validation}.parquet
输出：data/processed/hotpotqa_sf_titles.json（{问题: [标题, ...]}）及其 sha256
"""
import hashlib
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "raw" / "hotpot_qa_hf"
AR1 = ROOT / "data" / "raw" / "agent-r1-data" / "hotpotqa"
OUT = ROOT / "data" / "processed" / "hotpotqa_sf_titles.json"


def main():
    sf = {}
    for f in sorted(HF.glob("distractor_*.parquet")):
        df = pd.read_parquet(f, columns=["question", "supporting_facts"])
        for q, s in zip(df.question, df.supporting_facts):
            sf[q.strip()] = sorted({str(t) for t in s["title"]})
    for name in ("train", "validation"):
        d = pd.read_parquet(AR1 / f"{name}.parquet")
        qs = d.prompt.map(lambda p: p[0]["content"].strip())
        miss = int((~qs.isin(sf.keys())).sum())
        print(f"{name}: {len(qs)} 题，未匹配 {miss}")
        assert miss == 0, f"{name} 有 {miss} 道题匹配不到 supporting facts"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sf, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, len(sf), "questions, sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
