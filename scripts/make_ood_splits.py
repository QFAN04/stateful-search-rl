"""从 Agent-R1-data 里的 OOD 验证集各抽 500 题（PLAN §5：MuSiQue 500、2Wiki 500，只评测不训练）。

输入：Agent-R1 处理好的 `musique_validation.parquet`（MuSiQue-Ans dev，2417 题）和
`2wikimultihopqa_validation.parquet`（2Wiki dev，12576 题），字段与训练用 HotpotQA parquet 相同。
随机种子与 make_splits.py 相同（20261103），可复现。

用法：python -I scripts/make_ood_splits.py <dir_with_parquets> <out_dir>
输出：<out_dir>/{musique,2wiki}_500.parquet，manifest 写到 data/manifests/ood_{musique,2wiki}_500.json。
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

SEED = 20261103
N = 500
SOURCES = {"musique": "musique_validation.parquet", "2wiki": "2wikimultihopqa_validation.parquet"}


def main() -> None:
    src_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    man_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "manifests"

    for name, fname in SOURCES.items():
        src = src_dir / fname
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        df = pd.read_parquet(src)
        qids = df["extra_info"].map(lambda e: e["question_id"])
        assert qids.is_unique, f"{name}: question_id 不唯一"
        part = df.iloc[np.random.default_rng(SEED).permutation(len(df))[:N]].reset_index(drop=True)
        part.to_parquet(out / f"{name}_500.parquet")
        ids = part["extra_info"].map(lambda e: e["question_id"]).tolist()
        blob = "\n".join(ids).encode()
        man = {"name": f"{name}_500", "n": len(ids), "seed": SEED, "source": fname, "source_rows": len(df),
               "source_sha256": src_sha, "ids_sha256": hashlib.sha256(blob).hexdigest(), "question_ids": ids}
        (man_dir / f"ood_{name}_500.json").write_text(json.dumps(man, indent=1))
        n_alias = int(part["extra_info"].map(lambda e: len(e.get("answers", [])) > 1).sum())
        print(name, len(part), "of", len(df), "| 有多个答案别名的题:", n_alias)


if __name__ == "__main__":
    main()
