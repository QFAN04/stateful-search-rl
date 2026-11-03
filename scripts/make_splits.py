"""从 Agent-R1 处理好的 HotpotQA validation.parquet（官方 dev，7405 题）切出 PLAN §5 的固定划分。

- dev_select：500 题，只用于选 checkpoint / probe / B0、B1 dev 基线
- test：1000 题，与 dev_select 互斥，封存，配置冻结后只跑一次
- probe：dev_select 的前 300 题（PLAN §6 Pre-GRPO probe）
- probe_x8：probe 每题复制 8 行（每行一条独立轨迹）。Agent-R1 的 _validate 在 val_kwargs.n>1 时
  让同题 n 条轨迹共用 uid，落盘时会被合并成一条，所以 probe 采样改为"复制行 + n=1"。

用法：python -I scripts/make_splits.py <validation.parquet> <out_dir>
输出 parquet 到 out_dir，manifest（question_id 列表 + sha256）到 data/manifests/。随机种子固定，可复现。
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

SEED = 20261103
N_DEV, N_TEST, N_PROBE, N_SAMPLES = 500, 1000, 300, 8


def manifest(name: str, ids: list[str], src_sha: str) -> dict:
    blob = "\n".join(ids).encode()
    return {"name": name, "n": len(ids), "seed": SEED, "source_sha256": src_sha,
            "ids_sha256": hashlib.sha256(blob).hexdigest(), "question_ids": ids}


def main() -> None:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    man_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    df = pd.read_parquet(src)
    qids = df["extra_info"].map(lambda e: e["question_id"])
    assert qids.is_unique, "question_id 不唯一"

    perm = np.random.default_rng(SEED).permutation(len(df))
    dev = df.iloc[perm[:N_DEV]].reset_index(drop=True)
    test = df.iloc[perm[N_DEV:N_DEV + N_TEST]].reset_index(drop=True)
    probe = dev.iloc[:N_PROBE].reset_index(drop=True)
    probe_x8 = probe.loc[probe.index.repeat(N_SAMPLES)].reset_index(drop=True)

    for name, part in [("dev_select", dev), ("test", test), ("probe", probe), ("probe_x8", probe_x8)]:
        part.to_parquet(out / f"{name}.parquet")
        if name != "probe_x8":
            ids = part["extra_info"].map(lambda e: e["question_id"]).tolist()
            (man_dir / f"hotpotqa_{name}.json").write_text(json.dumps(manifest(name, ids, src_sha), indent=1))
        print(name, len(part), dict(part["extra_info"].map(lambda e: e["level"]).value_counts()))

    dev_ids = set(dev["extra_info"].map(lambda e: e["question_id"]))
    test_ids = set(test["extra_info"].map(lambda e: e["question_id"]))
    assert not dev_ids & test_ids, "dev_select 与 test 重叠"


if __name__ == "__main__":
    main()
