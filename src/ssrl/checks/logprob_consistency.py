"""训推 logprob 一致性检查（PLAN §6「训推一致性」）。

Agent-R1 每步样本直接使用 vLLM 返回的 token ids 与 logprobs（接口笔记 §1.3），不存在拼接序列下"上下文与重算不一致"
或重新分词的问题；剩下的不一致来源是推理引擎与训练前向的数值差异（kernel、精度、批次）。本模块测的就是这一项：
同一组 token，vLLM 采样时给出的 logprob 与 HF 训练式前向（bf16）重算的 logprob 之差。

- `compare()`：纯 numpy，按 mask 只统计有效 response token，单元测试见 tests/test_logprob.py
- `main()`：GPU 上运行（Colab 即可）。用 B1 式 prompt（问题 + 检索段落）在 dev_select 上采样，再用 HF 重算并比较
  用法：python -I -m ssrl.checks.logprob_consistency --split <dev_select.parquet> --corpus-dir <dir> --out runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def compare(rollout_lp: np.ndarray, ref_lp: np.ndarray, mask: np.ndarray) -> dict:
    """rollout_lp / ref_lp / mask 形状相同 (B, T)；只在 mask==1 的 token 上统计。

    返回：逐 token |Δlogp| 的均值 / 中位数 / p99 / 最大值，|Δp| 均值，按序列平均的 |Δlogp| 的最大值，
    以及 mean(rollout - ref)（符号表示系统性偏差方向）和有效 token 数。
    """
    rollout_lp, ref_lp, mask = (np.asarray(x, dtype=np.float64) for x in (rollout_lp, ref_lp, mask))
    if not (rollout_lp.shape == ref_lp.shape == mask.shape):
        raise ValueError(f"shape mismatch: {rollout_lp.shape} {ref_lp.shape} {mask.shape}")
    m = mask > 0
    n = int(m.sum())
    if n == 0:
        raise ValueError("mask has no valid tokens")
    diff = (rollout_lp - ref_lp)[m]
    ad = np.abs(diff)
    per_seq = np.where(m, np.abs(rollout_lp - ref_lp), 0).sum(1) / np.maximum(m.sum(1), 1)
    return {
        "n_tokens": n,
        "abs_logp_diff_mean": float(ad.mean()),
        "abs_logp_diff_p50": float(np.percentile(ad, 50)),
        "abs_logp_diff_p99": float(np.percentile(ad, 99)),
        "abs_logp_diff_max": float(ad.max()),
        "signed_logp_diff_mean": float(diff.mean()),
        "abs_prob_diff_mean": float(np.abs(np.exp(rollout_lp[m]) - np.exp(ref_lp[m])).mean()),
        "per_seq_abs_logp_diff_max": float(per_seq[m.sum(1) > 0].max()),
    }


def _pad(rows: list[list[float]], fill: float = 0.0) -> np.ndarray:
    t = max(len(r) for r in rows)
    return np.array([r + [fill] * (t - len(r)) for r in rows], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=64, help="题数")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=9216)
    args = ap.parse_args()

    import gc
    import sys

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "third_party" / "Agent-R1"))
    from baselines_b0b1 import b1_messages, retrieve

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.split).head(args.n)
    questions = [p[0]["content"] for p in df["prompt"]]
    msgs = [b1_messages(q, ps) for q, ps in zip(questions, retrieve(questions, args.corpus_dir))]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True) for m in msgs]

    # 1) vLLM 采样，temperature=1.0：返回的 logprob 即采样分布本身，与 HF 原始 logits 的 log_softmax 可直接比较
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.6, seed=0, dtype="bfloat16")
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=0, seed=0)
    gens = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp)
    resp_ids, rollout = [], []
    for g in gens:
        o = g.outputs[0]
        resp_ids.append(list(o.token_ids))
        rollout.append([lp[t].logprob for t, lp in zip(o.token_ids, o.logprobs)])
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 2) HF 前向重算（bf16，与 FSDP actor 的计算精度一致），逐条计算以免 padding 影响
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    ref = []
    with torch.no_grad():
        for p, r in zip(prompt_ids, resp_ids):
            ids = torch.tensor([p + r], device="cuda")
            logits = model(ids).logits[0, len(p) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r, device="cuda")[:, None])[:, 0]
            ref.append(lp.cpu().tolist())

    mask = _pad([[1.0] * len(r) for r in resp_ids])
    res = compare(_pad(rollout), _pad(ref), mask)
    res.update({"model": args.model, "n_sequences": len(resp_ids), "prompt": "B1-style (question + top-5 passages)",
                "vllm_dtype": "bfloat16", "hf_dtype": "bfloat16", "temperature": 1.0})
    (out / "logprob_consistency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
