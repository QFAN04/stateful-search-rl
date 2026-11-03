"""B0 / B1 只推理基线（PLAN §7），设计为在 Colab 单卡上跑（不需要 Docker / veRL）。

- B0：无检索，只给问题，直接作答
- B1：用问题原文检索一次（与 recipe 的 force_first_search 相同的检索器、top-5、同一语料与索引），再一次作答
两者与 E0 用同一模型、同一 EM 评分（recipes/hotpotqa/reward_fn.compute_score），贪心解码。

用法（仓库根目录）：
  python -I scripts/baselines_b0b1.py --split data/processed/hotpotqa_splits/dev_select.parquet \
      --corpus-dir /content/data/hotpotqa_corpus --out runs/<run_id> [--model Qwen/Qwen3-4B-Instruct-2507] [--limit N]
输出：<out>/b0.jsonl、<out>/b1.jsonl（逐题 prompt / 输出 / 得分）和 <out>/summary.json。
依赖：vllm、transformers、pandas、faiss-cpu、FlagEmbedding。已完成的基线（jsonl 行数等于题数）会跳过。
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

# 检索器（torch / FlagEmbedding）会先在主进程初始化 CUDA，vLLM 默认 fork 出引擎进程会因此报
# "CUDA error: initialization error"，所以强制用 spawn
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "Agent-R1"))

from recipes.hotpotqa.prompts import HOTPOTQA_SYSTEM_PROMPT
from recipes.hotpotqa.reward_fn import compute_score

ANSWER_INSTRUCTION = (
    "### Instructions\nAnswer the User Query. Put only the short final answer inside `<answer></answer>` tags "
    "(no explanation)."
)


def b0_messages(question: str) -> list[dict]:
    user = f"### User Query\n{question}\n\n{ANSWER_INSTRUCTION}"
    return [{"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def b1_messages(question: str, passages: list[str]) -> list[dict]:
    # 段落格式与 recipe PassagePool.passage_list 一致：[i] 文本前 512 字符
    lines = [f"[{i}] {p[:512].replace(chr(10), ' ')}" for i, p in enumerate(passages, start=1)] or ["None"]
    user = (f"### User Query\n{question}\n\n### Retrieved Passages\n" + "\n".join(lines)
            + "\n\n" + ANSWER_INSTRUCTION.replace("Answer the User Query.",
                                                  "Answer the User Query using the Retrieved Passages."))
    return [{"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retrieve(questions: list[str], corpus_dir: str, batch: int = 64) -> list[list[str]]:
    from recipes.hotpotqa.env.search_tool import HotpotQASearchToolLegacy

    tool = HotpotQASearchToolLegacy(corpus_data_dir=corpus_dir)
    out: list[list[str]] = []
    for i in range(0, len(questions), batch):
        for r in tool.batch_execute([{"query": q} for q in questions[i:i + batch]]):
            if not r["success"]:
                raise RuntimeError(f"retrieval failed: {r['content'][:200]}")
            out.append(json.loads(r["content"])["results"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max-model-len", type=int, default=9216)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.split)
    if args.limit:
        df = df.head(args.limit)
    questions = [p[0]["content"] for p in df["prompt"]]
    meta = {"split": args.split, "split_sha256": hashlib.sha256(pathlib.Path(args.split).read_bytes()).hexdigest(),
            "n": len(df), "model": args.model, "max_tokens": args.max_tokens, "decoding": "greedy",
            "retrieval": "recipe HotpotQASearchToolLegacy, query=question, top-5"}

    todo = [b for b in ("b0", "b1") if not ((out / f"{b}.jsonl").exists()
                                           and sum(1 for _ in (out / f"{b}.jsonl").open()) == len(df))]
    if not todo:
        print("both baselines done")
    else:
        msgs = {"b0": [b0_messages(q) for q in questions]}
        if "b1" in todo:
            t = time.time()
            passages = retrieve(questions, args.corpus_dir)
            meta["retrieval_seconds"] = round(time.time() - t, 1)
            msgs["b1"] = [b1_messages(q, ps) for q, ps in zip(questions, passages)]
        llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85, seed=0)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        for b in todo:
            t = time.time()
            outs = llm.chat(msgs[b], sp, use_tqdm=True)
            meta[f"{b}_gen_seconds"] = round(time.time() - t, 1)
            with (out / f"{b}.jsonl").open("w", encoding="utf-8") as f:
                for i, o in enumerate(outs):
                    row = df.iloc[i]
                    text = o.outputs[0].text
                    score = compute_score(row["data_source"], text, row["reward_model"]["ground_truth"],
                                          dict(row["extra_info"]))
                    f.write(json.dumps({"question_id": row["extra_info"]["question_id"], "question": questions[i],
                                        "messages": msgs[b][i], "output": text, "score": score,
                                        "has_answer_tag": "<answer>" in text and "</answer>" in text},
                                       ensure_ascii=False) + "\n")

    summary = dict(meta)
    for b in ("b0", "b1"):
        rows = [json.loads(line) for line in (out / f"{b}.jsonl").open(encoding="utf-8")]
        summary[b] = {"em": round(sum(r["score"] for r in rows) / len(rows), 4),
                      "answer_tag_rate": round(sum(r["has_answer_tag"] for r in rows) / len(rows), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
