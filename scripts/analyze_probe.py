"""分析 D2 probe 的 validation 落盘（trainer.validation_data_dir 下的 <step>.jsonl）。

用法：python -I scripts/analyze_probe.py runs/<run_id>
读 val_greedy/*.jsonl 和 val_sample/*.jsonl（各自目录下应只有一个文件）。
同一道题由第一步的 input 文本识别（同题首步 prompt 相同：问题 + 确定性的首次检索结果）。
"""
import collections
import json
import math
import pathlib
import re
import sys

SEARCH_RE = re.compile(r'"name"\s*:\s*"search"')


def load(d: pathlib.Path) -> list[dict]:
    files = sorted(d.glob("*.jsonl"))
    assert len(files) == 1, f"{d}: expected exactly one jsonl, got {files}"
    return [json.loads(line) for line in files[0].open(encoding="utf-8")]


def end_reason(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return "answer"
    if "<tool_call>" in text:
        return "tool_call_unparsed"
    return "no_answer_no_tool"


def behavior(trajs: list[dict]) -> dict:
    steps = collections.Counter(t["num_steps"] for t in trajs)
    reasons = collections.Counter(end_reason(t["steps"][-1]["output"]) for t in trajs)
    searches = [sum(len(SEARCH_RE.findall(s["output"])) for s in t["steps"][:-1]) for t in trajs]
    n = len(trajs)
    return {
        "n_traj": n,
        "num_steps": dict(sorted(steps.items())),
        "end_reason_frac": {k: round(v / n, 3) for k, v in reasons.most_common()},
        "searches_per_traj_mean": round(sum(searches) / n, 3),
        "frac_traj_with_extra_search": round(sum(s > 0 for s in searches) / n, 3),
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> None:
    run = pathlib.Path(sys.argv[1])
    out = {}
    g = load(run / "val_greedy")
    out["greedy"] = {"em": round(sum(t["score"] > 0 for t in g) / len(g), 4), **behavior(g)}

    s = load(run / "val_sample")
    groups = collections.defaultdict(list)
    for t in s:
        groups[t["steps"][0]["input"]].append(t["score"] > 0)
    sizes = collections.Counter(len(v) for v in groups.values())
    res = {"n_questions": len(groups), "group_sizes": dict(sizes)}
    for k in (1, 2, 4, 8):
        res[f"pass@{k}"] = round(sum(pass_at_k(len(v), sum(v), k) for v in groups.values()) / len(groups), 4)
    cs = [sum(v) for v in groups.values()]
    res["all_wrong_frac"] = round(sum(c == 0 for c in cs) / len(cs), 4)
    res["all_correct_frac"] = round(sum(c == len(v) for c, v in zip(cs, groups.values())) / len(cs), 4)
    res["mixed_frac"] = round(1 - res["all_wrong_frac"] - res["all_correct_frac"], 4)
    out["sample"] = {**res, **behavior(s)}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (run / "probe_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
