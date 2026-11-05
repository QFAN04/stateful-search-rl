"""统计每个训练步里"有效题"的数量：同一道题的 G 条轨迹有对有错（mixed）。

GRPO 中全对或全错的组优势为 0，不贡献梯度，所以每步真正提供信号的题数 = mixed 组数。
用法：python scripts/count_mixed_groups.py runs/<run_id> [runs/<run_id> ...]
rollout 文件没有 uid 字段，按第一步 prompt 里的 "### User Query" 段落分组。
"""
import collections
import json
import pathlib
import sys


def question(inp: str) -> str:
    return inp.split("### User Query", 1)[1].split("###", 1)[0].strip()


def count(path: pathlib.Path) -> dict:
    groups = collections.defaultdict(list)
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        groups[question(r["input"])].append(float(r["score"] or 0.0) > 0.5)
    n_right = [sum(v) for v in groups.values()]
    sizes = [len(v) for v in groups.values()]
    return {
        "prompts": len(groups),
        "group_sizes": sorted(set(sizes)),
        "mixed": sum(0 < k < s for k, s in zip(n_right, sizes)),
        "all_right": sum(k == s for k, s in zip(n_right, sizes)),
        "all_wrong": sum(k == 0 for k in n_right),
    }


def main() -> None:
    for run in map(pathlib.Path, sys.argv[1:]):
        files = sorted((run / "rollouts").glob("*.jsonl"), key=lambda p: int(p.stem))
        mixed_total = prompts_total = 0
        for f in files:
            c = count(f)
            mixed_total += c["mixed"]
            prompts_total += c["prompts"]
            print(run.name, f"step {f.stem}", json.dumps(c))
        if prompts_total:
            print(run.name, f"mixed fraction {mixed_total / prompts_total:.3f} over {len(files)} steps")


if __name__ == "__main__":
    main()
