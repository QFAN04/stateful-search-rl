"""D1 Go/No-Go 检查：从 trainer.rollout_data_dir 的 <step>.jsonl 回放轨迹。

用法：python -I scripts/check_replay.py runs/<run_id>/rollouts [--show 1]
检查每条轨迹能否按 step_index 读出每一步完整的 input / output，并汇总步数与得分分布。
"""
import argparse
import collections
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout_dir")
    ap.add_argument("--show", type=int, default=1, help="打印前几条轨迹的逐步内容")
    args = ap.parse_args()

    files = sorted(pathlib.Path(args.rollout_dir).glob("*.jsonl"), key=lambda p: int(p.stem))
    shown = 0
    for f in files:
        trajs = [json.loads(line) for line in f.open(encoding="utf-8")]
        num_steps = collections.Counter(t["num_steps"] for t in trajs)
        scores = [t["score"] for t in trajs]
        bad = 0
        for t in trajs:
            idx = [s.get("step_index") for s in t["steps"]]
            ok = len(t["steps"]) == t["num_steps"] and idx == sorted(idx) and all(
                s.get("input") and s.get("output") is not None for s in t["steps"]
            )
            bad += not ok
            if shown < args.show:
                shown += 1
                print(f"=== {f.name} trajectory {t['trajectory_uid']} steps={t['num_steps']} score={t['score']} gts={t.get('gts')}")
                for s in t["steps"]:
                    print(f"--- step {s.get('step_index')} input[-300:]:\n{s['input'][-300:]}\n--- output[:300]:\n{s['output'][:300]}")
        print(f"{f.name}: trajectories={len(trajs)} unreplayable={bad} num_steps={dict(sorted(num_steps.items()))} "
              f"mean_score={sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
