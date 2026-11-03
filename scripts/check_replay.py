"""D1 Go/No-Go 检查：从 trainer.rollout_data_dir 的 <step>.jsonl 回放轨迹。

用法：python -I scripts/check_replay.py runs/<run_id>/rollouts --expect-steps 12 --expect-trajs 32 [--show 1]
检查：期望的训练步文件都在；每个文件轨迹数正确、trajectory_uid 唯一；每条轨迹 num_steps>0，
step_index 严格等于 range(num_steps)，每步 input 非空、output 为字符串。任何问题都以非零退出码结束。
另外按最后一步输出统计终止原因（含 <answer> / 含 <tool_call> / 都没有），用于判断"一步作答"是否是格式失败。
注意：这是"逐步文本可审阅"的检查；文件里是 skip_special_tokens 解码后的文本，不能用于 token 级重建。
"""
import argparse
import collections
import json
import pathlib
import sys


def end_reason(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return "answer"
    if "<tool_call>" in text:
        return "tool_call_unparsed"
    return "no_answer_no_tool"


def check_traj(t: dict) -> list[str]:
    errs = []
    n = t.get("num_steps")
    steps = t.get("steps")
    if not isinstance(n, int) or n <= 0:
        errs.append(f"num_steps={n!r}")
    if not isinstance(steps, list) or len(steps) != n:
        errs.append(f"len(steps)={len(steps) if isinstance(steps, list) else None} != num_steps={n}")
        return errs
    if [s.get("step_index") for s in steps] != list(range(n)):
        errs.append(f"step_index={[s.get('step_index') for s in steps]}")
    for s in steps:
        if not isinstance(s.get("input"), str) or not s["input"]:
            errs.append(f"step {s.get('step_index')}: empty input")
        if not isinstance(s.get("output"), str):
            errs.append(f"step {s.get('step_index')}: output not str")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout_dir")
    ap.add_argument("--expect-steps", type=int, required=True, help="期望的训练步数（文件 1..N.jsonl）")
    ap.add_argument("--expect-trajs", type=int, required=True, help="每步期望轨迹数 = train_batch_size * rollout.n")
    ap.add_argument("--show", type=int, default=1, help="打印前几条多步轨迹的逐步内容")
    args = ap.parse_args()

    d = pathlib.Path(args.rollout_dir)
    problems: list[str] = []
    expected = {f"{i}.jsonl" for i in range(1, args.expect_steps + 1)}
    present = {p.name for p in d.glob("*.jsonl")}
    if expected - present:
        problems.append(f"missing files: {sorted(expected - present)}")

    total_steps = collections.Counter()
    total_reasons = collections.Counter()
    shown = 0
    for i in range(1, args.expect_steps + 1):
        f = d / f"{i}.jsonl"
        if not f.exists():
            continue
        trajs = [json.loads(line) for line in f.open(encoding="utf-8")]
        if len(trajs) != args.expect_trajs:
            problems.append(f"{f.name}: {len(trajs)} trajectories, expected {args.expect_trajs}")
        uids = [t.get("trajectory_uid") for t in trajs]
        if len(set(uids)) != len(uids) or None in uids:
            problems.append(f"{f.name}: trajectory_uid missing or duplicated")
        bad = 0
        reasons = collections.Counter()
        for t in trajs:
            errs = check_traj(t)
            if errs:
                bad += 1
                problems.append(f"{f.name} {t.get('trajectory_uid')}: {'; '.join(errs)}")
                continue
            total_steps[t["num_steps"]] += 1
            reasons[end_reason(t["steps"][-1]["output"])] += 1
            if shown < args.show and t["num_steps"] > 1:
                shown += 1
                print(f"=== {f.name} trajectory {t['trajectory_uid']} steps={t['num_steps']} score={t['score']} gts={t.get('gts')}")
                for s in t["steps"]:
                    print(f"--- step {s['step_index']} input[-300:]:\n{s['input'][-300:]}\n--- output[:400]:\n{s['output'][:400]}")
        total_reasons.update(reasons)
        scores = [t.get("score", 0.0) for t in trajs]
        print(f"{f.name}: trajectories={len(trajs)} bad={bad} end_reasons={dict(reasons)} "
              f"mean_score={sum(scores) / max(len(scores), 1):.3f}")

    print(f"TOTAL num_steps={dict(sorted(total_steps.items()))} end_reasons={dict(total_reasons)}")
    if problems:
        print("FAIL:", *problems[:50], sep="\n  ")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
