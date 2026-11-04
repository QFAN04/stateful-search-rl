"""LoRA 断点续训检查（GPU 机器上运行，需要 torch；safetensors 可选）。

场景：第一次运行训练到 A 步（每 SAVE_FREQ 步存档），第二次用 resume_mode=auto、total_training_steps=B 续跑。
检查：
  1. 第一次的 metrics 步号是 1..A，第二次是 A+1..B（曲线连续、不重复、不从头开始）；
  2. 第二次的训练日志里出现从 global_step_A 加载的记录；
  3. LoRA 权重真的被加载了，而不是重新初始化：LoRA 的 B 矩阵初始化为 0，
     如果续训时重新初始化，|B_B - B_A| 会和 |B_A| 同一量级；正确续训时只多走了 B-A 步，差值应明显更小。
     判据：|B_B - B_A| / |B_A| < 0.5（A=10、B=12 时，正常约 0.1–0.3）。

用法：python scripts/check_resume.py runs/<run_id> --a 10 --b 12
任何一项不通过都以非零退出码结束；结果同时写入 runs/<run_id>/resume_check.txt。
"""
import argparse
import json
import pathlib
import re
import sys


def steps_of(path: pathlib.Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.open(encoding="utf-8") if line.strip()]


def lora_b(step_dir: pathlib.Path) -> dict:
    """读某个 checkpoint 的全部 lora_B 张量（name -> tensor）。"""
    import torch

    out = {}
    for f in sorted(step_dir.rglob("adapter_model.safetensors")):
        from safetensors.torch import load_file

        out.update({k: v.float() for k, v in load_file(str(f)).items() if "lora_B" in k})
        if out:
            return out
    for f in sorted(step_dir.rglob("model_world_size_*_rank_*.pt")):
        sd = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in sd.items():
            if "lora_B" in k:
                v = getattr(v, "_local_tensor", v)  # FSDP2 存的是 DTensor
                out[f"{f.name}:{k}"] = v.float()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--a", type=int, required=True, help="第一次运行结束（且存档）的步数")
    ap.add_argument("--b", type=int, required=True, help="续跑后的总步数")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir)
    lines, ok = [], True

    def report(passed: bool, msg: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(("PASS " if passed else "FAIL ") + msg)

    s1, s2 = steps_of(run / "metrics.attempt1.jsonl"), steps_of(run / "metrics.attempt2.jsonl")
    report(s1 == list(range(1, args.a + 1)), f"attempt1 steps {s1[:3]}...{s1[-2:]}")
    report(s2 == list(range(args.a + 1, args.b + 1)), f"attempt2 steps {s2}")

    log = (run / "train.attempt2.log").read_text(encoding="utf-8", errors="replace")
    hit = re.search(rf"[^\n]*global_step_{args.a}[^\n]*", log)
    report(bool(hit), f"attempt2 log mentions global_step_{args.a}: {hit.group(0)[:160] if hit else None}")

    ckpt = run / "ckpt"
    try:
        ba, bb = lora_b(ckpt / f"global_step_{args.a}"), lora_b(ckpt / f"global_step_{args.b}")
        common = sorted(set(ba) & set(bb))
        if not common:
            report(False, f"no lora_B tensors found (A: {len(ba)}, B: {len(bb)}) — 检查 checkpoint 格式")
        else:
            import torch

            na = torch.sqrt(sum((ba[k] ** 2).sum() for k in common)).item()
            nd = torch.sqrt(sum(((bb[k] - ba[k]) ** 2).sum() for k in common)).item()
            ratio = nd / na if na > 0 else float("inf")
            report(na > 0 and ratio < 0.5, f"lora_B tensors={len(common)} |B_A|={na:.4g} |B_B-B_A|={nd:.4g} ratio={ratio:.3f}")
    except Exception as e:  # noqa: BLE001
        report(False, f"lora weight check error: {e!r}")

    lines.append("RESUME_CHECK " + ("PASS" if ok else "FAIL"))
    text = "\n".join(lines)
    print(text)
    (run / "resume_check.txt").write_text(text + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
