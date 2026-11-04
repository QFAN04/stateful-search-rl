"""★ 优势广播检查：step-level GRPO 的优势必须"按轨迹算、广播到该轨迹每一步"。

Agent-R1 的做法（core_algos.compute_grpo_outcome_advantage）：
  1. 同一 trajectory_uid 的所有步，把 token 奖励求和 → 轨迹总分；
  2. 按 uid（同一道题的 G 条轨迹）求均值和标准差；
  3. (总分 - 均值) / (标准差 + eps) 作为该轨迹的优势，赋给每一步的每个有效 token。

本模块不依赖 torch，输入 numpy 数组，可以对训练中 dump 的 batch 做检查，也可以在单测里对上游函数做对拍。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def expected_advantages(
    token_level_rewards: np.ndarray,  # (bs, L)
    response_mask: np.ndarray,  # (bs, L)
    index: np.ndarray,  # (bs,) uid：同一道题
    trajectory_uids: np.ndarray,  # (bs,) 同一条轨迹
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
) -> np.ndarray:
    """独立重算一遍期望的优势（bs, L）。标准差用无偏估计（与 torch.std 默认一致）。"""
    step_score = (token_level_rewards * response_mask).sum(-1)
    traj_total: dict = defaultdict(float)
    traj_group: dict = {}
    for s, g, t in zip(step_score, index, trajectory_uids):
        traj_total[t] += float(s)
        assert traj_group.setdefault(t, g) == g, f"trajectory {t} 跨了多个 uid"
    groups: dict = defaultdict(list)
    for t, total in traj_total.items():
        groups[traj_group[t]].append(total)
    stats = {
        g: (0.0, 1.0) if len(v) == 1 else (float(np.mean(v)), float(np.std(v, ddof=1)))
        for g, v in groups.items()
    }
    adv = np.empty(len(step_score))
    for i, t in enumerate(trajectory_uids):
        mean, std = stats[traj_group[t]]
        adv[i] = (traj_total[t] - mean) / (std + epsilon) if norm_by_std else traj_total[t] - mean
    return adv[:, None] * response_mask


def check(
    advantages: np.ndarray,
    token_level_rewards: np.ndarray,
    response_mask: np.ndarray,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    *,
    norm_by_std: bool = True,
    atol: float = 1e-4,
) -> list[str]:
    """返回发现的问题（空列表 = 通过）。"""
    problems: list[str] = []
    mask = response_mask.astype(bool)
    if np.any(advantages[~mask] != 0):
        problems.append("mask 之外的 token 优势不为 0")
    row_adv = np.array([a[m][0] if m.any() else 0.0 for a, m in zip(advantages, mask)])
    for i, (a, m) in enumerate(zip(advantages, mask)):
        if m.any() and not np.allclose(a[m], row_adv[i], atol=atol):
            problems.append(f"第 {i} 行内各 token 优势不同")
    by_traj: dict = defaultdict(set)
    for i, t in enumerate(trajectory_uids):
        if mask[i].any():
            by_traj[t].add(round(float(row_adv[i]), 4))
    bad = [t for t, v in by_traj.items() if len(v) > 1]
    if bad:
        problems.append(f"{len(bad)} 条轨迹的各步优势不一致，例如 {bad[0]}")
    exp = expected_advantages(token_level_rewards, response_mask, index, trajectory_uids, norm_by_std=norm_by_std)
    if not np.allclose(advantages, exp, atol=atol):
        diff = np.abs(advantages - exp).max()
        problems.append(f"与独立重算的优势不一致（最大差 {diff:.3g}）")
    # 组内（按轨迹而不是按步）优势和约为 0
    group_sum: dict = defaultdict(float)
    group_n: dict = defaultdict(int)
    seen = set()
    for i, (g, t) in enumerate(zip(index, trajectory_uids)):
        if t not in seen and mask[i].any():
            seen.add(t)
            group_sum[g] += float(row_adv[i])
            group_n[g] += 1
    off = [g for g in group_sum if group_n[g] > 1 and abs(group_sum[g]) > atol * 10 * group_n[g]]
    if off:
        problems.append(f"{len(off)} 个题目组的轨迹优势和不为 0")
    return problems
