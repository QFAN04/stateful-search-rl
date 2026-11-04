"""★ 优势广播的单测。

- 本地：用一个手算的小 batch 验证 expected_advantages，并确认 check() 能发现几类典型错误；
- GPU 环境（有 torch 和 Agent-R1）：对上游 compute_grpo_outcome_advantage 做对拍。
"""
import numpy as np
import pytest

from ssrl.checks.step_advantage import check, expected_advantages

L = 4


def _batch():
    """2 道题（uid A/B），A 有 3 条轨迹、B 有 2 条；轨迹步数 1–3 不等。奖励只在终局步的最后一个有效 token。"""
    rows = [  # (uid, traj, 终局奖励 or None, 有效 token 数)
        ("A", "a1", None, 4), ("A", "a1", 1.0, 2),
        ("A", "a2", None, 3), ("A", "a2", None, 4), ("A", "a2", 0.0, 1),
        ("A", "a3", 1.0, 3),
        ("B", "b1", 0.0, 4),
        ("B", "b2", None, 2), ("B", "b2", 1.0, 4),
    ]
    rewards = np.zeros((len(rows), L))
    mask = np.zeros((len(rows), L))
    for i, (_, _, r, n) in enumerate(rows):
        mask[i, :n] = 1
        if r is not None:
            rewards[i, n - 1] = r
    index = np.array([r[0] for r in rows], dtype=object)
    traj = np.array([r[1] for r in rows], dtype=object)
    return rewards, mask, index, traj


def test_expected_by_hand():
    rewards, mask, index, traj = _batch()
    adv = expected_advantages(rewards, mask, index, traj)
    # A 组轨迹分 [1, 0, 1]：均值 2/3，无偏标准差 sqrt(1/3)
    sa = np.sqrt(1 / 3)
    a_pos, a_neg = (1 - 2 / 3) / (sa + 1e-6), (0 - 2 / 3) / (sa + 1e-6)
    # B 组 [0, 1]：均值 0.5，标准差 sqrt(0.5)
    sb = np.sqrt(0.5)
    b_pos, b_neg = 0.5 / (sb + 1e-6), -0.5 / (sb + 1e-6)
    want_rows = [a_pos, a_pos, a_neg, a_neg, a_neg, a_pos, b_neg, b_pos, b_pos]
    np.testing.assert_allclose(adv, np.array(want_rows)[:, None] * mask, rtol=1e-6)
    assert check(adv, rewards, mask, index, traj) == []


def test_single_trajectory_group_uses_mean0_std1():
    rewards = np.array([[0.0, 1.0]])
    mask = np.ones((1, 2))
    adv = expected_advantages(rewards, mask, np.array(["A"]), np.array(["a1"]))
    np.testing.assert_allclose(adv, [[1 / (1 + 1e-6)] * 2])


def test_dr_grpo_no_std():
    rewards, mask, index, traj = _batch()
    adv = expected_advantages(rewards, mask, index, traj, norm_by_std=False)
    assert adv[0, 0] == pytest.approx(1 / 3)
    assert check(adv, rewards, mask, index, traj, norm_by_std=False) == []


def test_check_catches_per_step_advantage():
    """常见错误：每一步按自己的即时奖励单独算优势（非终局步得到负优势）。"""
    rewards, mask, index, traj = _batch()
    step_score = (rewards * mask).sum(-1)
    wrong = (step_score - step_score.mean())[:, None] * mask
    problems = check(wrong, rewards, mask, index, traj)
    assert any("各步优势不一致" in p for p in problems)


def test_check_catches_mask_leak_and_mismatch():
    rewards, mask, index, traj = _batch()
    adv = expected_advantages(rewards, mask, index, traj)
    leak = adv.copy()
    leak[1, 3] = 0.7  # 第 1 行只有前 2 个 token 有效
    assert any("mask 之外" in p for p in check(leak, rewards, mask, index, traj))
    scaled = adv * 2
    assert any("独立重算" in p for p in check(scaled, rewards, mask, index, traj))


def test_trajectory_spanning_uids_rejected():
    rewards, mask, index, traj = _batch()
    index = index.copy()
    index[1] = "B"
    with pytest.raises(AssertionError):
        expected_advantages(rewards, mask, index, traj)


@pytest.mark.gpu
def test_matches_agent_r1_upstream():
    torch = pytest.importorskip("torch")
    core = pytest.importorskip("agent_r1.trainer.ppo.core_algos")
    rewards, mask, index, traj = _batch()
    adv, _ = core.compute_grpo_outcome_advantage(
        token_level_rewards=torch.tensor(rewards, dtype=torch.float32),
        response_mask=torch.tensor(mask, dtype=torch.float32),
        index=index,
        trajectory_uids=traj,
    )
    assert check(adv.numpy(), rewards, mask, index, traj) == []
