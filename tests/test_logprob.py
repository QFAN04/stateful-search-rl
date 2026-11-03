"""logprob 一致性统计只看有效 token：padding、mask 外的巨大差异不能影响结果。"""
import numpy as np
import pytest

from ssrl.checks.logprob_consistency import compare


def test_identical_is_zero():
    lp = np.log(np.array([[0.5, 0.25, 0.1]]))
    res = compare(lp, lp, np.ones_like(lp))
    assert res["abs_logp_diff_mean"] == 0 and res["abs_logp_diff_max"] == 0 and res["n_tokens"] == 3


def test_masked_tokens_ignored():
    rollout = np.array([[-1.0, -2.0, 0.0], [-1.0, 0.0, 0.0]])
    ref = np.array([[-1.1, -2.0, -99.0], [-0.8, -50.0, -50.0]])  # mask 外的位置差异巨大
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    res = compare(rollout, ref, mask)
    assert res["n_tokens"] == 3
    assert res["abs_logp_diff_max"] == pytest.approx(0.2)
    assert res["abs_logp_diff_mean"] == pytest.approx((0.1 + 0.0 + 0.2) / 3)
    assert res["signed_logp_diff_mean"] == pytest.approx((0.1 + 0.0 - 0.2) / 3)
    # 按序列平均：序列 1 = 0.05，序列 2 = 0.2
    assert res["per_seq_abs_logp_diff_max"] == pytest.approx(0.2)


def test_shape_and_empty_mask_errors():
    with pytest.raises(ValueError):
        compare(np.zeros((1, 2)), np.zeros((1, 3)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        compare(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
