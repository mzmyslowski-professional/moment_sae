# tests/test_train.py
import pytest
from train_sae import get_lr_scale, get_l1_coeff
from config import CFG


def test_lr_scale_at_step_0():
    assert get_lr_scale(0, warmup_steps=1000) == pytest.approx(0.0)


def test_lr_scale_at_warmup_end():
    assert get_lr_scale(1000, warmup_steps=1000) == pytest.approx(1.0)


def test_lr_scale_flat_after_warmup():
    assert get_lr_scale(5000, warmup_steps=1000) == pytest.approx(1.0)
    assert get_lr_scale(50000, warmup_steps=1000) == pytest.approx(1.0)


def test_lr_scale_linear_during_warmup():
    assert get_lr_scale(500, warmup_steps=1000) == pytest.approx(0.5)


def test_l1_coeff_at_step_0():
    assert get_l1_coeff(0) == pytest.approx(CFG.l1_start)


def test_l1_coeff_at_warmup_end():
    assert get_l1_coeff(CFG.l1_warmup_steps) == pytest.approx(CFG.l1_end)


def test_l1_coeff_flat_after_warmup():
    assert get_l1_coeff(CFG.l1_warmup_steps + 1000) == pytest.approx(CFG.l1_end)


def test_l1_coeff_monotone_during_warmup():
    values = [get_l1_coeff(s) for s in range(0, CFG.l1_warmup_steps + 1, 100)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
