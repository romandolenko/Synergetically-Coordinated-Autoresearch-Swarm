"""Adversarial-commit tests for the safety-envelope validator (AC-B5').

20 cases total: 4 per rule in design §8. Each asserts the validator rejects with
a non-empty reason.
"""

from __future__ import annotations

import pytest

from scas.coordinator import (
    BASELINE_PREPARE_HASH,
    BASELINE_PYPROJECT_HASH,
    validate,
)


def good_commit() -> dict:
    return {
        "agent_id": "a0",
        "step": 1,
        "files_changed": ["train.py"],
        "pyproject_hash": BASELINE_PYPROJECT_HASH,
        "prepare_hash": BASELINE_PREPARE_HASH,
        "val_bpb": 1.0,
    }


@pytest.mark.parametrize(
    "files",
    [
        ["train.py", "prepare.py"],
        ["prepare.py"],
        ["pyproject.toml"],
        ["train.py", "scas/coordinator.py"],
    ],
)
def test_rule1_files_changed_must_be_only_train_py(files):
    c = good_commit()
    c["files_changed"] = files
    ok, reason = validate(c)
    assert not ok and reason


@pytest.mark.parametrize(
    "h",
    ["", "deadbeef", None, "a" * 64],
)
def test_rule2_pyproject_hash_must_match_baseline(h):
    c = good_commit()
    c["pyproject_hash"] = h
    ok, reason = validate(c)
    assert not ok and reason


@pytest.mark.parametrize(
    "h",
    ["", "0" * 64, None, "feedface" * 8],
)
def test_rule3_prepare_hash_must_match_baseline(h):
    c = good_commit()
    c["prepare_hash"] = h
    ok, reason = validate(c)
    assert not ok and reason


@pytest.mark.parametrize(
    "v",
    [0.0, -1.0, 0.09, None],
)
def test_rule4_val_bpb_below_floor_rejected(v):
    c = good_commit()
    c["val_bpb"] = v
    ok, reason = validate(c)
    assert not ok and reason


@pytest.mark.parametrize(
    "n",
    [3, 4, 5, 10],
)
def test_rule5_crash_budget_exceeded(n):
    ok, reason = validate(good_commit(), recent_failures=n)
    assert not ok and reason


@pytest.mark.parametrize(
    "v,ref,ok_expected",
    [
        (0.464, 0.577, True),    # the real vast.ai sweep: honest 20% gain accepted
        (0.28, 0.577, False),    # > 2x better than anything accepted: implausible
        (0.24, 0.464, True),
        (0.23, 0.464, False),
        (0.15, None, True),      # no reference yet: only the absolute backstop
        (0.15, float("inf"), True),
    ],
)
def test_rule4_floor_is_relative_to_best_accepted(v, ref, ok_expected):
    c = good_commit()
    c["val_bpb"] = v
    ok, reason = validate(c, reference_best=ref)
    assert ok is ok_expected
    assert ok or reason


def test_file_hash_ignores_line_endings(tmp_path):
    from scas.coordinator import _file_sha256

    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert _file_sha256(lf) == _file_sha256(crlf)
