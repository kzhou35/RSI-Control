"""open_run must not mistake a rate-limited existence probe for a missing per-run volume."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "modal"))
import rsi_modal as R  # noqa: E402
from modal.exception import NotFoundError, ResourceExhaustedError  # noqa: E402


class _Vol:
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def listdir(self, _path):
        self.calls += 1
        if self.calls <= self.fails:
            raise ResourceExhaustedError("VolumeListFiles rate limit exceeded. Please wait and retry.")
        return iter(())


class _Missing:
    def listdir(self, _path):
        raise NotFoundError("Volume 'x' not found in environment 'main'")


def test_open_run_retries_the_rate_limit_and_keeps_the_per_run_volume(monkeypatch):
    v = _Vol(fails=2)
    monkeypatch.setattr(R.modal.Volume, "from_name", staticmethod(lambda name: v))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    vol, pre = R.open_run("attack_goal_only_row05_trait_backdoor_20260913-080413-1c2a61")
    assert vol is v and pre == "" and v.calls == 3


def test_open_run_falls_back_to_the_shared_volume_only_when_the_volume_is_missing(monkeypatch):
    monkeypatch.setattr(R.modal.Volume, "from_name", staticmethod(lambda name: _Missing()))
    vol, pre = R.open_run("honest_na_row01_old_20260801-000000-abcdef")
    assert vol is R.runs and pre == "honest_na_row01_old_20260801-000000-abcdef/"


def test_open_run_propagates_other_errors(monkeypatch):
    class _Broken:
        def listdir(self, _path):
            raise PermissionError("nope")
    monkeypatch.setattr(R.modal.Volume, "from_name", staticmethod(lambda name: _Broken()))
    with pytest.raises(PermissionError):
        R.open_run("honest_na_row01_x_20260913-000000-abcdef")


def test_retry_rate_limited_backs_off_then_gives_up():
    slept, n = [], [0]
    def always():
        n[0] += 1
        raise ResourceExhaustedError("rate limit")
    with pytest.raises(ResourceExhaustedError):
        R.retry_rate_limited(always, _sleep=slept.append)
    assert slept == list(R.RATE_LIMIT_BACKOFF_S) and n[0] == len(R.RATE_LIMIT_BACKOFF_S) + 1
