import importlib

import pytest

import src.assembleur_engine_runtime as rt


class FakePerfCounter:
    def __init__(self, times):
        self.times = list(times)
        self.i = 0

    def __call__(self):
        if not self.times:
            return 0.0
        if self.i >= len(self.times):
            return float(self.times[-1])
        v = float(self.times[self.i])
        self.i += 1
        return v


@pytest.fixture
def fakePerfCounter():
    def _make(times):
        return FakePerfCounter(times)

    return _make


@pytest.fixture
def patchPerfCounter(monkeypatch, fakePerfCounter):
    def _patch(times):
        fake = fakePerfCounter(times)
        monkeypatch.setattr(rt.time, "perf_counter", fake)
        return fake

    return _patch


def test_enginecontrol_initial_state():
    ctrl = rt.EngineControl()
    assert ctrl.isStopRequested() is False
    assert ctrl.isPauseRequested() is False


def test_enginecontrol_pause_resume():
    ctrl = rt.EngineControl()
    ctrl.requestPause()
    assert ctrl.isPauseRequested() is True
    ctrl.resume()
    assert ctrl.isPauseRequested() is False


def test_enginecontrol_stop_persists():
    ctrl = rt.EngineControl()
    ctrl.requestStop()
    assert ctrl.isStopRequested() is True
    ctrl.resume()
    assert ctrl.isStopRequested() is True


def test_eventqueue_fifo_and_empty():
    q = rt.EventQueue()
    assert q.empty() is True
    assert q.getNowait() is None

    q.put("STATUS", "RUNNING")
    q.put("PROGRESS", 0.5)

    e1 = q.getNowait()
    assert e1 is not None and e1.type == "STATUS" and e1.payload == "RUNNING"

    e2 = q.getNowait()
    assert e2 is not None and e2.type == "PROGRESS" and e2.payload == 0.5

    assert q.getNowait() is None
    assert q.empty() is True


def test_checkpointpolicy_threshold_and_reset(monkeypatch):
    fake = FakePerfCounter([0.0, 0.1])  # init, then checkpoint
    monkeypatch.setattr(rt.time, "perf_counter", fake)

    cfg = rt.RunControlConfig(
        maxSolutions=50,
        minBatchCells=10,
        maxBatchCells=100,
        targetBatchSec=0.05,
        progressMinIntervalSec=0.2,
    )
    policy = rt.CheckpointPolicy(cfg, enabled=True)

    for _ in range(9):
        policy.onCellTested()
    assert policy.shouldCheckpoint() is False

    policy.onCellTested()
    assert policy.shouldCheckpoint() is True

    info = policy.onCheckpoint()
    assert policy.shouldCheckpoint() is False
    assert set(info.keys()) == {"emitProgress", "batchCellsTarget", "lastBatchElapsed"}
    assert info["lastBatchElapsed"] == 0.1


def test_checkpointpolicy_adapt_up(monkeypatch):
    fake = FakePerfCounter([0.0, 0.01])  # init, checkpoint (fast)
    monkeypatch.setattr(rt.time, "perf_counter", fake)

    cfg = rt.RunControlConfig(
        minBatchCells=10,
        maxBatchCells=80,
        targetBatchSec=0.05,
        progressMinIntervalSec=0.0,
    )
    policy = rt.CheckpointPolicy(cfg, enabled=True)
    policy.onCellTested(10)
    info = policy.onCheckpoint()
    assert info["batchCellsTarget"] == 20


def test_checkpointpolicy_adapt_bounded_max(monkeypatch):
    fake = FakePerfCounter([0.0, 0.01, 0.02])  # init, chk1, chk2 (fast)
    monkeypatch.setattr(rt.time, "perf_counter", fake)

    cfg = rt.RunControlConfig(
        minBatchCells=10,
        maxBatchCells=20,
        targetBatchSec=0.05,
        progressMinIntervalSec=0.0,
    )
    policy = rt.CheckpointPolicy(cfg, enabled=True)

    policy.onCellTested(10)
    info1 = policy.onCheckpoint()
    assert info1["batchCellsTarget"] == 20

    policy.onCellTested(20)
    info2 = policy.onCheckpoint()
    assert info2["batchCellsTarget"] == 20


def test_checkpointpolicy_progress_cadence(monkeypatch):
    fake = FakePerfCounter([0.0, 0.01, 0.10, 0.25])  # init, chk1, chk2, chk3
    monkeypatch.setattr(rt.time, "perf_counter", fake)

    cfg = rt.RunControlConfig(
        minBatchCells=1,
        maxBatchCells=100,
        targetBatchSec=0.05,
        progressMinIntervalSec=0.2,
    )
    policy = rt.CheckpointPolicy(cfg, enabled=True)

    policy.onCellTested(1)
    info1 = policy.onCheckpoint()
    assert info1["emitProgress"] is True

    policy.onCellTested(1)
    info2 = policy.onCheckpoint()
    assert info2["emitProgress"] is False

    policy.onCellTested(1)
    info3 = policy.onCheckpoint()
    assert info3["emitProgress"] is True
