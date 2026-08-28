import pytest

from gpu_risk_simulator.brownian.execution import create_engine, HostEngine
from gpu_risk_simulator.brownian.execution.device import DeviceEngine

TARGET = "gpu_risk_simulator.brownian.execution.get_auto_threshold"


def test_auto_selects_host_below_threshold(monkeypatch):
    monkeypatch.setattr(TARGET, lambda: 1000)
    engine = create_engine(kind="auto", n_paths=500)
    assert isinstance(engine, HostEngine)


def test_auto_selects_device_at_threshold(monkeypatch):
    monkeypatch.setattr(TARGET, lambda: 1000)

    try:
        engine = create_engine(kind="auto", n_paths=1000)
    except RuntimeError as e:
        if "cupy is not installed or no CUDA device is available" in str(e):
            pytest.skip("No working CUDA device")
        raise

    assert isinstance(engine, DeviceEngine)