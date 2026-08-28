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
    engine = create_engine(kind="auto", n_paths=1000)
    assert isinstance(engine, DeviceEngine)

def test_auto_requires_n_paths():
    with pytest.raises(ValueError, match="requires n_paths"):
        create_engine(kind="auto")