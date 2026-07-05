import pytest

from brownian.execution import create_engine, HostEngine
from brownian.execution.device import DeviceEngine


def test_auto_selects_host_below_threshold(monkeypatch):
    monkeypatch.setattr("brownian.execution.get_auto_threshold", lambda: 1000)
    engine = create_engine(kind="auto", n_paths=500)
    assert isinstance(engine, HostEngine)

def test_auto_selects_device_at_threshold(monkeypatch):
    monkeypatch.setattr("brownian.execution.get_auto_threshold", lambda: 1000)
    engine = create_engine(kind="auto", n_paths=1000)
    assert isinstance(engine, DeviceEngine)

def test_auto_requires_n_paths():
    with pytest.raises(ValueError, match="requires n_paths"):
        create_engine(kind="auto")