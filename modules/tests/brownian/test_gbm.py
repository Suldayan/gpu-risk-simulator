import pytest
import numpy as np

from unittest.mock import patch
from brownian.gbm import GBMParams, GeometricBrownianMotion
from brownian.errors import GBMParameterError, GBMNumericalError


@pytest.fixture
def default_params():
    return GBMParams(x0=1.0, mu=0.1, sigma=0.2, T=1.0, n_steps=100)


@pytest.fixture
def default_gbm(default_params):
    return GeometricBrownianMotion(default_params)


def test_gbm_sets_dt(default_gbm):
    assert default_gbm.dt == pytest.approx(1.0 / 100)


def test_gbm_times_shape(default_gbm):
    assert default_gbm.times.shape == (101,)


def test_gbm_times_bounds(default_gbm):
    assert default_gbm.times[0] == 0.0
    assert default_gbm.times[-1] == pytest.approx(1.0)


def test_invalid_x0_zero():
    with pytest.raises(GBMParameterError, match="x0"):
        GBMParams(x0=0.0, mu=0.1, sigma=0.2, T=1.0, n_steps=100)


def test_invalid_x0_negative():
    with pytest.raises(GBMParameterError, match="x0"):
        GBMParams(x0=-1.0, mu=0.1, sigma=0.2, T=1.0, n_steps=100)


def test_invalid_sigma_zero():
    with pytest.raises(GBMParameterError, match="sigma"):
        GBMParams(x0=1.0, mu=0.1, sigma=0.0, T=1.0, n_steps=100)


def test_invalid_sigma_negative():
    with pytest.raises(GBMParameterError, match="sigma"):
        GBMParams(x0=1.0, mu=0.1, sigma=-0.2, T=1.0, n_steps=100)


def test_invalid_T_zero():
    with pytest.raises(GBMParameterError, match="T"):
        GBMParams(x0=1.0, mu=0.1, sigma=0.2, T=0.0, n_steps=100)


def test_invalid_n_steps_zero():
    with pytest.raises(GBMParameterError, match="n_steps"):
        GBMParams(x0=1.0, mu=0.1, sigma=0.2, T=1.0, n_steps=0)


def test_negative_mu_is_valid():
    params = GBMParams(x0=1.0, mu=-1.0, sigma=0.2, T=1.0, n_steps=100)
    assert params.mu == -1.0


def simulate_paths(self, n_paths: int) -> np.ndarray:
    if not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
    return np.array([self.simulate_path() for _ in range(n_paths)])


def test_simulate_path_shape(default_gbm):
    path = default_gbm.simulate_path()
    assert path.shape == (101,)


def test_simulate_path_starts_at_x0(default_gbm):
    path = default_gbm.simulate_path()
    assert path[0] == pytest.approx(default_gbm.params.x0)


def test_simulate_path_all_positive(default_gbm):
    path = default_gbm.simulate_path()
    assert np.all(path > 0)


def test_simulate_path_all_finite(default_gbm):
    path = default_gbm.simulate_path()
    assert np.all(np.isfinite(path))


def test_simulate_paths_shape(default_gbm):
    paths = default_gbm.simulate_paths(n_paths=50)
    assert paths.shape == (50, 101)


def test_simulate_paths_invalid_n_paths_zero(default_gbm):
    with pytest.raises(ValueError, match="n_paths"):
        default_gbm.simulate_paths(n_paths=0)


def test_simulate_paths_invalid_n_paths_negative(default_gbm):
    with pytest.raises(ValueError, match="n_paths"):
        default_gbm.simulate_paths(n_paths=-1)


def test_simulate_path_raises_on_non_finite(default_gbm):
    with patch("brownian.gbm.np.exp", return_value=np.array([np.inf] * 101)):
        with pytest.raises(GBMNumericalError):
            default_gbm.simulate_path()