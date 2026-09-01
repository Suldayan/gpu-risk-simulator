import numpy as np
import pytest

from risk_simulator.brownian.errors import GBMParameterError
from risk_simulator.brownian.models import GBMParams
from risk_simulator.brownian.gbm import GeometricBrownianMotion


@pytest.fixture
def default_params() -> GBMParams:
    return GBMParams(
        tickers=("TEST",),
        x0=(1.0,),
        mu=(0.1,),
        sigma=(0.2,),
        correlation=np.array([[1.0]]),
        T=1.0,
        n_steps=100,
    )


@pytest.fixture
def default_gbm(default_params: GBMParams) -> GeometricBrownianMotion:
    return GeometricBrownianMotion(default_params)


def test_gbm_sets_dt(default_gbm: GeometricBrownianMotion):
    assert default_gbm.dt == pytest.approx(1.0 / 100)


def test_gbm_times_shape(default_gbm: GeometricBrownianMotion):
    assert default_gbm.times.shape == (101,)


def test_gbm_times_bounds(default_gbm: GeometricBrownianMotion):
    assert default_gbm.times[0] == 0.0
    assert default_gbm.times[-1] == pytest.approx(1.0)


def test_invalid_x0_zero():
    with pytest.raises(GBMParameterError, match="x0"):
        GBMParams(
            tickers=("TEST",), x0=(0.0,), mu=(0.1,), sigma=(0.2,),
            correlation=np.array([[1.0]]), T=1.0, n_steps=100,
        )


def test_invalid_x0_negative():
    with pytest.raises(GBMParameterError, match="x0"):
        GBMParams(
            tickers=("TEST",), x0=(-1.0,), mu=(0.1,), sigma=(0.2,),
            correlation=np.array([[1.0]]), T=1.0, n_steps=100,
        )


def test_invalid_sigma_zero():
    with pytest.raises(GBMParameterError, match="sigma"):
        GBMParams(
            tickers=("TEST",), x0=(1.0,), mu=(0.1,), sigma=(0.0,),
            correlation=np.array([[1.0]]), T=1.0, n_steps=100,
        )


def test_invalid_sigma_negative():
    with pytest.raises(GBMParameterError, match="sigma"):
        GBMParams(
            tickers=("TEST",), x0=(1.0,), mu=(0.1,), sigma=(-0.2,),
            correlation=np.array([[1.0]]), T=1.0, n_steps=100,
        )


def test_invalid_T_zero():
    with pytest.raises(GBMParameterError, match="T"):
        GBMParams(
            tickers=("TEST",), x0=(1.0,), mu=(0.1,), sigma=(0.2,),
            correlation=np.array([[1.0]]), T=0.0, n_steps=100,
        )


def test_invalid_n_steps_zero():
    with pytest.raises(GBMParameterError, match="n_steps"):
        GBMParams(
            tickers=("TEST",), x0=(1.0,), mu=(0.1,), sigma=(0.2,),
            correlation=np.array([[1.0]]), T=1.0, n_steps=0,
        )


def test_negative_mu_is_valid():
    params = GBMParams(
        tickers=("TEST",), x0=(1.0,), mu=(-1.0,), sigma=(0.2,),
        correlation=np.array([[1.0]]), T=1.0, n_steps=100,
    )
    assert params.mu == (-1.0,)


def test_simulate_paths_shape(default_gbm: GeometricBrownianMotion):
    paths = default_gbm.simulate_paths(n_paths=50)
    # Unified return: (n_assets, n_paths, n_steps + 1)
    assert paths.shape == (1, 50, 101)
    # If you want the old 2D slice:
    assert paths[0].shape == (50, 101)


def test_simulate_paths_invalid_n_paths_zero(default_gbm: GeometricBrownianMotion):
    with pytest.raises(ValueError, match="n_paths"):
        default_gbm.simulate_paths(n_paths=0)


def test_simulate_paths_invalid_n_paths_negative(default_gbm: GeometricBrownianMotion):
    with pytest.raises(ValueError, match="n_paths"):
        default_gbm.simulate_paths(n_paths=-1)