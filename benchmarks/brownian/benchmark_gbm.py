import time
import logging

from brownian.gbm import GeometricBrownianMotion, GBMParams
from logs.logging_config import setup_logging

logger = logging.getLogger(__name__)


def benchmark(n_paths: int, n_steps: int, engine: str, n_runs: int = 5, warmup: int = 1) -> None:
    params = GBMParams(x0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=n_steps)
    gbm = GeometricBrownianMotion(params, engine=engine, seed=42)

    for _ in range(warmup):
        gbm.simulate_paths(n_paths)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        gbm.simulate_paths(n_paths)
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    print(
        f"engine={engine:>6} | n_paths={n_paths:>8,} | n_steps={n_steps:>5} | "
        f"avg={avg:.4f}s | min={min(times):.4f}s | max={max(times):.4f}s"
    )


def main() -> None:
    setup_logging(level=logging.WARNING)

    configs = [
        (1_000, 252),
        (10_000, 252),
        (100_000, 252),
        (10_000, 1_000),
        (100_000, 1_000),
    ]

    print("GBM Benchmark: host vs device")
    print("-" * 80)
    for n_paths in [1, 10, 100, 1_000]:
        benchmark(n_paths, 252, engine="host")
        benchmark(n_paths, 252, engine="device")

    for n_paths, n_steps in configs:
        benchmark(n_paths, n_steps, engine="host")
        benchmark(n_paths, n_steps, engine="device")
        time.sleep(2)

if __name__ == "__main__":
    main()