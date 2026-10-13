"""Bound submitted-but-not-consumed tasks; collect every result exactly once."""
import time

import ray


@ray.remote(num_cpus=1)
def work(i):
    time.sleep(0.03 * (i % 3 + 1))
    return i * i


if __name__ == "__main__":
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        pending, results = [], []
        limit, peak = 3, 0
        for i in range(12):
            if len(pending) >= limit:
                ready, pending = ray.wait(pending, num_returns=1, timeout=30)
                assert ready, "No task completed within the example timeout"
                results.extend(ray.get(ready))
            pending.append(work.remote(i))
            peak = max(peak, len(pending))
        results.extend(ray.get(pending, timeout=30))
        assert peak <= limit
        assert sorted(results) == [i * i for i in range(12)]
        print(f"PASS backpressure: results={len(results)}, peak_pending={peak}, limit={limit}")
    finally:
        ray.shutdown()
