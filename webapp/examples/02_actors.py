"""A persistent counter, plus an independent actor instance."""
import os

import ray


@ray.remote(num_cpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n
        return {"value": self.value, "pid": os.getpid()}


if __name__ == "__main__":
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    actors = []
    try:
        a, b = Counter.remote(), Counter.remote()
        actors = [a, b]
        refs = [a.add.remote(2), a.add.remote(3), b.add.remote(1)]
        values = ray.get(refs, timeout=30)
        assert [v["value"] for v in values] == [2, 5, 1]
        assert values[0]["pid"] == values[1]["pid"]
        assert values[0]["pid"] != values[2]["pid"]
        print("PASS actors:", values)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()
