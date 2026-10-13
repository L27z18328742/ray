"""A task result is an ObjectRef until the driver gets it."""
import os

import ray


@ray.remote(num_cpus=1)
def square(x):
    return {"value": x * x, "worker_pid": os.getpid()}


if __name__ == "__main__":
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        refs = [square.remote(x) for x in [2, 3]]
        assert all(isinstance(ref, ray.ObjectRef) for ref in refs)
        values = ray.get(refs, timeout=30)
        assert [v["value"] for v in values] == [4, 9]
        assert all(v["worker_pid"] != os.getpid() for v in values)
        print("driver:", os.getpid(), "type:", type(refs[0]).__name__)
        print("PASS tasks:", values)
    finally:
        ray.shutdown()
