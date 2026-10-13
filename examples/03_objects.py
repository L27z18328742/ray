"""Top-level refs resolve; refs inside a Box remain refs."""
from dataclasses import dataclass

import ray


@dataclass
class Box:
    inner: object


@ray.remote(num_cpus=1)
def plus_one(value):
    assert not isinstance(value, ray.ObjectRef)
    return value + 1


@ray.remote(num_cpus=1)
def read_shard(boxes, rank):
    assert isinstance(boxes[rank].inner, ray.ObjectRef)
    return ray.get(boxes[rank].inner)


if __name__ == "__main__":
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        value_ref = ray.put(20)
        assert ray.get(plus_one.remote(value_ref), timeout=30) == 21
        boxes = [Box(ray.put([0, 2])), Box(ray.put([1, 3]))]
        output = ray.get([read_shard.remote(boxes, r) for r in range(2)], timeout=30)
        assert output == [[0, 2], [1, 3]]
        print("PASS objects: top-level=21; boxed shards=", output)
    finally:
        ray.shutdown()
