"""A CPU orchestration model; no LLM, gradients, optimizer, NCCL or SGLang."""
from dataclasses import dataclass

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@dataclass
class Box:
    inner: object


@ray.remote(num_cpus=1)
class Manager:
    def __init__(self):
        self.version = -1

    def set_version(self, version):
        self.version = version
        return self.version

    def generate(self, rollout_id):
        # Fixed strided shards are a teaching simplification, not slime's DP scheduler.
        return [Box(ray.put({"rank": r, "items": list(range(r, 4, 2)),
                             "rollout": rollout_id, "version": self.version})) for r in range(2)]


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, rank):
        self.rank = rank
        self.version = 0

    def train(self, rollout_id, boxes):
        assert isinstance(boxes[self.rank].inner, ray.ObjectRef)
        shard = ray.get(boxes[self.rank].inner)
        assert shard["rank"] == self.rank
        assert shard["items"] == list(range(self.rank, 4, 2))
        assert shard["rollout"] == rollout_id
        assert shard["version"] == self.version, "Training on an unexpected serving version"
        self.version += 1  # Not a numerical optimizer step.
        return self.version


class TrainGroup:
    """A driver-local object containing remote actor handles."""
    def __init__(self, pg):
        self.actors = [Trainer.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=r
        )).remote(r) for r in range(2)]

    def async_train(self, rollout_id, boxes):
        return [a.train.remote(rollout_id, boxes) for a in self.actors]


if __name__ == "__main__":
    ray.init(address="local", num_cpus=4, include_dashboard=False)
    pg, manager, group = None, None, None
    try:
        pg = placement_group([{"CPU": 1}, {"CPU": 1}], strategy="PACK")
        ray.get(pg.ready(), timeout=30)
        manager = Manager.remote()  # Has a CPU outside the reserved training bundles.
        group = TrainGroup(pg)
        ray.get(manager.set_version.remote(0), timeout=30)
        for step in range(3):
            boxes = ray.get(manager.generate.remote(step), timeout=30)
            assert len(boxes) == 2 and all(isinstance(b.inner, ray.ObjectRef) for b in boxes)
            versions = ray.get(group.async_train(step, boxes), timeout=30)
            assert versions == [step + 1, step + 1]
            synced = ray.get(manager.set_version.remote(versions[0]), timeout=30)
            print(f"round {step}: serving={step} -> trained={versions} -> synced={synced}")
        print("PASS mini slime: 3 rounds, distinct DP shards, consistent versions")
    finally:
        if group:
            for actor in group.actors:
                ray.kill(actor, no_restart=True)
        if manager:
            ray.kill(manager, no_restart=True)
        if pg:
            remove_placement_group(pg)
        ray.shutdown()
