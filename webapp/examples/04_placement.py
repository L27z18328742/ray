"""Exercise real Ray placement groups with CPU resources, not fake GPUs."""
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1)
class Seat:
    def identify(self):
        return ray.get_runtime_context().get_node_id()


if __name__ == "__main__":
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    groups, actors = [], []
    try:
        pg = placement_group([{"CPU": 1}, {"CPU": 1}], strategy="PACK")
        groups.append(pg)
        ray.get(pg.ready(), timeout=30)
        for i in range(2):
            actors.append(Seat.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i
            )).remote())
        node_ids = ray.get([a.identify.remote() for a in actors], timeout=30)
        assert len(set(node_ids)) == 1  # This example deliberately starts one local node.
        impossible = placement_group([{"CPU": 3}], strategy="PACK")
        groups.append(impossible)
        ready, pending = ray.wait([impossible.ready()], timeout=1)
        assert not ready and len(pending) == 1
        print("PASS placement: 2 CPU bundles ready; 3 CPU bundle pending (expected)")
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        for pg in groups:
            remove_placement_group(pg)
        ray.shutdown()
