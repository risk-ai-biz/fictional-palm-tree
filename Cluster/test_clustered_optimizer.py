
import numpy as np
import clustered_optimizer as co

class SGD:
    def __init__(self, lr: float = 0.1) -> None:
        self.lr = float(lr)
    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        params -= self.lr * grads

def test_identity_mode_maps_and_offsets():
    schema = co.ModelSchema(raw_sizes={"A": 3, "B": 2}, pairs=[("A","B")])
    cfg = co.ClusteringConfig(cluster_enabled=False)
    # dummy stats (unused because cluster disabled)
    stats = {
        ("A",): (np.zeros(3), np.ones(3)),
        ("B",): (np.zeros(2), np.ones(2)),
        ("A","B"): (np.zeros(6), np.ones(6)),
    }
    maps = co.refresh_maps(schema, stats, cfg)
    assert np.array_equal(maps[("A",)], np.arange(3))
    assert np.array_equal(maps[("B",)], np.arange(2))
    assert np.array_equal(maps[("A","B")], np.arange(6))
    offsets = co.compute_offsets(schema, maps)
    assert offsets[("A",)] == 0
    assert offsets[("B",)] == 3
    assert offsets[("A","B")] == 5

def test_pack_and_remap_consistency_identity():
    schema = co.ModelSchema(raw_sizes={"A": 3}, pairs=[])
    cfg = co.ClusteringConfig(cluster_enabled=False)
    stats = {("A",): (np.zeros(3), np.ones(3))}
    maps = co.refresh_maps(schema, stats, cfg)
    # raw params and grads
    raw_params = {("A",): np.array([1.0, 2.0, 3.0])}
    theta = co.pack_raw_params_to_clusters(schema, maps, raw_params)
    # identity means theta equals raw params
    assert np.allclose(theta, raw_params[("A",)])
    raw_grads = {("A",): np.ones(3)}
    gflat = co.remap_raw_grads_to_clusters(schema, maps, raw_grads)
    assert np.allclose(gflat, np.ones(3))

def test_repack_state_on_refresh_single_feature():
    schema = co.ModelSchema(raw_sizes={"A": 3})
    # Old map: [0,0,1]; New map: [0,1,1]
    old_maps = {("A",): np.array([0,0,1], dtype=np.int32)}
    new_maps = {("A",): np.array([0,1,1], dtype=np.int32)}
    # Compose old theta consistent with old map from raw params [1,3,5] → cluster means [2,5]
    raw_params = {("A",): np.array([1.0, 3.0, 5.0])}
    old_theta = co.pack_raw_params_to_clusters(schema, old_maps, raw_params)
    # Repack
    new_theta, new_m, new_v = co.repack_state_on_refresh(schema, old_maps, new_maps, old_theta)
    # Expected: new cluster0 = raw0 -> 2.0; new cluster1 = avg(raw1=2.0, raw2=5.0) = 3.5
    assert np.allclose(new_theta, np.array([2.0, 3.5]))

def test_pairwise_cluster_shape_and_labels():
    # A clusterable, B not clusterable
    schema = co.ModelSchema(raw_sizes={"A": 3, "B": 2}, pairs=[("A","B")])
    cfg = co.ClusteringConfig(clusterable={"A": True, "B": False}, min_gap_single=0.0, cluster_enabled=True)
    # Stats: make single A produce 1 cluster (min_gap_single=0.0)
    stats = {
        ("A",): (np.array([0.1, 0.1, 0.1]), np.array([0.01, 0.01, 0.01])),
        ("B",): (np.zeros(2), np.ones(2)),
        ("A","B"): (np.zeros(6), np.ones(6)),
    }
    maps = co.refresh_maps(schema, stats, cfg)
    cmap_a = maps[("A",)]
    assert cmap_a.max() == 0  # single cluster
    labels_pair = maps[("A","B")]
    # Expect label = cluster_a * n_b + raw_b = 0 * 2 + [0,1] repeating
    expected = np.array([0,1, 0,1, 0,1], dtype=np.int32)
    assert np.array_equal(labels_pair, expected)
    # K for pair should be n_b (since K_a=1)
    assert int(labels_pair.max()) + 1 == 2

def test_optimizer_refresh_toggle_maps():
    schema = co.ModelSchema(raw_sizes={"A": 3}, pairs=[])
    cfg = co.ClusteringConfig(cluster_enabled=True, min_gap_single=0.0)
    sgd = SGD(lr=0.1)
    # Two maps to toggle between
    maps0 = {("A",): np.array([0,0,0], dtype=np.int32)}
    maps1 = {("A",): np.array([0,1,1], dtype=np.int32)}
    # Toggle refresh_fn
    toggle = {"t": 0}
    def refresh_fn(schema_, stats_, cfg_):
        toggle["t"] += 1
        return maps1 if (toggle["t"] % 2 == 1) else maps0
    # Initial stats (unused)
    stats = {("A",): (np.zeros(3), np.ones(3))}
    opt = co.ClusteredOptimizer(sgd, schema, cfg, initial_maps=maps0, refresh_every_steps=1, refresh_fn=refresh_fn)
    raw_params = {("A",): np.zeros(3)}
    opt.scatter_from_raw_params(raw_params)
    raw_grads = {("A",): np.array([1.0, 0.0, -1.0])}
    # Step 1 → maps become maps1
    opt.step(raw_grads, stats)
    assert np.array_equal(opt.maps[("A",)], maps1[("A",)])
    # Step 2 → maps become maps0
    opt.step(raw_grads, stats)
    assert np.array_equal(opt.maps[("A",)], maps0[("A",)])

def test_null_pool_single_feature_applied():
    schema = co.ModelSchema(raw_sizes={"A": 4})
    cfg = co.ClusteringConfig(cluster_enabled=True, min_gap_single=0.8, null_mu=0.05, null_var=1e-6)
    # Two values near zero → null; two farther away
    means = np.array([0.0, 0.01, 0.2, -0.3])
    vars_ = np.array([1e-8, 1e-8, 0.02, 0.02])
    stats = {("A",): (means, vars_)}
    maps = co.refresh_maps(schema, stats, cfg)
    labels = maps[("A",)]
    assert labels[0] == 0 and labels[1] == 0  # null pool
    # Non-null labels should be > 0 (shifted)
    assert labels[2] > 0 and labels[3] > 0
