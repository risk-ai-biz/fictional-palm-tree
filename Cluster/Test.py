# test_clustered_optimizer.py

import numpy as np
import clustered_optimizer as co

# --- Minimal base optimizers -------------------------------------------

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    def step(self, params: np.ndarray, grads: np.ndarray):
        params -= self.lr * grads

class Adam:
    def __init__(self, lr=0.05):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
    def step(self, params: np.ndarray, grads: np.ndarray):
        if not hasattr(self, 'm'):
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --- Synthetic feature setup -----------------------------------------

raw_sizes = {'a': 4, 'b': 3}
clusterable = {'a': True, 'b': False}
interaction_pairs = [('a', 'b')]

np.random.seed(0)
value_stats = {
    ('a',): (np.random.randn(4), np.full(4, 0.1)),
    ('b',): (np.random.randn(3), np.full(3, 0.1)),
    ('a','b'): (np.random.randn(4*3), np.full(4*3, 0.1)),
}

# --- Build initial cluster maps --------------------------------------

cluster_maps = co.refresh_maps(
    value_stats, raw_sizes, clusterable, interaction_pairs,
    min_gap_single=0.5,
    split_thresh_single=1.0,
    min_gap_pair=0.5,
    split_thresh_pair=1.0
)

# Ensure correct shapes
assert cluster_maps[('a',)].shape == (4,)
assert cluster_maps[('b',)].shape == (3,)
assert cluster_maps[('a','b')].shape == (12,)

# --- Test with SGD optimizer ----------------------------------------

sgd = SGD(lr=0.1)
opt_sgd = co.ClusteredOptimizer(
    base_optimizer=sgd,
    raw_sizes=raw_sizes,
    cluster_maps=cluster_maps,
    clusterable=clusterable,
    interaction_pairs=interaction_pairs,
    min_gap_single=0.5,
    split_thresh_single=1.0,
    min_gap_pair=0.5,
    split_thresh_pair=1.0,
    refresh_freq=100  # no refresh during simple test
)

# Scatter zero initial params
initial_params = {
    ('a',): np.zeros(raw_sizes['a']),
    ('b',): np.zeros(raw_sizes['b']),
    ('a','b'): np.zeros(raw_sizes['a']*raw_sizes['b'])
}
opt_sgd._scatter_params(initial_params)

# Create synthetic raw gradients
raw_grads = {
    ('a',): np.array([ 1.0, -2.0,  0.5,  0.0 ]),
    ('b',): np.array([ 0.1, -0.1,  0.0 ]),
    ('a','b'): np.random.randn(12),
}

# Perform two steps
opt_sgd.step(raw_grads, value_stats)
opt_sgd.step(raw_grads, value_stats)

# After stepping, theta_flat should be nonzero and match flat dimension
assert opt_sgd.theta_flat.size > 0

# --- Test with Adam optimizer ---------------------------------------

adam = Adam(lr=0.05)
opt_adam = co.ClusteredOptimizer(
    base_optimizer=adam,
    raw_sizes=raw_sizes,
    cluster_maps=cluster_maps,
    clusterable=clusterable,
    interaction_pairs=interaction_pairs,
    min_gap_single=0.5,
    split_thresh_single=1.0,
    min_gap_pair=0.5,
    split_thresh_pair=1.0,
    refresh_freq=100
)
opt_adam._scatter_params(initial_params)

# Single step
opt_adam.step(raw_grads, value_stats)

# Adam state `m` and `v` should match theta_flat shape
assert hasattr(adam, 'm') and hasattr(adam, 'v')
assert adam.m.shape == opt_adam.theta_flat.shape
assert adam.v.shape == opt_adam.theta_flat.shape

print("All clustered_optimizer tests passed!")
