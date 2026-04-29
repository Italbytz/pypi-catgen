"""Structured datasets (deep trees, sequential thresholds, hierarchical interactions).

These datasets have inherent tree or hierarchical structure and are
therefore natural for CART but challenging for flat rule-set learners.
"""

from __future__ import annotations

import numpy as np


def generate_deep_tree_dataset(
    depth: int = 5,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset whose true concept is a deep binary tree.

    At each internal node, a different feature is split on
    (feature i, threshold 0.5). The leaf determines class (0 or 1,
    alternating). CART can represent this tree directly; rule sets need
    2^(depth-1) rules because each path is a separate conjunction.

    Parameters
    ----------
    depth : int
        Depth of the true tree (requires at least ``depth`` features).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, depth + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_features = depth + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        node = 0
        for t in range(depth):
            if X[i, t] > 0.5:
                node = 2 * node + 2
            else:
                node = 2 * node + 1
        y[i] = node % 2

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_sequential_threshold_dataset(
    n_bins: int = 5,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with sequential thresholds on one feature.

    Class depends on which of ``n_bins`` intervals a single continuous
    feature falls into: even bins -> class 0, odd bins -> class 1.
    CART needs only (n_bins - 1) splits; rule sets must create one rule
    with lower and upper bound for each interval.

    Parameters
    ----------
    n_bins : int
        Number of intervals (alternating class 0/1).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1 + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_features = 1 + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    bin_idx = np.clip(np.floor(X[:, 0] * n_bins).astype(int), 0, n_bins - 1)
    y = (bin_idx % 2).astype(int)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_hierarchical_interaction_dataset(
    n_context_features: int = 3,
    n_response_features: int = 3,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with hierarchical feature interaction.

    A context feature first determines which response feature is relevant.
    Class then depends on the value of the selected response feature.
    CART represents this naturally as a tree (split on context, then on
    response); rule sets cannot encode this conditional relevance compactly.

    Parameters
    ----------
    n_context_features : int
        Number of possible contexts (one feature, n values).
    n_response_features : int
        Number of response features (one relevant per context).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1 + n_response_features + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_used_responses = min(n_context_features, n_response_features)
    n_features = 1 + n_response_features + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    context = np.clip(
        np.floor(X[:, 0] * n_context_features).astype(int), 0, n_context_features - 1
    )
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        resp_idx = context[i] % n_used_responses
        y[i] = int(X[i, 1 + resp_idx] > 0.5)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y
