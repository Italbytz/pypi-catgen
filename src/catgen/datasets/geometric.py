"""Geometric boundary datasets (checkerboard, circle, spiral, rings, diagonal).

These datasets feature non-axis-parallel decision boundaries that
challenge rule-set learners and CART alike.
"""

from __future__ import annotations

import numpy as np


def generate_checkerboard_dataset(
    n_tiles: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D checkerboard dataset with noise features.

    The checkerboard pattern (XOR over quantized continuous features)
    requires O(n_tiles^2) leaves for CART, while rule sets can represent
    regions directly as rules.

    Parameters
    ----------
    n_tiles : int
        Number of tiles per axis (e.g. 4 -> 4x4 checkerboard).
    n_noise_features : int
        Additional uniform noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2 + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    X_rel = rng.uniform(0, 1, size=(n_samples, 2))
    tile_x = np.floor(X_rel[:, 0] * n_tiles).astype(int)
    tile_y = np.floor(X_rel[:, 1] * n_tiles).astype(int)
    y = ((tile_x + tile_y) % 2).astype(int)

    X_noise = rng.uniform(0, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])

    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y


def generate_circle_boundary_dataset(
    n_noise_features: int = 8,
    n_samples: int = 2000,
    radius: float = 0.7,
    noise_std: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a circular decision boundary.

    Class 1 if the point lies inside a circle (Euclidean distance from
    center < radius). Axis-parallel rules (CART and rule sets) must
    approximate the circle with many rectangles.

    Parameters
    ----------
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    radius : float
        Radius of the circle boundary (features in [0, 1], center (0.5, 0.5)).
    noise_std : float
        Gaussian noise on the radius.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2 + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    X_rel = rng.uniform(0, 1, size=(n_samples, 2))
    dist = np.sqrt((X_rel[:, 0] - 0.5) ** 2 + (X_rel[:, 1] - 0.5) ** 2)
    effective_radius = radius / 2
    y = (dist + rng.normal(0, noise_std, n_samples) < effective_radius).astype(int)

    X_noise = rng.uniform(0, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y


def generate_diagonal_boundary_dataset(
    n_relevant: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    noise_std: float = 0.1,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a diagonal (45 degree) decision boundary.

    The true boundary is a hyperplane sum(X[:, :n_relevant]) > threshold.
    Axis-parallel methods need a staircase approximation; linear models,
    SVM, or kNN solve this directly.

    Parameters
    ----------
    n_relevant : int
        Number of relevant features (their sum determines class).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    noise_std : float
        Gaussian noise on the decision function.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_relevant + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    decision = np.sum(X[:, :n_relevant], axis=1) + rng.normal(0, noise_std, n_samples)
    threshold = n_relevant / 2.0
    y = (decision > threshold).astype(int)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_spiral_dataset(
    n_samples: int = 2000,
    n_noise_features: int = 8,
    noise_std: float = 0.15,
    n_turns: float = 1.5,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-spirals dataset with noise features.

    Two interleaved spirals are a classic benchmark for nonlinear classifiers.
    Neither CART nor rule sets can approximate spiral structure efficiently.

    Parameters
    ----------
    n_samples : int
        Total number of instances (half per spiral).
    n_noise_features : int
        Additional noise features.
    noise_std : float
        Radial noise.
    n_turns : float
        Number of spiral turns.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2 + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n_half)) * n_turns * 2 * np.pi

    r1 = theta + rng.normal(0, noise_std, n_half)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    r2 = theta + rng.normal(0, noise_std, n_half)
    x2 = -r2 * np.cos(theta)
    y2 = -r2 * np.sin(theta)

    X_rel = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_half)]).astype(int)

    X_noise = rng.uniform(X_rel.min(), X_rel.max(), size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])

    shuffle = rng.permutation(n_samples)
    X = X[shuffle]
    labels = labels[shuffle]
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, labels


def generate_concentric_rings_dataset(
    n_rings: int = 3,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    noise_std: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with concentric rings (alternating classes).

    Ring i has class (i mod 2). This pattern requires circular decision
    boundaries that axis-parallel methods only approximate coarsely.

    Parameters
    ----------
    n_rings : int
        Number of rings.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    noise_std : float
        Radial Gaussian noise.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2 + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    angle = rng.uniform(0, 2 * np.pi, n_samples)
    ring_idx = rng.integers(0, n_rings, n_samples)
    radius = (ring_idx + 0.5) / n_rings + rng.normal(0, noise_std, n_samples)
    X_rel = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = (ring_idx % 2).astype(int)

    X_noise = rng.uniform(-1, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y
