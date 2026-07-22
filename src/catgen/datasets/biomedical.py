"""Biomedical synthetic datasets (epistasis, high-dimensional, imbalanced).

These datasets simulate scenarios common in computational biology and
rare-disease research.
"""

from __future__ import annotations

import numpy as np


def generate_epistasis_dataset(
    n_samples: int = 1600,
    n_snps: int = 20,
    n_interacting: int = 2,
    heritability: float = 0.4,
    minor_allele_freq: float = 0.3,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic epistasis dataset (SNP-SNP interaction).

    The class is determined exclusively by an interaction among
    ``n_interacting`` SNPs; all other SNPs are noise features.
    The model emulates GAMETES behavior: causal SNPs jointly create an
    XOR-like pattern (no main effect, only interaction).

    Parameters
    ----------
    n_samples : int
        Number of instances.
    n_snps : int
        Total number of SNP features (including causal SNPs).
    n_interacting : int
        Number of interacting causal SNPs (2 or 3 recommended).
    heritability : float
        Strength of genetic signal (0 = no signal, 1 = perfectly separable).
        Controls the penetrance table.
    minor_allele_freq : float
        Minor allele frequency for Hardy-Weinberg equilibrium (0 < maf < 0.5).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_snps), dtype int  (values: 0, 1, 2)
    y : np.ndarray, shape (n_samples,), dtype int  (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    if n_interacting > n_snps:
        raise ValueError("n_interacting must not exceed n_snps")
    if not (0 < minor_allele_freq < 0.5):
        raise ValueError("minor_allele_freq must be in (0, 0.5)")

    p = minor_allele_freq
    hw_probs = [(1 - p) ** 2, 2 * p * (1 - p), p ** 2]
    X = rng.choice(3, size=(n_samples, n_snps), p=hw_probs)

    causal = X[:, :n_interacting]
    interaction_signal = np.sum(causal, axis=1) % 2

    pen_high = min(1.0, 0.5 + heritability / 2)
    pen_low = max(0.0, 0.5 - heritability / 2)
    probs = np.where(interaction_signal == 1, pen_high, pen_low)
    y = (rng.random(n_samples) < probs).astype(int)

    perm = rng.permutation(n_snps)
    X = X[:, perm]
    return X, y


def generate_highdim_lowsample_dataset(
    n_samples: int = 120,
    n_features: int = 500,
    n_informative: int = 5,
    n_classes: int = 2,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a high-dimensional dataset with few samples (p >> n).

    Typical for genomics scenarios (e.g. microarray, SNP panels): many features,
    few patients, and only a few informative features.

    Parameters
    ----------
    n_samples : int
        Number of instances (typically small, e.g. 80–200).
    n_features : int
        Total number of features (typically large, e.g. 200–2000).
    n_informative : int
        Number of truly informative features.
    n_classes : int
        Number of classes.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype int
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=n_classes,
        flip_y=0.03,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def generate_imbalanced_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    imbalance_ratio: float = 0.1,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with strong class imbalance.

    Simulates rare diseases or rare events.

    Parameters
    ----------
    n_samples : int
        Total number of instances.
    n_features : int
        Number of features.
    n_informative : int
        Number of informative features.
    imbalance_ratio : float
        Minority class ratio (e.g. 0.1 = 10%).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0.01,
        random_state=random_state,
    )
    return X, y
