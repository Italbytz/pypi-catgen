"""Boolean concept datasets (XOR/parity, DNF, MONK, overlapping rules, modular sum).

These datasets expose weaknesses of CART and are well-suited for evaluating
rule-set learners.
"""

from __future__ import annotations

import numpy as np


def generate_xor_parity_dataset(
    n_bits: int = 6,
    n_noise_features: int = 14,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an XOR/parity dataset with noise features.

    The class is the parity (XOR) of the first ``n_bits`` binary features.
    Additionally, ``n_noise_features`` random binary features are appended.
    This dataset is non-linearly separable and requires conjunctive rules.

    Parameters
    ----------
    n_bits : int
        Number of relevant parity bits.
    n_noise_features : int
        Number of irrelevant noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bits + n_noise_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    total_features = n_bits + n_noise_features
    X = rng.integers(0, 2, size=(n_samples, total_features))
    y = np.bitwise_xor.reduce(X[:, :n_bits], axis=1)

    perm = rng.permutation(total_features)
    X = X[:, perm]
    return X, y


def generate_dnf_concept_dataset(
    n_disjuncts: int = 3,
    n_conjuncts: int = 2,
    n_noise_features: int = 10,
    n_samples: int = 2000,
    noise_rate: float = 0.0,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a DNF decision rule.

    The true concept is a disjunction of ``n_disjuncts`` conjunctions,
    where each conjunction requires ``n_conjuncts`` different binary features.
    CART must duplicate subtrees for this; rule sets can represent each
    disjunct directly as a separate rule.

    Parameters
    ----------
    n_disjuncts : int
        Number of disjuncts (OR terms).
    n_conjuncts : int
        Number of conjuncts per disjunct (AND terms).
    n_noise_features : int
        Additional irrelevant binary features.
    n_samples : int
        Number of instances.
    noise_rate : float
        Fraction of randomly flipped labels (0 = no noise).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_disjuncts * n_conjuncts + n_noise_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_relevant = n_disjuncts * n_conjuncts
    n_features = n_relevant + n_noise_features
    X = rng.integers(0, 2, size=(n_samples, n_features))

    y = np.zeros(n_samples, dtype=int)
    for d in range(n_disjuncts):
        clause = np.ones(n_samples, dtype=bool)
        for c in range(n_conjuncts):
            clause &= X[:, d * n_conjuncts + c] == 1
        y |= clause.astype(int)

    if noise_rate > 0:
        flip = rng.random(n_samples) < noise_rate
        y[flip] = 1 - y[flip]

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_monk1_dataset(
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the MONK-1 dataset (disjunction: (a1==a2) OR (a5==1)).

    MONK-1 is a classic benchmark for rule learners. The decision rule is a
    disjunction that CART can represent only via subtree duplication.

    Attributes: a1 in {1,2,3}, a2 in {1,2,3}, a3 in {1,2},
                a4 in {1,2,3}, a5 in {1,2,3,4}, a6 in {1,2}

    Parameters
    ----------
    n_samples : int
        Number of instances (sampling with replacement if > 432).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 6), dtype float
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    domains = [3, 3, 2, 3, 4, 2]
    n_total = 1
    for d in domains:
        n_total *= d  # 432

    rows = np.zeros((n_total, 6), dtype=int)
    idx = 0
    for a1 in range(domains[0]):
        for a2 in range(domains[1]):
            for a3 in range(domains[2]):
                for a4 in range(domains[3]):
                    for a5 in range(domains[4]):
                        for a6 in range(domains[5]):
                            rows[idx] = [a1, a2, a3, a4, a5, a6]
                            idx += 1

    y_full = ((rows[:, 0] == rows[:, 1]) | (rows[:, 4] == 1)).astype(int)

    if n_samples >= n_total:
        X, y = rows, y_full
    else:
        chosen = rng.choice(n_total, size=n_samples, replace=True)
        X, y = rows[chosen], y_full[chosen]

    return X.astype(float), y


def generate_monk3_dataset(
    n_samples: int = 2000,
    noise_rate: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the MONK-3 dataset (conjunction + exception + noise).

    MONK-3 rule: (a5 != 3 AND a4 != 1) OR (a5 == 3 AND a2 != 3),
    plus ``noise_rate`` randomly flipped labels.

    Parameters
    ----------
    n_samples : int
        Number of instances.
    noise_rate : float
        Fraction of randomly flipped labels (default: 5%).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 6), dtype float
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    domains = [3, 3, 2, 3, 4, 2]
    n_total = 1
    for d in domains:
        n_total *= d

    rows = np.zeros((n_total, 6), dtype=int)
    idx = 0
    for a1 in range(domains[0]):
        for a2 in range(domains[1]):
            for a3 in range(domains[2]):
                for a4 in range(domains[3]):
                    for a5 in range(domains[4]):
                        for a6 in range(domains[5]):
                            rows[idx] = [a1, a2, a3, a4, a5, a6]
                            idx += 1

    y_full = (
        ((rows[:, 4] != 3) & (rows[:, 3] != 1))
        | ((rows[:, 4] == 3) & (rows[:, 1] != 2))
    ).astype(int)

    if n_samples >= n_total:
        X, y = rows, y_full
    else:
        chosen = rng.choice(n_total, size=n_samples, replace=True)
        X, y = rows[chosen], y_full[chosen]

    if noise_rate > 0:
        flip = rng.random(len(y)) < noise_rate
        y[flip] = 1 - y[flip]

    return X.astype(float), y


def generate_overlapping_rules_dataset(
    n_rules: int = 4,
    n_features_per_rule: int = 2,
    n_noise_features: int = 10,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with multiple independent, overlapping rules.

    Each rule defines a positive region using a threshold over
    ``n_features_per_rule`` continuous features. The class is 1 if at least
    one rule fires. CART struggles because the rules are not hierarchical.

    Parameters
    ----------
    n_rules : int
        Number of independent rules.
    n_features_per_rule : int
        Number of features per rule.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_rules * n_features_per_rule + n_noise_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_relevant = n_rules * n_features_per_rule
    n_total_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_total_features))

    y = np.zeros(n_samples, dtype=int)
    for r in range(n_rules):
        start = r * n_features_per_rule
        end = start + n_features_per_rule
        fired = np.all(X[:, start:end] > 0.6, axis=1)
        y[fired] = 1

    perm = rng.permutation(n_total_features)
    X = X[:, perm]
    return X, y


def generate_modular_sum_dataset(
    n_relevant: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    modulus: int = 3,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset where class depends on a modular sum.

    y = (sum(X[:, :n_relevant]) mod modulus == 0). This pattern creates
    non-axis-parallel decision boundaries that CART can only approximate
    inefficiently.

    Parameters
    ----------
    n_relevant : int
        Number of relevant features.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    modulus : int
        Modulus for the sum function.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_relevant + n_noise_features), dtype float
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    n_total = n_relevant + n_noise_features
    X = rng.integers(0, modulus + 1, size=(n_samples, n_total))
    y = (np.sum(X[:, :n_relevant], axis=1) % modulus == 0).astype(int)

    perm = rng.permutation(n_total)
    X = X[:, perm]
    return X.astype(float), y
