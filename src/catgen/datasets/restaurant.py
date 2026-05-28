"""AIMA restaurant dataset utilities.

Provides three modes around the classic restaurant waiting example:

- the canonical 12 observations used in AIMA,
- the full Cartesian observation space (9216 combinations),
- random sampling from either source.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

RESTAURANT_FEATURE_NAMES = [
    "Alternate",
    "Bar",
    "Fri/Sat",
    "Hungry",
    "Patrons",
    "Price",
    "Raining",
    "Reservation",
    "Type",
    "WaitEstimate",
]

RESTAURANT_FEATURE_DOMAINS: dict[str, tuple[str, ...]] = {
    "Alternate": ("No", "Yes"),
    "Bar": ("No", "Yes"),
    "Fri/Sat": ("No", "Yes"),
    "Hungry": ("No", "Yes"),
    "Patrons": ("None", "Some", "Full"),
    "Price": ("$", "$$", "$$$"),
    "Raining": ("No", "Yes"),
    "Reservation": ("No", "Yes"),
    "Type": ("French", "Thai", "Burger", "Italian"),
    "WaitEstimate": ("0-10", "10-30", "30-60", ">60"),
}

RESTAURANT_AIMA12_EXAMPLES: list[dict[str, str | int]] = [
    {"Alternate": "Yes", "Bar": "No", "Fri/Sat": "No", "Hungry": "Yes", "Patrons": "Some", "Price": "$$$", "Raining": "No", "Reservation": "Yes", "Type": "French", "WaitEstimate": "0-10", "Wait": 1},
    {"Alternate": "Yes", "Bar": "No", "Fri/Sat": "No", "Hungry": "Yes", "Patrons": "Full", "Price": "$", "Raining": "No", "Reservation": "No", "Type": "Thai", "WaitEstimate": "30-60", "Wait": 0},
    {"Alternate": "No", "Bar": "Yes", "Fri/Sat": "No", "Hungry": "No", "Patrons": "Some", "Price": "$", "Raining": "No", "Reservation": "No", "Type": "Burger", "WaitEstimate": "0-10", "Wait": 1},
    {"Alternate": "Yes", "Bar": "No", "Fri/Sat": "Yes", "Hungry": "Yes", "Patrons": "Full", "Price": "$", "Raining": "Yes", "Reservation": "No", "Type": "Thai", "WaitEstimate": "10-30", "Wait": 1},
    {"Alternate": "Yes", "Bar": "No", "Fri/Sat": "Yes", "Hungry": "No", "Patrons": "Full", "Price": "$$$", "Raining": "No", "Reservation": "Yes", "Type": "French", "WaitEstimate": ">60", "Wait": 0},
    {"Alternate": "No", "Bar": "Yes", "Fri/Sat": "No", "Hungry": "Yes", "Patrons": "Some", "Price": "$$", "Raining": "Yes", "Reservation": "Yes", "Type": "Italian", "WaitEstimate": "0-10", "Wait": 1},
    {"Alternate": "No", "Bar": "Yes", "Fri/Sat": "No", "Hungry": "No", "Patrons": "None", "Price": "$", "Raining": "Yes", "Reservation": "No", "Type": "Burger", "WaitEstimate": "0-10", "Wait": 0},
    {"Alternate": "No", "Bar": "No", "Fri/Sat": "No", "Hungry": "Yes", "Patrons": "Some", "Price": "$$", "Raining": "Yes", "Reservation": "Yes", "Type": "Thai", "WaitEstimate": "0-10", "Wait": 1},
    {"Alternate": "No", "Bar": "Yes", "Fri/Sat": "Yes", "Hungry": "No", "Patrons": "Full", "Price": "$", "Raining": "Yes", "Reservation": "No", "Type": "Burger", "WaitEstimate": ">60", "Wait": 0},
    {"Alternate": "Yes", "Bar": "Yes", "Fri/Sat": "Yes", "Hungry": "Yes", "Patrons": "Full", "Price": "$$$", "Raining": "No", "Reservation": "Yes", "Type": "Italian", "WaitEstimate": "10-30", "Wait": 0},
    {"Alternate": "No", "Bar": "No", "Fri/Sat": "No", "Hungry": "No", "Patrons": "None", "Price": "$", "Raining": "No", "Reservation": "No", "Type": "Thai", "WaitEstimate": "0-10", "Wait": 0},
    {"Alternate": "Yes", "Bar": "Yes", "Fri/Sat": "Yes", "Hungry": "Yes", "Patrons": "Full", "Price": "$", "Raining": "No", "Reservation": "No", "Type": "Burger", "WaitEstimate": "30-60", "Wait": 1},
]

_RESTAURANT_CATEGORY_TO_INT: dict[str, dict[str, int]] = {
    feature: {value: idx for idx, value in enumerate(values)}
    for feature, values in RESTAURANT_FEATURE_DOMAINS.items()
}


def _encode_rows(rows: list[tuple[str, ...]]) -> np.ndarray:
    encoded = np.zeros((len(rows), len(RESTAURANT_FEATURE_NAMES)), dtype=np.int8)
    for i, row in enumerate(rows):
        for j, feature in enumerate(RESTAURANT_FEATURE_NAMES):
            encoded[i, j] = _RESTAURANT_CATEGORY_TO_INT[feature][row[j]]
    return encoded


def _row_to_mapping(example: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    if isinstance(example, Mapping):
        return {name: str(example[name]) for name in RESTAURANT_FEATURE_NAMES}
    if len(example) != len(RESTAURANT_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(RESTAURANT_FEATURE_NAMES)} feature values, got {len(example)}"
        )
    return {
        name: str(value)
        for name, value in zip(RESTAURANT_FEATURE_NAMES, example)
    }


def restaurant_decision_rule(example: Mapping[str, str] | Sequence[str]) -> int:
    """Evaluate the AIMA waiting decision tree for one restaurant observation.

    Returns
    -------
    int
        1 if the decision is "Wait", otherwise 0.
    """
    ex = _row_to_mapping(example)

    if ex["Patrons"] == "None":
        return 0
    if ex["Patrons"] == "Some":
        return 1

    wait_estimate = ex["WaitEstimate"]
    if wait_estimate == ">60":
        return 0
    if wait_estimate == "0-10":
        return 1

    if wait_estimate == "30-60":
        if ex["Alternate"] == "No":
            if ex["Reservation"] == "Yes":
                return 1
            return 1 if ex["Bar"] == "Yes" else 0
        return 1 if ex["Fri/Sat"] == "Yes" else 0

    if wait_estimate == "10-30":
        if ex["Hungry"] == "No":
            return 1
        if ex["Alternate"] == "No":
            return 1
        return 1 if ex["Raining"] == "Yes" else 0

    raise ValueError(f"Unexpected WaitEstimate value: {wait_estimate}")


def load_restaurant_aima12_dataset(
    *,
    encode: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical 12 AIMA restaurant observations.

    Parameters
    ----------
    encode : bool
        If True, encode categorical features as integer category indices.
        If False, return string-valued object arrays.
    """
    rows = [
        tuple(str(example[name]) for name in RESTAURANT_FEATURE_NAMES)
        for example in RESTAURANT_AIMA12_EXAMPLES
    ]
    y = np.array([int(example["Wait"]) for example in RESTAURANT_AIMA12_EXAMPLES], dtype=np.int8)

    if encode:
        return _encode_rows(rows), y

    return np.array(rows, dtype=object), y


def generate_restaurant_full_observation_space(
    *,
    encode: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full restaurant observation space (9216 combinations).

    Labels are computed using :func:`restaurant_decision_rule`.
    """
    rows = list(product(*(RESTAURANT_FEATURE_DOMAINS[name] for name in RESTAURANT_FEATURE_NAMES)))
    y = np.array([restaurant_decision_rule(row) for row in rows], dtype=np.int8)

    if encode:
        return _encode_rows(rows), y

    return np.array(rows, dtype=object), y


def sample_restaurant_observations(
    n_samples: int,
    *,
    source: str = "full",
    replace: bool = False,
    encode: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random subset of restaurant observations.

    Parameters
    ----------
    n_samples : int
        Number of observations to draw.
    source : str
        "full" to sample from all 9216 combinations, "aima12" for the
        canonical 12 observations.
    replace : bool
        If True, sample with replacement.
    encode : bool
        If True, return integer-coded features, otherwise strings.
    random_state : int or None
        Seed for reproducibility.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if source == "full":
        X_pool, y_pool = generate_restaurant_full_observation_space(encode=encode)
    elif source == "aima12":
        X_pool, y_pool = load_restaurant_aima12_dataset(encode=encode)
    else:
        raise ValueError("source must be either 'full' or 'aima12'")

    n_pool = len(y_pool)
    if not replace and n_samples > n_pool:
        raise ValueError(
            f"Cannot draw {n_samples} samples without replacement from {n_pool} observations"
        )

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_pool, size=n_samples, replace=replace)
    return X_pool[indices], y_pool[indices]


def restaurant_classification_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute basic binary classification metrics for restaurant labels.

    If ``y_true`` is omitted, metrics are computed against the full 9216-case
    restaurant observation space.
    """
    y_pred = np.asarray(y_pred, dtype=np.int8).reshape(-1)

    if y_true is None:
        _, y_true = generate_restaurant_full_observation_space(encode=True)
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape}, y_true has shape {y_true.shape}"
        )

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n_samples": float(total),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
