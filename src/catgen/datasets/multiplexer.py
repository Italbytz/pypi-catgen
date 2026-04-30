"""Multiplexer boolean classification datasets.

A k-multiplexer has ``k`` address bits and ``2**k`` data bits.
The output is the data bit selected by the address bits.

Reference
---------
Koza, J. R. (1992). Genetic Programming. MIT Press.
"""

from __future__ import annotations

import numpy as np

# Standard configurations: name -> number of address bits.
_MUX_CONFIGS: tuple[tuple[str, int], ...] = (
    ("mux_6", 2),     # 2 address bits + 4 data bits  = 6 features, 2^6  = 64 instances
    ("mux_11", 3),    # 3 address bits + 8 data bits  = 11 features, 2^11 = 2048 instances
    ("mux_20", 4),    # 4 address bits + 16 data bits = 20 features, 2^20 = 1_048_576 instances
)


def generate_multiplexer_dataset(
    n_address_bits: int,
    *,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a multiplexer dataset.

    A k-multiplexer has ``k`` address bits and ``2**k`` data bits.
    The total number of features is ``k + 2**k``.
    The output value is the data bit at the position encoded by address bits.

    With full enumeration, the dataset contains ``2**(k + 2**k)`` rows.
    For large k, ``max_samples`` can cap row count via random sampling.

    Parameters
    ----------
    n_address_bits : int
        Number of address bits (1, 2, 3, ...).
    max_samples : int or None
        Maximum row count. ``None`` = full enumeration.
    random_state : int or None
        Seed for reproducible sampling (only relevant if ``max_samples`` is set).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    n_data_bits = 1 << n_address_bits  # 2**k
    n_features = n_address_bits + n_data_bits
    n_total = 1 << n_features  # 2**(k + 2**k)

    use_sampling = max_samples is not None and max_samples < n_total

    if use_sampling:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_total, size=max_samples, replace=False)
        indices.sort()
    else:
        indices = np.arange(n_total)

    n_samples = len(indices)
    X = np.zeros((n_samples, n_features), dtype=int)
    y = np.zeros(n_samples, dtype=int)

    for row, idx in enumerate(indices):
        bits = [(idx >> b) & 1 for b in range(n_features)]
        X[row] = bits
        address = sum(bits[a] << a for a in range(n_address_bits))
        y[row] = bits[n_address_bits + address]

    return X, y


def load_multiplexer_datasets(
    *,
    max_samples_large: int = 10_000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate standard multiplexer datasets (mux_6 to mux_20).

    For large multiplexers (>= 2^16 instances), sampling is used to keep
    runtime manageable.

    Parameters
    ----------
    max_samples_large : int
        Maximum row count for large multiplexers (default: 10_000).

    Returns
    -------
    dict mapping ``name`` -> ``(X, y)`` tuples.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, n_addr in _MUX_CONFIGS:
        n_features = n_addr + (1 << n_addr)
        n_total = 1 << n_features
        if n_total > max_samples_large:
            X, y = generate_multiplexer_dataset(
                n_addr, max_samples=max_samples_large, random_state=42
            )
        else:
            X, y = generate_multiplexer_dataset(n_addr)
        result[name] = (X, y)
    return result
