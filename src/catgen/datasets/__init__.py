"""catgen.datasets – synthetic benchmark dataset generators.

All generators return ``(X, y)`` tuples of NumPy arrays and accept a
``random_state`` keyword argument for reproducibility.

Submodules
----------
multiplexer
    k-multiplexer boolean classification tasks.
boolean_concepts
    XOR/parity, DNF concepts, MONK benchmarks, overlapping rules.
biomedical
    SNP epistasis (GAMETES-style), high-dimensional low-sample, imbalanced.
geometric
    Checkerboard, circle/spiral/ring boundaries.
structured
    Modular sum, deep tree, sequential thresholds, hierarchical interactions.
restaurant
    AIMA restaurant examples, full observation space, and sampling.
"""

from catgen.datasets.multiplexer import (
    generate_multiplexer_dataset,
    load_multiplexer_datasets,
)
from catgen.datasets.boolean_concepts import (
    generate_xor_parity_dataset,
    generate_dnf_concept_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
)
from catgen.datasets.biomedical import (
    generate_epistasis_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
)
from catgen.datasets.geometric import (
    generate_checkerboard_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
)
from catgen.datasets.structured import (
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
)
from catgen.datasets.restaurant import (
    RESTAURANT_AIMA12_EXAMPLES,
    RESTAURANT_FEATURE_NAMES,
    RESTAURANT_FEATURE_DOMAINS,
    restaurant_decision_rule,
    load_restaurant_aima12_dataset,
    generate_restaurant_full_observation_space,
    sample_restaurant_observations,
    restaurant_classification_metrics,
)

__all__ = [
    # multiplexer
    "generate_multiplexer_dataset",
    "load_multiplexer_datasets",
    # boolean concepts
    "generate_xor_parity_dataset",
    "generate_dnf_concept_dataset",
    "generate_monk1_dataset",
    "generate_monk3_dataset",
    "generate_overlapping_rules_dataset",
    "generate_modular_sum_dataset",
    # biomedical
    "generate_epistasis_dataset",
    "generate_highdim_lowsample_dataset",
    "generate_imbalanced_dataset",
    # geometric
    "generate_checkerboard_dataset",
    "generate_circle_boundary_dataset",
    "generate_diagonal_boundary_dataset",
    "generate_spiral_dataset",
    "generate_concentric_rings_dataset",
    # structured
    "generate_deep_tree_dataset",
    "generate_sequential_threshold_dataset",
    "generate_hierarchical_interaction_dataset",
    # restaurant
    "RESTAURANT_AIMA12_EXAMPLES",
    "RESTAURANT_FEATURE_NAMES",
    "RESTAURANT_FEATURE_DOMAINS",
    "restaurant_decision_rule",
    "load_restaurant_aima12_dataset",
    "generate_restaurant_full_observation_space",
    "sample_restaurant_observations",
    "restaurant_classification_metrics",
]
