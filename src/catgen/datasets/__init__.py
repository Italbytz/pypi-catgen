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
snp_generators
    SNP dataset generators with unified (X, y) interface.
"""

from catgen.datasets.multiplexer import generate_multiplexer_dataset
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
from catgen.datasets.snp_generators import (
    generate_snp_glm_dataset,
    generate_snp_glm_with_covariates_dataset,
)

__all__ = [
    # multiplexer
    "generate_multiplexer_dataset",
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
    # snp_generators
    "generate_snp_glm_dataset",
    "generate_snp_glm_with_covariates_dataset",
]
