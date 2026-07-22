"""catgen – Categorical data generator for machine learning benchmarks.

Includes SNP genetics simulation, k-multiplexer boolean datasets, DNF
concepts, MONK benchmarks, restaurant examples, epistasis, geometric
boundaries, and more.
"""

__version__ = "0.1.0"

from catgen.simulation.snp import (
    SimSNPGlm,
    SimSNPCovariateGlm,
    simulate_snp_glm,
    simulate_snp_glm_with_covariates,
)
from catgen.datasets import (
    generate_multiplexer_dataset,
    load_multiplexer_datasets,
    generate_xor_parity_dataset,
    generate_dnf_concept_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
    generate_epistasis_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
    generate_checkerboard_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
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
    "__version__",
    "SimSNPGlm",
    "SimSNPCovariateGlm",
    "simulate_snp_glm",
    "simulate_snp_glm_with_covariates",
    "generate_multiplexer_dataset",
    "load_multiplexer_datasets",
    "generate_xor_parity_dataset",
    "generate_dnf_concept_dataset",
    "generate_monk1_dataset",
    "generate_monk3_dataset",
    "generate_overlapping_rules_dataset",
    "generate_modular_sum_dataset",
    "generate_epistasis_dataset",
    "generate_highdim_lowsample_dataset",
    "generate_imbalanced_dataset",
    "generate_checkerboard_dataset",
    "generate_circle_boundary_dataset",
    "generate_diagonal_boundary_dataset",
    "generate_spiral_dataset",
    "generate_concentric_rings_dataset",
    "generate_deep_tree_dataset",
    "generate_sequential_threshold_dataset",
    "generate_hierarchical_interaction_dataset",
    "RESTAURANT_AIMA12_EXAMPLES",
    "RESTAURANT_FEATURE_NAMES",
    "RESTAURANT_FEATURE_DOMAINS",
    "restaurant_decision_rule",
    "load_restaurant_aima12_dataset",
    "generate_restaurant_full_observation_space",
    "sample_restaurant_observations",
    "restaurant_classification_metrics",
]
