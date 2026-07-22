"""Extended tests for catgen dataset generators."""

import numpy as np
import pytest

from catgen import (
    # Boolean concepts
    generate_xor_parity_dataset,
    generate_dnf_concept_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
    # Geometric
    generate_checkerboard_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
    # Biomedical/synthetic
    generate_epistasis_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
    # Structured
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
)


# ============================================================================
# Boolean Concept Generators
# ============================================================================


class TestBooleanConcepts:
    """Test suite for boolean concept dataset generators."""

    def test_xor_shape_and_values(self):
        """XOR: verify shape and binary values."""
        X, y = generate_xor_parity_dataset(n_obs=100, n_bits=3, random_state=42)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert np.all(np.isin(y, [0, 1]))
        assert np.all(np.isin(X, [0, 1]))

    def test_xor_determinism(self):
        """XOR: same seed produces identical output."""
        X1, y1 = generate_xor_parity_dataset(n_obs=50, n_bits=4, random_state=42)
        X2, y2 = generate_xor_parity_dataset(n_obs=50, n_bits=4, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_xor_different_seeds(self):
        """XOR: different seeds produce different outputs."""
        X1, y1 = generate_xor_parity_dataset(n_obs=50, n_bits=3, random_state=1)
        X2, y2 = generate_xor_parity_dataset(n_obs=50, n_bits=3, random_state=2)
        assert not np.array_equal(X1, X2)

    def test_dnf_shape_and_values(self):
        """DNF: verify shape and binary values."""
        X, y = generate_dnf_concept_dataset(n_obs=80, n_features=6, n_terms=3, random_state=42)
        assert X.shape == (80, 6)
        assert y.shape == (80,)
        assert np.all(np.isin(y, [0, 1]))
        assert np.all(np.isin(X, [0, 1]))

    def test_dnf_determinism(self):
        """DNF: same seed produces identical output."""
        X1, y1 = generate_dnf_concept_dataset(n_obs=100, n_features=5, n_terms=2, random_state=99)
        X2, y2 = generate_dnf_concept_dataset(n_obs=100, n_features=5, n_terms=2, random_state=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_monk1_shape(self):
        """MONK-1: verify output shape."""
        X, y = generate_monk1_dataset(n_obs=200, random_state=42)
        assert X.shape[0] == 200
        assert y.shape == (200,)
        assert np.all(np.isin(y, [0, 1]))

    def test_monk3_shape(self):
        """MONK-3: verify output shape (includes 5% noise)."""
        X, y = generate_monk3_dataset(n_obs=200, random_state=42)
        assert X.shape[0] == 200
        assert y.shape == (200,)
        assert np.all(np.isin(y, [0, 1]))

    def test_overlapping_rules_shape(self):
        """Overlapping rules: verify shape and binary values."""
        X, y = generate_overlapping_rules_dataset(n_obs=150, n_features=8, random_state=42)
        assert X.shape == (150, 8)
        assert y.shape == (150,)
        assert np.all(np.isin(y, [0, 1]))

    def test_modular_sum_shape(self):
        """Modular sum: verify shape and binary values."""
        X, y = generate_modular_sum_dataset(n_obs=120, n_features=6, modulus=2, random_state=42)
        assert X.shape == (120, 6)
        assert y.shape == (120,)
        assert np.all(np.isin(y, [0, 1]))


# ============================================================================
# Geometric Boundary Generators
# ============================================================================


class TestGeometricBoundaries:
    """Test suite for geometric boundary dataset generators."""

    def test_checkerboard_shape(self):
        """Checkerboard: verify shape and values in [0, 1]."""
        X, y = generate_checkerboard_dataset(n_obs=100, grid_size=4, random_state=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert np.all(np.isin(y, [0, 1]))
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_checkerboard_determinism(self):
        """Checkerboard: same seed → same output."""
        X1, y1 = generate_checkerboard_dataset(n_obs=80, grid_size=3, random_state=42)
        X2, y2 = generate_checkerboard_dataset(n_obs=80, grid_size=3, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_circle_shape(self):
        """Circle boundary: verify shape and continuous values."""
        X, y = generate_circle_boundary_dataset(n_obs=120, noise_std=0.05, random_state=42)
        assert X.shape == (120, 2)
        assert y.shape == (120,)
        assert np.all(np.isin(y, [0, 1]))
        assert np.all(X >= -1.5) and np.all(X <= 1.5)

    def test_circle_determinism(self):
        """Circle boundary: determinism check."""
        X1, y1 = generate_circle_boundary_dataset(n_obs=100, noise_std=0.1, random_state=1)
        X2, y2 = generate_circle_boundary_dataset(n_obs=100, noise_std=0.1, random_state=1)
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_diagonal_boundary_shape(self):
        """Diagonal boundary: verify shape."""
        X, y = generate_diagonal_boundary_dataset(n_obs=90, noise_std=0.05, random_state=42)
        assert X.shape == (90, 2)
        assert y.shape == (90,)
        assert np.all(np.isin(y, [0, 1]))

    def test_spiral_shape(self):
        """Spiral: verify shape and continuous values."""
        X, y = generate_spiral_dataset(n_obs=150, noise_std=0.05, random_state=42)
        assert X.shape == (150, 2)
        assert y.shape == (150,)
        assert np.all(np.isin(y, [0, 1]))

    def test_spiral_determinism(self):
        """Spiral: same seed produces same output."""
        X1, y1 = generate_spiral_dataset(n_obs=100, noise_std=0.05, random_state=99)
        X2, y2 = generate_spiral_dataset(n_obs=100, noise_std=0.05, random_state=99)
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_concentric_rings_shape(self):
        """Concentric rings: verify shape."""
        X, y = generate_concentric_rings_dataset(n_obs=200, n_rings=3, noise_std=0.05, random_state=42)
        assert X.shape == (200, 2)
        assert y.shape == (200,)
        # Multi-class (n_rings classes)
        assert len(np.unique(y)) <= 3


# ============================================================================
# Biomedical/Synthetic Generators
# ============================================================================


class TestBiomedicalGenerators:
    """Test suite for biomedical and synthetic scenario generators."""

    def test_epistasis_shape(self):
        """Epistasis: verify shape and binary values."""
        X, y = generate_epistasis_dataset(n_obs=100, n_snp=10, n_genes=2, random_state=42)
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert np.all(np.isin(y, [0, 1]))
        assert np.all(np.isin(X, [0, 1, 2]))  # SNP coding 0/1/2

    def test_epistasis_determinism(self):
        """Epistasis: determinism check."""
        X1, y1 = generate_epistasis_dataset(n_obs=100, n_snp=8, n_genes=3, random_state=123)
        X2, y2 = generate_epistasis_dataset(n_obs=100, n_snp=8, n_genes=3, random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_highdim_lowsample_shape(self):
        """High-dim low-sample: verify shape (p >> n scenario)."""
        X, y = generate_highdim_lowsample_dataset(n_obs=50, n_features=200, random_state=42)
        assert X.shape == (50, 200)
        assert y.shape == (50,)
        assert np.all(np.isin(y, [0, 1]))

    def test_highdim_lowsample_determinism(self):
        """High-dim low-sample: determinism check."""
        X1, y1 = generate_highdim_lowsample_dataset(n_obs=40, n_features=150, random_state=42)
        X2, y2 = generate_highdim_lowsample_dataset(n_obs=40, n_features=150, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_imbalanced_shape(self):
        """Imbalanced: verify shape and class imbalance."""
        X, y = generate_imbalanced_dataset(n_obs=1000, n_features=10, ratio=0.1, random_state=42)
        assert X.shape == (1000, 10)
        assert y.shape == (1000,)
        assert np.all(np.isin(y, [0, 1]))
        # Check class balance is approximately 10%/90%
        minority_ratio = np.sum(y == 1) / len(y)
        assert 0.05 < minority_ratio < 0.15

    def test_imbalanced_determinism(self):
        """Imbalanced: determinism check."""
        X1, y1 = generate_imbalanced_dataset(n_obs=500, n_features=5, ratio=0.2, random_state=7)
        X2, y2 = generate_imbalanced_dataset(n_obs=500, n_features=5, ratio=0.2, random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ============================================================================
# Structured Concept Generators
# ============================================================================


class TestStructuredConcepts:
    """Test suite for structured concept generators."""

    def test_deep_tree_shape(self):
        """Deep tree: verify shape and binary values."""
        X, y = generate_deep_tree_dataset(n_obs=150, n_features=8, depth=3, random_state=42)
        assert X.shape == (150, 8)
        assert y.shape == (150,)
        assert np.all(np.isin(y, [0, 1]))

    def test_deep_tree_determinism(self):
        """Deep tree: determinism check."""
        X1, y1 = generate_deep_tree_dataset(n_obs=100, n_features=6, depth=2, random_state=42)
        X2, y2 = generate_deep_tree_dataset(n_obs=100, n_features=6, depth=2, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_sequential_threshold_shape(self):
        """Sequential threshold: verify shape and binary values."""
        X, y = generate_sequential_threshold_dataset(n_obs=120, n_features=5, random_state=42)
        assert X.shape == (120, 5)
        assert y.shape == (120,)
        assert np.all(np.isin(y, [0, 1]))

    def test_sequential_threshold_determinism(self):
        """Sequential threshold: determinism check."""
        X1, y1 = generate_sequential_threshold_dataset(n_obs=100, n_features=4, random_state=55)
        X2, y2 = generate_sequential_threshold_dataset(n_obs=100, n_features=4, random_state=55)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_hierarchical_interaction_shape(self):
        """Hierarchical interaction: verify shape and binary values."""
        X, y = generate_hierarchical_interaction_dataset(
            n_obs=180,
            n_features=10,
            depth=2,
            random_state=42
        )
        assert X.shape == (180, 10)
        assert y.shape == (180,)
        assert np.all(np.isin(y, [0, 1]))

    def test_hierarchical_interaction_determinism(self):
        """Hierarchical interaction: determinism check."""
        X1, y1 = generate_hierarchical_interaction_dataset(
            n_obs=100,
            n_features=8,
            depth=2,
            random_state=42
        )
        X2, y2 = generate_hierarchical_interaction_dataset(
            n_obs=100,
            n_features=8,
            depth=2,
            random_state=42
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_xor_minimal_bits(self):
        """XOR: minimal bit count (n_bits=1)."""
        X, y = generate_xor_parity_dataset(n_obs=50, n_bits=1, random_state=42)
        assert X.shape == (50, 1)
        assert np.all(np.isin(X, [0, 1]))

    def test_dnf_single_term(self):
        """DNF: single term (minimal complexity)."""
        X, y = generate_dnf_concept_dataset(n_obs=50, n_features=3, n_terms=1, random_state=42)
        assert X.shape == (50, 3)
        assert np.all(np.isin(y, [0, 1]))

    def test_checkerboard_grid_size_one(self):
        """Checkerboard: grid_size=1 (degenerate case)."""
        X, y = generate_checkerboard_dataset(n_obs=50, grid_size=1, random_state=42)
        assert X.shape == (50, 2)
        # All same class in degenerate case, or very imbalanced
        assert len(np.unique(y)) <= 2

    def test_imbalanced_extreme_ratio(self):
        """Imbalanced: extreme minority ratio (1%)."""
        X, y = generate_imbalanced_dataset(n_obs=1000, n_features=5, ratio=0.01, random_state=42)
        minority_count = np.sum(y == 1)
        assert minority_count >= 1  # At least one minority sample

    def test_high_dimensional_minimal_samples(self):
        """High-dim low-sample: extreme case (p=500, n=10)."""
        X, y = generate_highdim_lowsample_dataset(n_obs=10, n_features=500, random_state=42)
        assert X.shape == (10, 500)
        assert y.shape == (10,)

    def test_deep_tree_depth_one(self):
        """Deep tree: depth=1 (shallow tree)."""
        X, y = generate_deep_tree_dataset(n_obs=100, n_features=5, depth=1, random_state=42)
        assert X.shape == (100, 5)
        assert np.all(np.isin(y, [0, 1]))

    def test_concentric_rings_single_ring(self):
        """Concentric rings: n_rings=1 (single ring, binary classification)."""
        X, y = generate_concentric_rings_dataset(n_obs=100, n_rings=1, random_state=42)
        assert X.shape == (100, 2)
        # With single ring, likely binary (inside/outside)
        assert len(np.unique(y)) <= 2


# ============================================================================
# Cross-Generator Consistency Checks
# ============================================================================


class TestConsistency:
    """Test consistency and relationships between generators."""

    def test_all_generators_have_consistent_output_format(self):
        """All generators should return (X, y) tuple with correct dtypes."""
        generators = [
            (generate_xor_parity_dataset, {"n_obs": 50, "n_bits": 3}),
            (generate_dnf_concept_dataset, {"n_obs": 50, "n_features": 5, "n_terms": 2}),
            (generate_monk1_dataset, {"n_obs": 50}),
            (generate_checkerboard_dataset, {"n_obs": 50, "grid_size": 3}),
            (generate_circle_boundary_dataset, {"n_obs": 50}),
            (generate_epistasis_dataset, {"n_obs": 50, "n_snp": 8, "n_genes": 2}),
            (generate_imbalanced_dataset, {"n_obs": 50, "n_features": 5, "ratio": 0.1}),
            (generate_deep_tree_dataset, {"n_obs": 50, "n_features": 6, "depth": 2}),
        ]

        for gen_func, kwargs in generators:
            X, y = gen_func(**kwargs, random_state=42)
            assert isinstance(X, np.ndarray), f"{gen_func.__name__}: X not ndarray"
            assert isinstance(y, np.ndarray), f"{gen_func.__name__}: y not ndarray"
            assert len(X.shape) == 2, f"{gen_func.__name__}: X not 2D"
            assert len(y.shape) == 1, f"{gen_func.__name__}: y not 1D"
            assert X.shape[0] == y.shape[0], f"{gen_func.__name__}: shape mismatch"

    def test_all_generators_respect_random_state(self):
        """All generators should produce reproducible results with random_state."""
        generators = [
            (generate_xor_parity_dataset, {"n_obs": 50, "n_bits": 3}),
            (generate_dnf_concept_dataset, {"n_obs": 50, "n_features": 5, "n_terms": 2}),
            (generate_checkerboard_dataset, {"n_obs": 50, "grid_size": 3}),
            (generate_epistasis_dataset, {"n_obs": 50, "n_snp": 8, "n_genes": 2}),
        ]

        for gen_func, kwargs in generators:
            X1, y1 = gen_func(**kwargs, random_state=42)
            X2, y2 = gen_func(**kwargs, random_state=42)
            np.testing.assert_array_equal(X1, X2, err_msg=f"{gen_func.__name__}: not deterministic")
            np.testing.assert_array_equal(y1, y2, err_msg=f"{gen_func.__name__}: y not deterministic")
