# Changelog – catgen

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(No unreleased changes yet.)

## [0.1.0] – 2026-04-29

### Added

- **SNP Dataset Generators** (unified `(X, y)` interface)
  - `generate_snp_glm_dataset()`: Wrapper around `simulate_snp_glm()`
  - `generate_snp_glm_with_covariates_dataset()`: Wrapper with covariates
  - Consistent interface with all other dataset generators (return `(X, y)` tuples)
  - Low-level `simulate_snp_glm*()` functions still available for detailed metadata

- **SNP Genetics Simulation** (low-level API)
  - `generate_xor_parity_dataset()`: XOR and k-bit parity functions
  - `generate_dnf_concept_dataset()`: Disjunctive normal form (DNF) expressions
  - `generate_monk1_dataset()`, `generate_monk3_dataset()`: MONK classification benchmarks
  - `generate_overlapping_rules_dataset()`: Redundant disjunctive rule sets
  - `generate_modular_sum_dataset()`: Modular arithmetic classification

- **Geometric Boundary Generators**
  - `generate_checkerboard_dataset()`: Grid-structured checkerboard pattern
  - `generate_circle_boundary_dataset()`: Concentric circle classification
  - `generate_diagonal_boundary_dataset()`: Diagonal decision boundaries
  - `generate_spiral_dataset()`: Logarithmic spiral classification
  - `generate_concentric_rings_dataset()`: Multiple concentric ring patterns

- **Biomedical/Synthetic Generators**
  - `generate_epistasis_dataset()`: Gene-gene interaction (epistasis) simulation
  - `generate_highdim_lowsample_dataset()`: High-dimensional low-sample-size scenarios
  - `generate_imbalanced_dataset()`: Class-imbalanced binary classification

- **Structured Concept Generators**
  - `generate_deep_tree_dataset()`: Decision tree with depth-based patterns
  - `generate_sequential_threshold_dataset()`: Sequential threshold rules
  - `generate_hierarchical_interaction_dataset()`: Nested interaction structures

- **k-Multiplexer Datasets**
  - `generate_multiplexer_dataset()`: k-multiplexer boolean function
  - `load_multiplexer_datasets()`: Predefined multiplexer (3x8, 4x16, 5x32)

- **Unit Tests** (~360 lines)
  - `test_simulation.py`: SNP simulation shape, value ranges, MAF variants, interaction patterns
  - `test_dataset_generators.py`: 40+ tests for boolean, geometric, biomedical, structured generators with edge cases

### Changed

- n/a (first release)

### Fixed

- n/a (first release)

### Notes

- Package is alpha (0.1.0) and may see API changes
- Primary use case: machine learning benchmark dataset generation
- Ported from R package [scrime](https://cran.r-project.org/package=scrime) (Schwender, 2007)
- See README.md for detailed usage examples and API documentation

---

**Links:**
- [PyPI: catgen](https://pypi.org/project/catgen/)
- [GitHub: Italbytz/pypi-catgen](https://github.com/Italbytz/pypi-catgen)
- [scrime R Package](https://cran.r-project.org/package=scrime)
