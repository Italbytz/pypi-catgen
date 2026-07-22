# catgen

**catgen** is a Python library for generating synthetic datasets for machine learning benchmarks. It provides a consistent `(X, y)` interface across a wide range of dataset types: boolean concepts, geometric boundaries, SNP genetics, multiplexer functions, epistasis models, and more.

## Installation

```bash
pip install catgen
```

## Quick Start

All generators return `(X, y)` NumPy tuples and accept a `random_state` parameter:

```python
from catgen import (
    generate_xor_parity_dataset,
    generate_checkerboard_dataset,
    generate_multiplexer_dataset,
    generate_snp_glm_dataset,
)

X, y = generate_xor_parity_dataset(n_samples=500, n_bits=6, random_state=0)
X, y = generate_checkerboard_dataset(n_samples=500, random_state=0)
X, y = generate_multiplexer_dataset(n_address_bits=3, random_state=0)
X, y = generate_snp_glm_dataset(n_obs=500, n_snp=20, random_state=0)
```

## Dataset Generators

### Boolean Concepts

```python
from catgen import (
    generate_xor_parity_dataset,
    generate_dnf_concept_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
)

# XOR / k-bit parity
X, y = generate_xor_parity_dataset(n_samples=1000, n_bits=6, random_state=0)

# Disjunctive normal form
X, y = generate_dnf_concept_dataset(n_samples=1000, random_state=0)

# MONK benchmarks
X, y = generate_monk1_dataset()
X, y = generate_monk3_dataset()
```

### Geometric Boundaries

```python
from catgen import (
    generate_checkerboard_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
)

X, y = generate_checkerboard_dataset(n_samples=1000, random_state=0)
X, y = generate_spiral_dataset(n_samples=1000, random_state=0)
X, y = generate_concentric_rings_dataset(n_samples=1000, random_state=0)
```

### k-Multiplexer

```python
from catgen import generate_multiplexer_dataset

# 2 address bits + 4 data bits = 6 features, 64 instances (full enumeration)
X, y = generate_multiplexer_dataset(n_address_bits=2)

# 3 address bits + 8 data bits = 11 features, 2048 instances
X, y = generate_multiplexer_dataset(n_address_bits=3)

# 4 address bits: 2^20 instances – cap with max_samples
X, y = generate_multiplexer_dataset(n_address_bits=4, max_samples=10_000, random_state=0)
```

### Biomedical / High-Dimensional

```python
from catgen import (
    generate_epistasis_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
)

# Gene-gene interaction (epistasis) simulation
X, y = generate_epistasis_dataset(n_samples=1000, n_snps=20, random_state=0)

# High-dimensional, low-sample-size scenario
X, y = generate_highdim_lowsample_dataset(n_samples=50, n_features=500, random_state=0)

# Class-imbalanced binary classification
X, y = generate_imbalanced_dataset(n_samples=1000, imbalance_ratio=0.1, random_state=0)
```

### Structured Concepts

```python
from catgen import (
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
    generate_boolean_concept_dataset,
)

X, y = generate_deep_tree_dataset(n_samples=1000, random_state=0)
X, y = generate_sequential_threshold_dataset(n_samples=1000, random_state=0)
```

## SNP Genetics Simulation

The SNP functions are a Python port of the R package [scrime](https://cran.r-project.org/package=scrime) (Schwender, 2007).

### High-level interface (X, y)

```python
from catgen import generate_snp_glm_dataset, generate_snp_glm_with_covariates_dataset

# Returns (X, y) like all other generators
X, y = generate_snp_glm_dataset(n_obs=1000, n_snp=50, random_state=42)
print(X.shape)  # (1000, 50)
print(y.shape)  # (1000,)

# With continuous covariates: X has shape (n_obs, n_snp + n_covariates)
X, y = generate_snp_glm_with_covariates_dataset(
    n_obs=1000, n_snp=50, n_covariates=3, random_state=42
)
print(X.shape)  # (1000, 53)
```

### Low-level interface (full metadata)

Use `simulate_snp_glm` when you need genotype details, MAF arrays, or
interaction descriptions beyond the `(X, y)` tuple:

```python
from catgen import simulate_snp_glm

sim = simulate_snp_glm(n_obs=1000, n_snp=50, random_state=42)

print(sim.x.shape)   # (1000, 50)  – genotype matrix, coded 0/1/2
print(sim.y.shape)   # (1000,)     – binary response
print(sim.ia)        # interaction descriptions
print(sim.maf)       # per-SNP minor allele frequencies
```

Custom interactions and per-SNP minor allele frequencies:

```python
list_ia  = [[-2, 1], [3]]    # (SNP4 != 2) & (SNP3 == 1)  and  (SNP5 == 3)
list_snp = [[4,  3], [5]]

sim = simulate_snp_glm(
    n_obs=600,
    n_snp=25,
    list_ia=list_ia,
    list_snp=list_snp,
    maf=(0.1, 0.4),           # random MAFs drawn from Uniform(0.1, 0.4)
    random_state=0,
)
```

With continuous covariates:

```python
from catgen import simulate_snp_glm_with_covariates

sim = simulate_snp_glm_with_covariates(
    n_obs=2000,
    n_snp=50,
    list_ia=[-1, -1, [-1, -1]],
    list_snp=[1, 2, [1, 4]],
    covariate_mean=[20.0, 20.0],
    covariate_cov=[[10.0, 5.0], [5.0, 10.0]],
    covariate_beta=[0.2, 0.0],
    covariate_interaction_ia=[-1],
    covariate_interaction_snp=[2],
    covariate_interaction_index=[2],
    covariate_interaction_beta=[0.8],
    random_state=1,
)
```

### Genotype coding

`catgen` uses **0/1/2** coding (numpy convention):

| Code | Genotype |
|------|----------|
| 0 | Homozygous reference (AA) |
| 1 | Heterozygous (Aa) |
| 2 | Homozygous variant (aa) |

The `list_ia` parameter uses scrime's **1/2/3** convention to stay compatible
with the original R API.

### SNP simulation parameters

`simulate_snp_glm` parameters:

| Parameter | Description |
|-----------|-------------|
| `n_obs` | Number of observations. |
| `n_snp` | Number of SNPs to simulate. |
| `list_ia` | Term specification in scrime coding: scalar (main effect) or list (interaction). 1=AA, 2=Aa, 3=aa, negative = NOT. |
| `list_snp` | 1-based SNP indices for each term in `list_ia`. |
| `beta0` | Logistic model intercept (default: −0.5). |
| `beta` | Regression coefficient(s) per interaction term (default: 1.5). |
| `maf` | Minor allele frequency: scalar, `(min, max)` range tuple, or per-SNP array. |
| `sample_y` | Sample y from Bernoulli(prob) if `True`; threshold by `p_cutoff` if `False`. |
| `p_cutoff` | Probability cutoff when `sample_y=False`. |
| `random_state` | Integer seed for reproducibility. |

`SimSNPGlm` return fields:

| Field | Description |
|-------|-------------|
| `x` | `(n_obs, n_snp)` int8 genotype matrix, coded 0/1/2. |
| `y` | `(n_obs,)` int8 binary response array. |
| `beta0` | Intercept used. |
| `beta` | Coefficient array. |
| `ia` | List of interaction description strings. |
| `maf` | `(n_snp,)` array of minor allele frequencies. |
| `prob` | `(n_obs,)` predicted case probabilities. |

## Background

The SNP simulation model follows Nunkesser et al. (2007) and Schwender (2007).
Each SNP is simulated independently under Hardy-Weinberg equilibrium. The binary
response is generated from the logistic model:

$$\text{logit}(P(Y=1)) = \beta_0 + \beta_1 L_1 + \beta_2 L_2 + \ldots$$

where each $L_k$ is a Boolean interaction term.

## References

- Schwender, H. (2007). *Statistical Analysis of Genotype and Gene Expression Data.*
  Dissertation, Department of Statistics, University of Dortmund.
- Nunkesser, R., Bernholt, T., Schwender, H., Ickstadt, K. and Wegener, I. (2007).
  Detecting High-Order Interactions of Single Nucleotide Polymorphisms Using Genetic
  Programming. *Bioinformatics*, 23, 3280–3288.
- Koza, J. R. (1992). *Genetic Programming.* MIT Press.

## License

MIT
