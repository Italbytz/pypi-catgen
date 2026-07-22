# catgen

**catgen** is a Python library for generating synthetic datasets for machine learning benchmarks, with a focus on categorical and high-dimensional data.

The SNP simulation functions (`simulate_snp_glm`, `simulate_snp_glm_with_covariates`) are a Python port of the R package [scrime](https://cran.r-project.org/package=scrime) (Schwender, 2007). All other dataset generators (k-multiplexer, DNF concepts, geometric boundaries, etc.) are independent implementations.

## Installation

```bash
pip install catgen
```

## Quick Start

Simulate 1,000 observations with 50 SNPs using the default interaction model from Nunkesser et al. (2007):

```python
from catgen import simulate_snp_glm

sim = simulate_snp_glm(n_obs=1000, n_snp=50, random_state=42)

print(sim.x.shape)   # (1000, 50)  – genotype matrix, coded 0/1/2
print(sim.y.shape)   # (1000,)     – binary response
print(sim.ia)        # interaction descriptions
```

Specify custom interactions and per-SNP minor allele frequencies:

```python
list_ia  = [[-2, 1], [3]]        # (SNP4 != 2) & (SNP3 == 1)  and  (SNP5 == 3)
list_snp = [[4,  3], [5]]

sim2 = simulate_snp_glm(
    n_obs=600,
    n_snp=25,
    list_ia=list_ia,
    list_snp=list_snp,
    maf=(0.1, 0.4),              # random MAFs drawn from Uniform(0.1, 0.4)
    random_state=0,
)
```

Use scrime-style mixed terms when you want to combine main effects and
interaction effects in one model specification:

```python
from catgen import simulate_snp_glm

sim3 = simulate_snp_glm(
    n_obs=2000,
    n_snp=50,
    list_ia=[-1, -1, -1, [-1, -1]],
    list_snp=[1, 2, 3, [4, 5]],
    beta0=0.0,
    beta=[0.2, 0.2, 0.2, 0.5],
    maf=(0.15, 0.45),
    random_state=1,
  )
```

Add continuous covariates and SNP-covariate interaction terms when you want to
mirror environmental simulation scenarios:

```python
from catgen import simulate_snp_glm_with_covariates

sim4 = simulate_snp_glm_with_covariates(
    n_obs=2000,
    n_snp=50,
    list_ia=[-1, -1, -1, [-1, -1]],
    list_snp=[1, 2, 3, [1, 4]],
    beta0=0.0,
    beta=[0.2, 0.3, 0.4, 0.6],
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

Generate SNP data with a unified `(X, y)` interface for benchmarking:

```python
from catgen import generate_snp_glm_dataset, generate_snp_glm_with_covariates_dataset

# Generate SNP data (returns X, y tuple like other dataset generators)
X, y = generate_snp_glm_dataset(n_obs=1000, n_snp=50, random_state=42)
print(X.shape)  # (1000, 50)
print(y.shape)  # (1000,)

# With covariates: returns combined feature matrix (SNPs + covariates)
X_cov, y_cov = generate_snp_glm_with_covariates_dataset(
    n_obs=1000,
    n_snp=50,
    n_covariates=3,
    random_state=42,
)
print(X_cov.shape)  # (1000, 53) – 50 SNPs + 3 covariates

# For full simulation details and metadata, use low-level functions:
from catgen import simulate_snp_glm
sim = simulate_snp_glm(n_obs=1000, n_snp=50, random_state=42)
print(sim.ia)    # interaction descriptions
print(sim.maf)   # minor allele frequencies
```

## Dataset Generators

All generators return `(X, y)` tuples of NumPy arrays with optional `random_state` for reproducibility.

### SNP/Genetics Generators

#### `generate_snp_glm_dataset` (unified interface)

```python
generate_snp_glm_dataset(
    n_obs=1000,
    n_snp=50,
    list_ia=None,
    list_snp=None,
    beta0=-0.5,
    beta=1.5,
    maf=0.25,
    sample_y=True,
    p_cutoff=0.5,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]
```

Wrapper around `simulate_snp_glm()` with standard `(X, y)` output.
For full simulation metadata, use `simulate_snp_glm()` directly.

#### `generate_snp_glm_with_covariates_dataset` (unified interface)

```python
generate_snp_glm_with_covariates_dataset(
    n_obs=1000,
    n_snp=50,
    n_covariates=2,
    list_ia=None,
    list_snp=None,
    ...
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]
```

Wrapper with continuous covariates. Returns combined `(X_combined, y)` where
X_combined has shape `(n_obs, n_snp + n_covariates)`.

### Boolean & Geometric Generators

- `generate_xor_parity_dataset()`, `generate_dnf_concept_dataset()`
- `generate_monk1_dataset()`, `generate_monk3_dataset()`
- `generate_checkerboard_dataset()`, `generate_circle_boundary_dataset()`
- `generate_spiral_dataset()`, `generate_concentric_rings_dataset()`
- `generate_multiplexer_dataset(n_address_bits, max_samples=None)`
- and more...

All follow the same `(X, y)` convention.

## Low-Level SNP Simulation API

### `simulate_snp_glm`

```python
simulate_snp_glm(
    n_obs=1000,
    n_snp=50,
    list_ia=None,
    list_snp=None,
    beta0=-0.5,
    beta=1.5,
    maf=0.25,
    sample_y=True,
    p_cutoff=0.5,
    random_state=None,
) -> SimSNPGlm
```

Simulates SNP genotypes under Hardy-Weinberg equilibrium and generates a binary
response via a logistic regression model.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `n_obs` | Number of observations. |
| `n_snp` | Number of SNPs to simulate. |
| `list_ia` | Term specification in scrime coding: each term may be a scalar main effect or an array-like interaction term; 1=AA, 2=Aa, 3=aa, negative = NOT. |
| `list_snp` | 1-based SNP indices for each term; each entry may be a scalar or an array-like matching the corresponding `list_ia` term. |
| `beta0` | Logistic model intercept. |
| `beta` | Regression coefficient(s) for each interaction term. |
| `maf` | Minor allele frequency: scalar, `(min, max)` tuple, or per-SNP list/NumPy array. A tuple is always interpreted as a sampling range. |
| `sample_y` | Sample y from Bernoulli(prob) if True; threshold by `p_cutoff` if False. |
| `p_cutoff` | Probability cutoff when `sample_y=False`. |
| `random_state` | Integer seed for reproducibility. |

If you rely on the historical scrime default interaction terms, `n_snp` must be at
least 10. For smaller SNP panels, pass explicit `list_ia` and `list_snp` values.

**Returns** `SimSNPGlm` with fields:

| Field | Description |
|-------|-------------|
| `x` | `(n_obs, n_snp)` int8 array, coded 0/1/2. |
| `y` | `(n_obs,)` int8 binary response array. |
| `beta0` | Intercept used. |
| `beta` | Coefficient array. |
| `ia` | List of interaction description strings. |
| `maf` | `(n_snp,)` array of minor allele frequencies. |
| `prob` | `(n_obs,)` predicted case probabilities. |

### `simulate_snp_glm_with_covariates`

```python
simulate_snp_glm_with_covariates(
  n_obs=1000,
  n_snp=50,
  list_ia=None,
  list_snp=None,
  beta0=-0.5,
  beta=1.5,
  maf=0.25,
  covariates=None,
  covariate_mean=None,
  covariate_cov=None,
  covariate_beta=None,
  covariate_interaction_ia=None,
  covariate_interaction_snp=None,
  covariate_interaction_index=None,
  covariate_interaction_beta=None,
  sample_y=True,
  p_cutoff=0.5,
  random_state=None,
) -> SimSNPCovariateGlm
```

Extends the SNP logistic model with continuous covariates. Covariates can be
passed explicitly via `covariates` or sampled from a multivariate normal using
`covariate_mean` and `covariate_cov`.

**Additional parameters**

| Parameter | Description |
|-----------|-------------|
| `covariates` | Explicit `(n_obs, n_covariates)` covariate matrix. If provided, no Gaussian sampling is performed. |
| `covariate_mean` | Mean vector for Gaussian covariate sampling. A scalar creates one covariate. |
| `covariate_cov` | Covariance specification for covariate sampling: scalar variance, per-covariate variance vector, or full covariance matrix. |
| `covariate_beta` | Additive coefficient(s) for the continuous covariates. Defaults to zero when omitted. |
| `covariate_interaction_ia` | scrime-style SNP term specification for SNP-covariate interactions. |
| `covariate_interaction_snp` | 1-based SNP indices for `covariate_interaction_ia`. |
| `covariate_interaction_index` | 1-based covariate index for each SNP-covariate interaction term. |
| `covariate_interaction_beta` | Coefficient(s) for the SNP-covariate interaction terms. Defaults to zero when omitted. |

**Additional return fields**

| Field | Description |
|-------|-------------|
| `covariates` | `(n_obs, n_covariates)` continuous covariate matrix. |
| `covariate_beta` | Additive covariate coefficients used in the model. |
| `covariate_names` | Generated covariate names such as `E1`, `E2`, ... |
| `covariate_interaction_beta` | Coefficients for SNP-covariate interaction terms. |
| `covariate_interactions` | Human-readable names for SNP-covariate interaction terms. |

### Genotype coding

`catgen` uses **0/1/2** coding (numpy convention):

| Code | Genotype |
|------|----------|
| 0 | Homozygous reference (AA) |
| 1 | Heterozygous (Aa) |
| 2 | Homozygous variant (aa) |

The `list_ia` parameter uses scrime's **1/2/3** convention to stay compatible with
the original R API.

## Background

The simulation model follows Nunkesser et al. (2007) and Schwender (2007):

- Each SNP is simulated independently under Hardy-Weinberg equilibrium.
- A binary response is generated from the logistic model:

$$\text{logit}(P(Y=1)) = \beta_0 + \beta_1 L_1 + \beta_2 L_2 + \ldots$$

where each $L_k$ is a Boolean interaction term.

## References

- Schwender, H. (2007). *Statistical Analysis of Genotype and Gene Expression Data.*
  Dissertation, Department of Statistics, University of Dortmund.
- Nunkesser, R., Bernholt, T., Schwender, H., Ickstadt, K. and Wegener, I. (2007).
  Detecting High-Order Interactions of Single Nucleotide Polymorphisms Using Genetic
  Programming. *Bioinformatics*, 23, 3280–3288.

## License

MIT
