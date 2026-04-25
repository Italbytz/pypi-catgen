# catgen

**catgen** is a Python library for simulation and analysis of high-dimensional categorical data, with a primary focus on SNP (Single Nucleotide Polymorphism) genotype data.

It is a Python port of the R package [scrime](https://cran.r-project.org/package=scrime) (Schwender, 2007), focused initially on the simulation functions used in genetic association studies.

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

## API

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
| `list_ia` | Interaction specification (scrime coding: 1=AA, 2=Aa, 3=aa; negative = NOT). |
| `list_snp` | 1-based SNP indices for each interaction. |
| `beta0` | Logistic model intercept. |
| `beta` | Regression coefficient(s) for each interaction term. |
| `maf` | Minor allele frequency: scalar, `(min, max)` tuple, or per-SNP array. |
| `sample_y` | Sample y from Bernoulli(prob) if True; threshold by `p_cutoff` if False. |
| `p_cutoff` | Probability cutoff when `sample_y=False`. |
| `random_state` | Integer seed for reproducibility. |

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
