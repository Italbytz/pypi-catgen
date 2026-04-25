"""SNP simulation functions – Python port of R's scrime package.

Reference
---------
Schwender, H. (2007). Statistical Analysis of Genotype and Gene Expression
Data. Dissertation, Department of Statistics, University of Dortmund.

Nunkesser, R., Bernholt, T., Schwender, H., Ickstadt, K. and Wegener, I.
(2007). Detecting High-Order Interactions of Single Nucleotide Polymorphisms
Using Genetic Programming. Bioinformatics, 23, 3280-3288.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class SimSNPGlm:
    """Result of :func:`simulate_snp_glm`.

    Attributes
    ----------
    x : np.ndarray, shape (n_obs, n_snp)
        Simulated SNP genotype matrix coded 0/1/2
        (0 = homozygous reference, 1 = heterozygous, 2 = homozygous variant).
    y : np.ndarray, shape (n_obs,)
        Binary response vector (0 = control, 1 = case).
    beta0 : float
        Intercept used in the logistic model.
    beta : np.ndarray
        Regression coefficients for each interaction term.
    ia : list of str
        Human-readable descriptions of the interaction terms used.
    maf : np.ndarray, shape (n_snp,)
        Minor allele frequencies used for each SNP.
    prob : np.ndarray or None, shape (n_obs,)
        Predicted case probabilities from the logistic model.
        ``None`` only if a linear model was used (not currently exposed).
    """

    x: np.ndarray
    y: np.ndarray
    beta0: float
    beta: np.ndarray
    ia: list[str]
    maf: np.ndarray
    prob: np.ndarray | None


def simulate_snp_glm(
    n_obs: int = 1000,
    n_snp: int = 50,
    list_ia: list | None = None,
    list_snp: list | None = None,
    beta0: float = -0.5,
    beta: Union[float, list[float]] = 1.5,
    maf: Union[float, list[float], tuple[float, float]] = 0.25,
    sample_y: bool = True,
    p_cutoff: float = 0.5,
    random_state: int | None = None,
) -> SimSNPGlm:
    """Simulate SNP data with a logistic regression disease model.

    A Python port of ``simulateSNPglm`` from the R package *scrime*
    (Schwender, 2007).

    SNP genotypes are drawn independently under Hardy-Weinberg equilibrium.
    A binary response is generated from a logistic regression model whose
    predictors are Boolean interaction terms derived from ``list_ia`` /
    ``list_snp``.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_snp : int
        Number of SNPs to simulate.
    list_ia : list of array-like, optional
        Interaction specification.  Each element is an array of genotype
        condition codes (using scrime's 1/2/3 convention): ``1`` = homozygous
        reference, ``2`` = heterozygous, ``3`` = homozygous variant.  A
        **negative** value means *not* that genotype.  E.g. ``[-1, 1]``
        encodes ``(SNP != 1) & (SNP == 1)``.

        If ``list_ia`` is given but ``list_snp`` is ``None``, the first *n*
        SNPs are used in order (where *n* is the total number of terms across
        all interactions).  If both are ``None``, the default interactions from
        Nunkesser et al. (2007) are used.
    list_snp : list of array-like, optional
        SNP indices (1-based, to match scrime) for each interaction in
        ``list_ia``.  Each element must have the same length as the
        corresponding element in ``list_ia``.
    beta0 : float
        Intercept of the logistic regression model.
    beta : float or list of float
        Coefficient(s) for the interaction terms.  A single scalar is
        broadcast to all interactions.
    maf : float, (min, max) tuple, or array-like of length n_snp
        Minor allele frequency.

        * **scalar** – all SNPs share this MAF.
        * **length-2 tuple** – MAFs are drawn i.i.d. from
          Uniform(min, max).
        * **array of length n_snp** – per-SNP MAFs.
    sample_y : bool
        If ``True`` (default), response values are sampled from
        ``Bernoulli(prob)``.  If ``False``, ``y[i] = 1`` iff
        ``prob[i] > p_cutoff``.
    p_cutoff : float
        Probability cutoff used when ``sample_y=False``.
    random_state : int, optional
        Seed for reproducibility (passed to ``numpy.random.default_rng``).

    Returns
    -------
    SimSNPGlm
        Named result object.  ``x`` is coded 0/1/2 (numpy convention).

    Examples
    --------
    Reproduce the default interactions from Nunkesser et al. (2007):

    >>> sim = simulate_snp_glm(n_obs=1000, n_snp=50, random_state=42)
    >>> sim.x.shape
    (1000, 50)
    >>> sim.ia
    ['SNP6!=1 & SNP7==1', 'SNP3==1 & SNP9==1 & SNP10==1']

    Custom interactions with explicit SNP assignments:

    >>> list_ia  = [[-2, 1], [3]]
    >>> list_snp = [[4,  3], [5]]
    >>> sim2 = simulate_snp_glm(600, 25, list_ia, list_snp,
    ...                         maf=(0.1, 0.4), random_state=0)
    >>> sim2.x.shape
    (600, 25)
    """
    rng = np.random.default_rng(random_state)

    # --- MAF ---
    maf_arr = _parse_maf(maf, n_snp, rng)

    # --- Genotype matrix under Hardy-Weinberg (0/1/2 coded) ---
    x = _simulate_snps_hw(n_obs, n_snp, maf_arr, rng)

    # --- Default interactions (Nunkesser et al. 2007 / scrime default) ---
    if list_ia is None and list_snp is None:
        list_ia = [[-1, 1], [1, 1, 1]]
        list_snp = [[6, 7], [3, 9, 10]]
    elif list_ia is not None and list_snp is None:
        cursor = 0
        list_snp = []
        for ia_vec in list_ia:
            n_terms = len(ia_vec) if hasattr(ia_vec, "__len__") else 1
            list_snp.append(list(range(cursor + 1, cursor + n_terms + 1)))
            cursor += n_terms

    # scrime genotype codes are 1/2/3; x is 0/1/2 → shift for comparison
    x_scrime = x + 1  # view in scrime coding for predicate evaluation

    n_ia = len(list_ia)
    L = np.ones((n_obs, n_ia), dtype=float)
    ia_names: list[str] = []

    for k, (ia_vec, snp_vec) in enumerate(zip(list_ia, list_snp)):
        ia_vec = np.asarray(ia_vec, dtype=int)
        snp_vec = np.asarray(snp_vec, dtype=int)  # 1-based
        parts: list[str] = []
        for g, s in zip(ia_vec, snp_vec):
            col = x_scrime[:, int(s) - 1]
            if g > 0:
                L[:, k] *= (col == g).astype(float)
                parts.append(f"SNP{s}=={g}")
            else:
                L[:, k] *= (col != -g).astype(float)
                parts.append(f"SNP{s}!={-g}")
        ia_names.append(" & ".join(parts))

    # --- Regression coefficients ---
    beta_arr = np.broadcast_to(np.asarray(beta, dtype=float), (n_ia,)).copy()

    # --- Logistic model ---
    eta = beta0 + L @ beta_arr
    prob = 1.0 / (1.0 + np.exp(-eta))

    if sample_y:
        y = rng.binomial(1, prob).astype(np.int8)
    else:
        y = (prob > p_cutoff).astype(np.int8)

    return SimSNPGlm(
        x=x,
        y=y,
        beta0=beta0,
        beta=beta_arr,
        ia=ia_names,
        maf=maf_arr,
        prob=prob,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_maf(
    maf: Union[float, list, tuple],
    n_snp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    arr = np.asarray(maf, dtype=float)
    if arr.ndim == 0:
        return np.full(n_snp, float(arr))
    if arr.shape == (2,) and n_snp != 2:
        lo, hi = float(arr[0]), float(arr[1])
        return rng.uniform(lo, hi, size=n_snp)
    if arr.shape == (n_snp,):
        return arr
    raise ValueError(
        f"maf must be a scalar, a length-2 (min, max) tuple, or an array of "
        f"length n_snp={n_snp}. Got shape {arr.shape}."
    )


def _simulate_snps_hw(
    n_obs: int,
    n_snp: int,
    maf: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate SNP genotypes under Hardy-Weinberg equilibrium.

    Returns an (n_obs, n_snp) int8 matrix coded 0/1/2:
    0 = homozygous reference (AA), 1 = heterozygous (Aa),
    2 = homozygous variant (aa).
    """
    p = maf  # minor allele frequencies, shape (n_snp,)
    q = 1.0 - p
    # Hardy-Weinberg probabilities: [P(AA), P(Aa), P(aa)]
    hw_probs = np.column_stack([q**2, 2 * p * q, p**2])  # (n_snp, 3)

    x = np.empty((n_obs, n_snp), dtype=np.int8)
    for j in range(n_snp):
        x[:, j] = rng.choice(3, size=n_obs, p=hw_probs[j])
    return x
