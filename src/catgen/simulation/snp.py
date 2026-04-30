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


@dataclass
class SimSNPCovariateGlm(SimSNPGlm):
    """Result of :func:`simulate_snp_glm_with_covariates`."""

    covariates: np.ndarray
    covariate_beta: np.ndarray
    covariate_names: list[str]
    covariate_interaction_beta: np.ndarray
    covariate_interactions: list[str]


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
    list_ia : list, optional
        Interaction specification. Each element describes one model term and
        may be either a scalar genotype condition code or an array-like of
        multiple codes for an interaction term. Codes use scrime's 1/2/3
        convention: ``1`` = homozygous reference, ``2`` = heterozygous,
        ``3`` = homozygous variant. A **negative** value means *not* that
        genotype. E.g. ``[-1, 1]`` encodes ``(SNP != 1) & (SNP == 1)``.

        This mirrors scrime's mixed term style, so calls such as
        ``[-1, -1, -1, [-1, -1]]`` are accepted and interpreted as three main
        effects plus one interaction.

        If ``list_ia`` is given but ``list_snp`` is ``None``, the first *n*
        SNPs are used in order (where *n* is the total number of terms across
        all interactions).  If both are ``None``, the default interactions from
        Nunkesser et al. (2007) are used.
    list_snp : list, optional
        SNP indices (1-based, to match scrime) for each term in ``list_ia``.
        Each element may again be a scalar or an array-like and must have the
        same length as the corresponding element in ``list_ia``.
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

    list_ia, list_snp = _resolve_snp_term_spec(
        list_ia,
        list_snp,
        n_snp=n_snp,
        use_defaults=True,
    )

    # scrime genotype codes are 1/2/3; x is 0/1/2 → shift for comparison
    x_scrime = x + 1  # view in scrime coding for predicate evaluation

    L, ia_names = _build_snp_term_matrix(x_scrime, list_ia, list_snp)

    # --- Regression coefficients ---
    beta_arr = _broadcast_coefficients(beta, len(ia_names), "beta")

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


def simulate_snp_glm_with_covariates(
    n_obs: int = 1000,
    n_snp: int = 50,
    list_ia: list | None = None,
    list_snp: list | None = None,
    beta0: float = -0.5,
    beta: Union[float, list[float]] = 1.5,
    maf: Union[float, list[float], tuple[float, float]] = 0.25,
    covariates: np.ndarray | list | None = None,
    covariate_mean: Union[float, list[float], tuple[float, ...], None] = None,
    covariate_cov: Union[float, list[float], list[list[float]], np.ndarray, None] = None,
    covariate_beta: Union[float, list[float], None] = None,
    covariate_interaction_ia: list | None = None,
    covariate_interaction_snp: list | None = None,
    covariate_interaction_index: list[int] | tuple[int, ...] | np.ndarray | None = None,
    covariate_interaction_beta: Union[float, list[float], None] = None,
    sample_y: bool = True,
    p_cutoff: float = 0.5,
    random_state: int | None = None,
) -> SimSNPCovariateGlm:
    """Simulate SNP data with additional continuous covariates.

    This extends :func:`simulate_snp_glm` with sampled or user-provided
    continuous covariates and optional SNP-covariate interaction terms.
    Covariates are sampled from a multivariate normal distribution when
    ``covariates`` is not provided explicitly.
    """
    rng = np.random.default_rng(random_state)

    maf_arr = _parse_maf(maf, n_snp, rng)
    x = _simulate_snps_hw(n_obs, n_snp, maf_arr, rng)
    x_scrime = x + 1

    list_ia, list_snp = _resolve_snp_term_spec(
        list_ia,
        list_snp,
        n_snp=n_snp,
        use_defaults=False,
    )
    L, ia_names = _build_snp_term_matrix(x_scrime, list_ia, list_snp)
    beta_arr = _broadcast_coefficients(beta, len(ia_names), "beta")

    covariate_arr, covariate_names = _resolve_covariates(
        covariates=covariates,
        n_obs=n_obs,
        covariate_mean=covariate_mean,
        covariate_cov=covariate_cov,
        rng=rng,
    )

    eta = beta0 + L @ beta_arr

    covariate_beta_arr = _broadcast_coefficients(
        0.0 if covariate_beta is None else covariate_beta,
        covariate_arr.shape[1],
        "covariate_beta",
    )
    if covariate_beta_arr.size:
        eta += covariate_arr @ covariate_beta_arr

    covariate_interaction_beta_arr = np.empty(0, dtype=float)
    covariate_interactions: list[str] = []
    if covariate_interaction_ia is not None or covariate_interaction_index is not None:
        if covariate_arr.shape[1] == 0:
            raise ValueError(
                "Covariate interaction terms require at least one covariate."
            )
        if covariate_interaction_ia is None or covariate_interaction_index is None:
            raise ValueError(
                "covariate_interaction_ia and covariate_interaction_index must both be provided."
            )

        interaction_ia, interaction_snp = _resolve_snp_term_spec(
            covariate_interaction_ia,
            covariate_interaction_snp,
            n_snp=n_snp,
            use_defaults=False,
        )
        interaction_matrix, interaction_names = _build_snp_term_matrix(
            x_scrime,
            interaction_ia,
            interaction_snp,
        )
        covariate_index_arr = np.atleast_1d(
            np.asarray(covariate_interaction_index, dtype=int)
        )
        if covariate_index_arr.shape != (len(interaction_names),):
            raise ValueError(
                "covariate_interaction_index must contain one 1-based covariate index "
                "for each covariate interaction term."
            )
        if np.any(covariate_index_arr < 1) or np.any(covariate_index_arr > covariate_arr.shape[1]):
            raise ValueError(
                "covariate_interaction_index contains an out-of-range covariate index."
            )

        covariate_interaction_beta_arr = _broadcast_coefficients(
            0.0 if covariate_interaction_beta is None else covariate_interaction_beta,
            len(interaction_names),
            "covariate_interaction_beta",
        )
        interaction_terms = np.column_stack(
            [
                covariate_arr[:, covariate_index_arr[k] - 1] * interaction_matrix[:, k]
                for k in range(len(interaction_names))
            ]
        )
        eta += interaction_terms @ covariate_interaction_beta_arr
        covariate_interactions = [
            f"{covariate_names[covariate_index_arr[k] - 1]} * {interaction_names[k]}"
            for k in range(len(interaction_names))
        ]

    prob = 1.0 / (1.0 + np.exp(-eta))

    if sample_y:
        y = rng.binomial(1, prob).astype(np.int8)
    else:
        y = (prob > p_cutoff).astype(np.int8)

    return SimSNPCovariateGlm(
        x=x,
        y=y,
        beta0=beta0,
        beta=beta_arr,
        ia=ia_names,
        maf=maf_arr,
        prob=prob,
        covariates=covariate_arr,
        covariate_beta=covariate_beta_arr,
        covariate_names=covariate_names,
        covariate_interaction_beta=covariate_interaction_beta_arr,
        covariate_interactions=covariate_interactions,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_maf(
    maf: Union[float, list, tuple],
    n_snp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.isscalar(maf):
        return np.full(n_snp, float(maf))
    if isinstance(maf, tuple):
        if len(maf) != 2:
            raise ValueError(
                "maf tuple values must be given as (min, max). For per-SNP MAFs, "
                "pass a list or numpy array of length n_snp."
            )
        lo, hi = float(maf[0]), float(maf[1])
        return rng.uniform(lo, hi, size=n_snp)

    arr = np.asarray(maf, dtype=float)
    if arr.shape == (n_snp,):
        return arr
    raise ValueError(
        f"maf must be a scalar, a length-2 (min, max) tuple, or a list/array of "
        f"length n_snp={n_snp}. Got shape {arr.shape}."
    )


def _resolve_snp_term_spec(
    list_ia: list | None,
    list_snp: list | None,
    *,
    n_snp: int,
    use_defaults: bool = True,
) -> tuple[list, list]:
    uses_default_terms = False
    if list_ia is None and list_snp is None:
        if not use_defaults:
            return [], []
        uses_default_terms = True
        list_ia = [[-1, 1], [1, 1, 1]]
        list_snp = [[6, 7], [3, 9, 10]]
    elif list_ia is None:
        raise ValueError("list_snp requires a matching list_ia specification.")
    elif list_snp is None:
        cursor = 0
        list_snp = []
        for ia_vec in list_ia:
            n_terms = len(ia_vec) if hasattr(ia_vec, "__len__") else 1
            list_snp.append(list(range(cursor + 1, cursor + n_terms + 1)))
            cursor += n_terms

    list_ia, list_snp = _normalize_term_spec(list_ia, list_snp)
    _validate_snp_term_indices(list_snp, n_snp, uses_default_terms=uses_default_terms)
    return list_ia, list_snp


def _build_snp_term_matrix(
    x_scrime: np.ndarray,
    list_ia: list,
    list_snp: list,
) -> tuple[np.ndarray, list[str]]:
    n_obs = x_scrime.shape[0]
    n_terms = len(list_ia)
    if n_terms == 0:
        return np.empty((n_obs, 0), dtype=float), []

    term_matrix = np.ones((n_obs, n_terms), dtype=float)
    term_names: list[str] = []

    for k, (ia_vec, snp_vec) in enumerate(zip(list_ia, list_snp)):
        ia_vec = np.atleast_1d(np.asarray(ia_vec, dtype=int))
        snp_vec = np.atleast_1d(np.asarray(snp_vec, dtype=int))
        parts: list[str] = []
        for g, s in zip(ia_vec, snp_vec):
            col = x_scrime[:, int(s) - 1]
            if g > 0:
                term_matrix[:, k] *= (col == g).astype(float)
                parts.append(f"SNP{s}=={g}")
            else:
                term_matrix[:, k] *= (col != -g).astype(float)
                parts.append(f"SNP{s}!={-g}")
        term_names.append(" & ".join(parts))

    return term_matrix, term_names


def _normalize_term_spec(list_ia: list, list_snp: list) -> tuple[list, list]:
    if len(list_ia) != len(list_snp):
        raise ValueError("list_ia and list_snp must contain the same number of terms.")

    normalized_ia: list[np.ndarray] = []
    normalized_snp: list[np.ndarray] = []

    for ia_term, snp_term in zip(list_ia, list_snp):
        ia_arr = np.atleast_1d(np.asarray(ia_term, dtype=int))
        snp_arr = np.atleast_1d(np.asarray(snp_term, dtype=int))

        if ia_arr.shape != snp_arr.shape:
            raise ValueError(
                "Each list_ia term must have the same number of entries as the "
                "corresponding list_snp term. "
                f"Got shapes {ia_arr.shape} and {snp_arr.shape}."
            )

        normalized_ia.append(ia_arr)
        normalized_snp.append(snp_arr)

    return normalized_ia, normalized_snp


def _validate_snp_term_indices(
    list_snp: list[np.ndarray],
    n_snp: int,
    *,
    uses_default_terms: bool,
) -> None:
    if not list_snp:
        return

    max_index = max(int(np.max(snp_term)) for snp_term in list_snp)
    min_index = min(int(np.min(snp_term)) for snp_term in list_snp)

    if min_index < 1:
        raise ValueError("SNP indices must be 1-based and greater than or equal to 1.")
    if max_index <= n_snp:
        return
    if uses_default_terms:
        raise ValueError(
            "The default scrime interaction terms require n_snp >= 10. "
            "Provide a larger n_snp or pass explicit list_ia/list_snp values."
        )
    raise ValueError(
        f"SNP term specification references SNP{max_index}, but n_snp={n_snp}."
    )


def _resolve_covariates(
    covariates: np.ndarray | list | None,
    n_obs: int,
    covariate_mean: Union[float, list[float], tuple[float, ...], None],
    covariate_cov: Union[float, list[float], list[list[float]], np.ndarray, None],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    if covariates is not None:
        covariate_arr = np.asarray(covariates, dtype=float)
        if covariate_arr.ndim == 1:
            covariate_arr = covariate_arr[:, np.newaxis]
        if covariate_arr.ndim != 2:
            raise ValueError("covariates must be a 1D or 2D array-like object.")
        if covariate_arr.shape[0] != n_obs:
            raise ValueError(
                f"covariates must have n_obs={n_obs} rows. Got {covariate_arr.shape[0]}."
            )
        return covariate_arr, _covariate_names(covariate_arr.shape[1])

    if covariate_mean is None:
        return np.empty((n_obs, 0), dtype=float), []

    mean_arr = np.atleast_1d(np.asarray(covariate_mean, dtype=float))
    cov_arr = _parse_covariate_cov(covariate_cov, mean_arr.shape[0])
    sampled = rng.multivariate_normal(mean_arr, cov_arr, size=n_obs)
    if sampled.ndim == 1:
        sampled = sampled[:, np.newaxis]
    return sampled, _covariate_names(sampled.shape[1])


def _parse_covariate_cov(
    covariate_cov: Union[float, list[float], list[list[float]], np.ndarray, None],
    n_covariates: int,
) -> np.ndarray:
    if covariate_cov is None:
        return np.eye(n_covariates, dtype=float)

    cov_arr = np.asarray(covariate_cov, dtype=float)
    if cov_arr.ndim == 0:
        return np.eye(n_covariates, dtype=float) * float(cov_arr)
    if cov_arr.shape == (n_covariates,):
        return np.diag(cov_arr)
    if cov_arr.shape == (n_covariates, n_covariates):
        return cov_arr
    raise ValueError(
        "covariate_cov must be a scalar variance, a per-covariate variance vector, "
        "or a full covariance matrix matching covariate_mean."
    )


def _broadcast_coefficients(
    coefficients: Union[float, list[float], np.ndarray],
    n_terms: int,
    name: str,
) -> np.ndarray:
    if n_terms == 0:
        return np.empty(0, dtype=float)
    try:
        return np.broadcast_to(np.asarray(coefficients, dtype=float), (n_terms,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {n_terms} coefficient(s).") from exc


def _covariate_names(n_covariates: int) -> list[str]:
    return [f"E{i}" for i in range(1, n_covariates + 1)]


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
