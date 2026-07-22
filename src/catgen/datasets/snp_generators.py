"""SNP dataset generators – unified (X, y) interface.

Provides convenience wrappers around the low-level SNP simulation functions
that return (X, y) tuples matching the standard dataset generator API.

For full control over model parameters, simulation details, and access to
the complete SimSNPGlm dataclass, use :func:`catgen.simulation.snp.simulate_snp_glm`
and related functions directly.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from catgen.simulation.snp import (
    simulate_snp_glm,
    simulate_snp_glm_with_covariates,
)


def generate_snp_glm_dataset(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SNP data with logistic regression disease model.

    Wrapper around :func:`catgen.simulation.snp.simulate_snp_glm` that
    returns ``(X, y)`` for consistency with other dataset generators.

    For access to simulation details (parameters, interaction descriptions),
    use :func:`catgen.simulation.snp.simulate_snp_glm` directly, which returns
    a :class:`catgen.simulation.snp.SimSNPGlm` dataclass.

    Parameters
    ----------
    n_obs : int, default=1000
        Number of observations to simulate.
    n_snp : int, default=50
        Number of SNPs.
    list_ia : list | None, default=None
        Interaction term specification (default: Nunkesser et al. 2007 model).
    list_snp : list | None, default=None
        SNP indices for each interaction term.
    beta0 : float, default=-0.5
        Intercept for the logistic model.
    beta : float | list[float], default=1.5
        Coefficient(s) for interaction terms.
    maf : float | list[float] | tuple[float, float], default=0.25
        Minor allele frequency per SNP or range.
    sample_y : bool, default=True
        If True, sample y from the logistic model. If False, use probabilities.
    p_cutoff : float, default=0.5
        Cutoff for converting probabilities to binary class.
    random_state : int | None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_obs, n_snp)
        Genotype matrix (0/1/2 coding).
    y : np.ndarray, shape (n_obs,)
        Binary response vector.

    Examples
    --------
    >>> X, y = generate_snp_glm_dataset(n_obs=200, n_snp=25, random_state=42)
    >>> X.shape
    (200, 25)
    >>> y.shape
    (200,)
    """
    sim = simulate_snp_glm(
        n_obs=n_obs,
        n_snp=n_snp,
        list_ia=list_ia,
        list_snp=list_snp,
        beta0=beta0,
        beta=beta,
        maf=maf,
        sample_y=sample_y,
        p_cutoff=p_cutoff,
        random_state=random_state,
    )
    return sim.x, sim.y


def generate_snp_glm_with_covariates_dataset(
    n_obs: int = 1000,
    n_snp: int = 50,
    n_covariates: int = 2,
    list_ia: list | None = None,
    list_snp: list | None = None,
    beta0: float = -0.5,
    beta: Union[float, list[float]] = 1.5,
    covariate_beta: Union[float, list[float]] = 0.5,
    maf: Union[float, list[float], tuple[float, float]] = 0.25,
    covariate_interaction_snp: list[int] | None = None,
    covariate_interaction_beta: Union[float, list[float]] | None = None,
    sample_y: bool = True,
    p_cutoff: float = 0.5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SNP data with covariates and SNP-covariate interactions.

    Wrapper around :func:`catgen.simulation.snp.simulate_snp_glm_with_covariates`
    that returns ``(X, y)`` for consistency with other dataset generators.

    For access to covariate details, interaction descriptions, and full
    simulation parameters, use :func:`catgen.simulation.snp.simulate_snp_glm_with_covariates`
    directly, which returns a :class:`catgen.simulation.snp.SimSNPCovariateGlm` dataclass.

    Parameters
    ----------
    n_obs : int, default=1000
        Number of observations.
    n_snp : int, default=50
        Number of SNPs.
    n_covariates : int, default=2
        Number of continuous covariates.
    list_ia : list | None, default=None
        SNP interaction specification.
    list_snp : list | None, default=None
        SNP indices for interactions.
    beta0 : float, default=-0.5
        Intercept for the logistic model.
    beta : float | list[float], default=1.5
        Coefficients for SNP interaction terms.
    covariate_beta : float | list[float], default=0.5
        Coefficients for continuous covariates.
    maf : float | list[float] | tuple[float, float], default=0.25
        Minor allele frequencies.
    covariate_interaction_snp : list[int] | None, default=None
        SNP indices to interact with covariates.
    covariate_interaction_beta : float | list[float] | None, default=None
        Coefficients for SNP-covariate interactions.
    sample_y : bool, default=True
        If True, sample y; otherwise use probabilities.
    p_cutoff : float, default=0.5
        Probability cutoff for classification.
    random_state : int | None, default=None
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n_obs, n_snp + n_covariates)
        Combined feature matrix (SNPs + covariates).
    y : np.ndarray, shape (n_obs,)
        Binary response vector.

    Examples
    --------
    >>> X, y = generate_snp_glm_with_covariates_dataset(
    ...     n_obs=200, n_snp=25, n_covariates=3, random_state=42
    ... )
    >>> X.shape
    (200, 28)
    """
    sim = simulate_snp_glm_with_covariates(
        n_obs=n_obs,
        n_snp=n_snp,
        n_covariates=n_covariates,
        list_ia=list_ia,
        list_snp=list_snp,
        beta0=beta0,
        beta=beta,
        covariate_beta=covariate_beta,
        maf=maf,
        covariate_interaction_snp=covariate_interaction_snp,
        covariate_interaction_beta=covariate_interaction_beta,
        sample_y=sample_y,
        p_cutoff=p_cutoff,
        random_state=random_state,
    )
    # Combine SNPs and covariates into single feature matrix
    X_combined = np.hstack([sim.x, sim.covariates])
    return X_combined, sim.y
