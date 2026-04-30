"""catgen.simulation – sub-package for SNP and categorical data simulation."""

from catgen.simulation.snp import (
    SimSNPGlm,
    SimSNPCovariateGlm,
    simulate_snp_glm,
    simulate_snp_glm_with_covariates,
)

__all__ = [
    "SimSNPGlm",
    "SimSNPCovariateGlm",
    "simulate_snp_glm",
    "simulate_snp_glm_with_covariates",
]
