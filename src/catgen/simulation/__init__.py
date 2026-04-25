"""catgen.simulation – sub-package for SNP and categorical data simulation."""

from catgen.simulation.snp import SimSNPGlm, simulate_snp_glm

__all__ = [
    "SimSNPGlm",
    "simulate_snp_glm",
]
