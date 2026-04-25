"""catgen – Simulation and analysis of high-dimensional categorical data.

Primary focus: SNP genotype simulation under Hardy-Weinberg equilibrium
with logistic regression disease models, porting the R scrime package.
"""

from catgen.simulation.snp import SimSNPGlm, simulate_snp_glm

__all__ = [
    "SimSNPGlm",
    "simulate_snp_glm",
]
