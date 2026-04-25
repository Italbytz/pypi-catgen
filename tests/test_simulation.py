"""Tests for catgen.simulation.snp – simulate_snp_glm."""

import numpy as np
import pytest

from catgen import SimSNPGlm, simulate_snp_glm


# ---------------------------------------------------------------------------
# Basic shape and value-range tests
# ---------------------------------------------------------------------------


def test_default_shape():
    result = simulate_snp_glm(n_obs=200, n_snp=15, random_state=42)
    assert result.x.shape == (200, 15)
    assert result.y.shape == (200,)


def test_genotype_values():
    result = simulate_snp_glm(n_obs=500, n_snp=20, random_state=0)
    assert np.all(np.isin(result.x, [0, 1, 2])), "x must be coded 0/1/2"


def test_response_binary():
    result = simulate_snp_glm(n_obs=500, n_snp=20, random_state=0)
    assert np.all(np.isin(result.y, [0, 1])), "y must be binary"


# ---------------------------------------------------------------------------
# MAF variants
# ---------------------------------------------------------------------------


def test_maf_scalar():
    result = simulate_snp_glm(n_obs=500, n_snp=10, maf=0.3, random_state=1)
    assert result.maf.shape == (10,)
    np.testing.assert_array_equal(result.maf, 0.3)


def test_maf_range_tuple():
    result = simulate_snp_glm(n_obs=500, n_snp=10, maf=(0.1, 0.4), random_state=1)
    assert result.maf.shape == (10,)
    assert np.all(result.maf >= 0.1)
    assert np.all(result.maf <= 0.4)


def test_maf_per_snp_array():
    maf_vec = np.linspace(0.05, 0.45, 10)
    result = simulate_snp_glm(n_obs=300, n_snp=10, maf=maf_vec, random_state=2)
    np.testing.assert_array_equal(result.maf, maf_vec)


def test_maf_invalid_raises():
    with pytest.raises(ValueError, match="n_snp=5"):
        simulate_snp_glm(n_obs=100, n_snp=5, maf=[0.1, 0.2, 0.3], random_state=0)


# ---------------------------------------------------------------------------
# Hardy-Weinberg equilibrium (large sample)
# ---------------------------------------------------------------------------


def test_hw_empirical_frequencies():
    """Empirical MAF should be close to specified MAF for large n_obs."""
    maf_target = 0.3
    result = simulate_snp_glm(
        n_obs=20_000,
        n_snp=1,
        maf=maf_target,
        list_ia=[[1]],
        list_snp=[[1]],
        random_state=42,
    )
    x = result.x[:, 0]
    empirical_maf = (np.sum(x == 2) * 2 + np.sum(x == 1)) / (2 * len(x))
    assert abs(empirical_maf - maf_target) < 0.02, (
        f"Empirical MAF {empirical_maf:.4f} deviates from target {maf_target}"
    )


# ---------------------------------------------------------------------------
# Interaction specification
# ---------------------------------------------------------------------------


def test_custom_list_ia_and_snp():
    list_ia = [[-2, 1], [3]]
    list_snp = [[4, 3], [5]]
    result = simulate_snp_glm(
        n_obs=300, n_snp=10, list_ia=list_ia, list_snp=list_snp, random_state=3
    )
    assert result.x.shape == (300, 10)
    assert len(result.ia) == 2
    assert "SNP4!=2" in result.ia[0]
    assert "SNP3==1" in result.ia[0]
    assert "SNP5==3" in result.ia[1]


def test_list_ia_without_list_snp():
    """list_ia without list_snp should use first n SNPs in order."""
    result = simulate_snp_glm(
        n_obs=200, n_snp=10, list_ia=[[1, 2], [3]], random_state=4
    )
    assert result.x.shape == (200, 10)
    assert len(result.ia) == 2
    # First interaction uses SNP1 and SNP2
    assert "SNP1==1" in result.ia[0]
    assert "SNP2==2" in result.ia[0]
    # Second uses SNP3
    assert "SNP3==3" in result.ia[1]


def test_default_ia_names():
    """Default interactions must match Nunkesser et al. 2007 / scrime default."""
    result = simulate_snp_glm(n_obs=100, n_snp=15, random_state=0)
    assert result.ia[0] == "SNP6!=1 & SNP7==1"
    assert result.ia[1] == "SNP3==1 & SNP9==1 & SNP10==1"


# ---------------------------------------------------------------------------
# Response sampling
# ---------------------------------------------------------------------------


def test_sample_y_false():
    result = simulate_snp_glm(
        n_obs=500, n_snp=15, sample_y=False, p_cutoff=0.5, random_state=5
    )
    assert np.all(np.isin(result.y, [0, 1]))
    # Deterministic: y should equal (prob > 0.5)
    np.testing.assert_array_equal(result.y, (result.prob > 0.5).astype(np.int8))


def test_prob_in_unit_interval():
    result = simulate_snp_glm(n_obs=300, n_snp=15, random_state=6)
    assert result.prob is not None
    assert np.all((result.prob >= 0.0) & (result.prob <= 1.0))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility():
    r1 = simulate_snp_glm(n_obs=100, n_snp=10, random_state=99)
    r2 = simulate_snp_glm(n_obs=100, n_snp=10, random_state=99)
    np.testing.assert_array_equal(r1.x, r2.x)
    np.testing.assert_array_equal(r1.y, r2.y)
    np.testing.assert_array_almost_equal(r1.prob, r2.prob)


def test_different_seeds_differ():
    r1 = simulate_snp_glm(n_obs=200, n_snp=15, random_state=1)
    r2 = simulate_snp_glm(n_obs=200, n_snp=15, random_state=2)
    assert not np.array_equal(r1.x, r2.x)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_snp_glm(n_obs=50, n_snp=15, random_state=0)
    assert isinstance(result, SimSNPGlm)
    assert isinstance(result.x, np.ndarray)
    assert isinstance(result.y, np.ndarray)
    assert isinstance(result.beta, np.ndarray)
    assert isinstance(result.maf, np.ndarray)
    assert isinstance(result.ia, list)
