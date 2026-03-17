import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jaxsce.coordinates_3d import get_coordinate_system
from jaxsce.integrate import sce_winf_prime_model
from jaxsce.optimize import AngularOptimizationResult


def _result_from_saved(path: str):
    npz = np.load(Path(path) / "1025.npz", allow_pickle=True)
    data = json.load(open(Path(path) / "1025.json", encoding="utf-8"))
    return SimpleNamespace(
        angles=npz["angles"],
        f=npz["f"],
        Ne=npz["Ne"],
        r=npz["r"],
        rho=npz["rho"],
        opt=SimpleNamespace(grid=data["opt"]["grid"], coordinates=get_coordinate_system("reduced")),
    )


def test_sce_winf_prime_model_smoke():
    res = AngularOptimizationResult.load("1025", "data/sqrt_r/10")

    val_reduced = sce_winf_prime_model(res, integrator="simpson")
    val_angular = sce_winf_prime_model(res, integrator="simpson", mode="angular")

    assert math.isfinite(val_reduced)
    assert val_reduced > 0.0
    assert val_reduced > val_angular


def test_sce_winf_prime_model_he_is_in_expected_range():
    res = _result_from_saved("data/He/aug-cc-pVQZ")
    val_he = sce_winf_prime_model(res, integrator="simpson")

    assert 0.5 < val_he < 0.7


def test_sce_winf_prime_model_mu_start_reduces_value():
    res = _result_from_saved("data/He/aug-cc-pVQZ")

    val_all = sce_winf_prime_model(res, integrator="simpson", mode="reduced", mu_start=0)
    val_mu4 = sce_winf_prime_model(res, integrator="simpson", mode="reduced", mu_start=3)

    assert val_all > 0.0
    assert val_mu4 >= 0.0
    assert val_mu4 <= val_all


def test_sce_winf_prime_model_linear_transform_differs():
    res = _result_from_saved("data/He/aug-cc-pVQZ")

    val_sqrt = sce_winf_prime_model(res, integrator="simpson", mode="reduced", eig_transform="sqrt")
    val_linear = sce_winf_prime_model(
        res, integrator="simpson", mode="reduced", eig_transform="linear"
    )

    assert val_linear > 0.0
    assert val_sqrt > 0.0
    assert val_linear != val_sqrt


def test_spherical_full_matches_reduced_block():
    res = _result_from_saved("data/He/aug-cc-pVQZ")

    val_spherical = sce_winf_prime_model(res, integrator="simpson", mode="spherical_full")
    val_reduced = sce_winf_prime_model(res, integrator="simpson", mode="reduced")

    assert abs(val_spherical - val_reduced) / val_reduced < 1e-8
