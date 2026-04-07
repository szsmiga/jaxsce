import numpy as np

from jaxsce.densities.pyscf import _transform_dm_to_ao


def test_transform_dm_to_ao_restricted():
    mo = np.array([[1.0, 0.0], [0.2, 0.98]])
    dm = np.array([[2.0, 0.1], [0.1, 0.5]])

    out = _transform_dm_to_ao(dm, mo)
    exp = mo @ dm @ mo.T

    assert np.allclose(out, exp)


def test_transform_dm_to_ao_unrestricted_tuple_input():
    mo_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    mo_b = np.array([[0.9, 0.1], [0.1, 0.95]])
    dm_a = np.array([[1.0, 0.2], [0.2, 0.3]])
    dm_b = np.array([[0.7, 0.1], [0.1, 0.4]])

    out = _transform_dm_to_ao((dm_a, dm_b), (mo_a, mo_b))
    exp = np.stack([mo_a @ dm_a @ mo_a.T, mo_b @ dm_b @ mo_b.T], axis=0)

    assert out.shape == (2, 2, 2)
    assert np.allclose(out, exp)


def test_transform_dm_to_ao_unrestricted_dm_with_shared_mo_coeff():
    mo = np.array([[1.0, 0.0], [0.1, 0.95]])
    dm_a = np.array([[1.0, 0.0], [0.0, 0.2]])
    dm_b = np.array([[0.5, 0.1], [0.1, 0.3]])

    out = _transform_dm_to_ao(np.stack([dm_a, dm_b], axis=0), mo)
    exp = np.stack([mo @ dm_a @ mo.T, mo @ dm_b @ mo.T], axis=0)

    assert np.allclose(out, exp)
