"""Integration to get the Vee from the Vee(r) or Vee(Ne)"""

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
import jax.numpy as jnp
from jax import grad, jacfwd
import scipy.integrate
import scipy.interpolate

from .coordinates_3d import Vee_cartesian
from .optimize import AngularOptimizationResult


def integrand_r(r: np.ndarray, rho: np.ndarray, integ: np.ndarray) -> np.ndarray:
    """
    Integrand for r grid, that is 4*pi*r**2*rho(r)*Vee(r)

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.

    Returns
    -------
    integrand: np.ndarray (N_grid,)
        Integrand with 4*pi*r**2*rho.
    """
    return 4 * np.pi * r**2 * rho * integ


def dintegrand_r(
    r: np.ndarray, rho: np.ndarray, drho: np.ndarray, integ: np.ndarray, dinteg: np.ndarray
) -> np.ndarray:
    """
    Derivative of integrand for r grid.

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    drho : np.ndarray (N_grid,)
        Derivative of density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.
    dinteg : np.ndarray (N_grid,)
        Derivative of integrand without 4*pi*r**2*rho.

    Returns
    -------
    dintegrand: np.ndarray (N_grid,)
        Derivative of integrand with 4*pi*r**2*rho.
    """
    return 4 * np.pi * r**2 * (drho * integ + rho * dinteg) + 8 * np.pi * r * rho * integ


def d2integrand_r(
    r: np.ndarray,
    rho: np.ndarray,
    drho: np.ndarray,
    d2rho: np.ndarray,
    integ: np.ndarray,
    dinteg: np.ndarray,
    d2integ: np.ndarray,
) -> np.ndarray:
    """
    Second derivative of integrand for r grid

    Params
    ------
    r : np.ndarray (N_grid,)
        Gridpoints.
    rho : np.ndarray (N_grid,)
        Density on grid.
    drho : np.ndarray (N_grid,)
        Derivative of density on grid.
    d2rho : np.ndarray (N_grid,)
        Second derivative of density on grid.
    integ : np.ndarray (N_grid,)
        Integrand without 4*pi*r**2*rho.
    dinteg : np.ndarray (N_grid,)
        Derivative of integrand without 4*pi*r**2*rho.
    d2integ : np.ndarray (N_grid,)
        Second derivative of integrand without 4*pi*r**2*rho.
    Returns
    -------
    d2integrand: np.ndarray (N_grid,)
        Second derivative of integrand with 4*pi*r**2*rho.
    """
    return (
        4 * np.pi * r**2 * (d2rho * integ + 2 * drho * dinteg + rho * d2integ)
        + 16 * np.pi * r * (drho * integ + rho * dinteg)
        + 8 * np.pi * rho * integ
    )


# Dictionary of the scipy interpolators
scipy_interpolators = {
    "cubic": scipy.interpolate.CubicSpline,
    "akima": scipy.interpolate.Akima1DInterpolator,
    "pchip": scipy.interpolate.PchipInterpolator,
}

# Dictionary of the scipy integrators
scipy_integrators = {
    "trapezoid": scipy.integrate.trapezoid,
    "simpson": scipy.integrate.simpson,
    "romb": scipy.integrate.romb,
}


def _mode_sum_from_eigvals(
    eigvals: np.ndarray,
    min_eig: float = 0.0,
    mu_start: int = 0,
    transform: str = "sqrt",
) -> float:
    r"""Compute :math:`\sum_{\mu=\mu_0}^{M} f(\lambda_\mu)` from Hessian eigenvalues."""

    eigvals = np.asarray(eigvals)
    if mu_start < 0:
        raise ValueError(f"mu_start must be non-negative, got {mu_start}")
    if mu_start >= eigvals.size:
        return 0.0
    eigvals = np.clip(eigvals[mu_start:], min_eig, None)

    if transform == "sqrt":
        return float(np.sum(np.sqrt(eigvals)))
    if transform == "linear":
        return float(np.sum(eigvals))
    raise ValueError(f"Unknown transform {transform}")


def _hessian_eigvals(
    coordinates,
    angles: np.ndarray,
    f: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build eigenvalues for the requested Hessian model."""

    if mode == "angular":
        hessian = np.asarray(coordinates.Vee_hessian_angles(angles, f))
    elif mode == "reduced":
        h_aa = np.asarray(coordinates.Vee_hessian_angles(angles, f))
        h_rr = np.asarray(coordinates.Vee_hessian_radials(angles, f))
        h_ar = np.asarray(coordinates.Vee_hessian_angles_radials(angles, f))
        hessian = np.block([[h_aa, h_ar], [h_ar.T, h_rr]])
    elif mode == "spherical_full":
        x0 = np.concatenate((np.asarray(f), np.asarray(angles)))
        Nel = f.shape[0]

        def vee_spherical(x):
            return coordinates.Vee(x[Nel:], x[:Nel])

        hessian = np.asarray(jacfwd(grad(vee_spherical))(x0))
    elif mode == "cartesian":
        coords = np.asarray(coordinates.cartesian_coordinates(jnp.asarray(f), jnp.asarray(angles)))
        Nel = coords.shape[0]

        def vee_flat(x_flat):
            return Vee_cartesian(x_flat.reshape(Nel, 3))

        hessian = np.asarray(jacfwd(grad(vee_flat))(coords.reshape(-1)))
    else:
        raise ValueError(f"Unknown mode {mode}")

    return np.linalg.eigvalsh(hessian)


def sce_winf_prime_model(
    res: AngularOptimizationResult,
    *,
    min_eig: float = 0.0,
    integrator: str = "simpson",
    mode: str = "spherical_full",
    mu_start: int = 0,
    eig_transform: str = "sqrt",
) -> float:
    r"""
    Compute an SCE :math:`W_\infty'` model from Eq. 81-inspired local frequencies.

    Parameters
    ----------
    res : AngularOptimizationResult
        Result object containing the optimized angles and co-motion functions.
    min_eig : float, optional
        Lower clipping value for Hessian eigenvalues before taking square roots.
    integrator : str, optional
        Numerical quadrature method, either ``"trapezoid"`` or ``"simpson"``.
    mode : str, optional
        ``"spherical_full"`` (default) computes the Hessian of ``Vee`` with
        respect to all spherical variables used by the coordinate system
        (all radial + all active angular coordinates).
        ``"reduced"`` uses the equivalent block assembly from angle-angle,
        angle-radial and radial-radial derivatives.
        ``"angular"`` reproduces the angular-only approximation.
        ``"cartesian"`` uses the full 3N Cartesian Hessian of ``Vee``.
    mu_start : int, optional
        Index of the first mode to include in the local sum. For the formal
        expression :math:`\sum_{\mu=4}^{3N}` this corresponds to ``mu_start=3``
        (0-indexed).
    eig_transform : str, optional
        Mode contribution transform. ``"sqrt"`` uses ``sum(sqrt(eigvals))``
        (current physical model). ``"linear"`` uses ``sum(eigvals)`` for
        sensitivity checks.

    Returns
    -------
    float
        Model value for :math:`W_\infty'` on the code's working shell grid.
    """
    if isinstance(res, AngularOptimizationResult):
        res = AngularOptimizationResult.to_numpy(res)

    if hasattr(res.opt, "coordinates"):
        coordinates = res.opt.coordinates
    else:
        raise ValueError("SCE W'_inf model requires a coordinate system in the result object.")

    N_grid_end = res.angles.shape[0]
    omega_sums = np.zeros(N_grid_end)
    for idx in range(N_grid_end):
        eigvals = _hessian_eigvals(coordinates, res.angles[idx], res.f[idx], mode=mode)
        omega_sums[idx] = _mode_sum_from_eigvals(
            eigvals, min_eig=min_eig, mu_start=mu_start, transform=eig_transform
        )

    if res.opt.grid == "r":
        x = res.r[:N_grid_end]
        y = 4 * np.pi * x**2 * res.rho[:N_grid_end] * omega_sums
    elif res.opt.grid == "Ne":
        x = res.Ne[:N_grid_end]
        y = omega_sums
    else:
        raise ValueError(f"Unknown grid type {res.opt.grid}")

    if integrator == "trapezoid":
        integral = scipy.integrate.trapezoid(y, x)
    elif integrator == "simpson":
        integral = scipy.integrate.simpson(y, x)
    else:
        raise ValueError(f"Unknown integrator {integrator}")

    # The current optimization/integration pipeline works on one shell in Ne-space
    # (Ne in [1, 2]); matching the repository's Winf convention this gives
    # the correct normalization without an additional 1/N factor.
    return float(0.5 * integral)


class VeeIntegration:
    """Class for integration of Vee(r) or Vee(Ne) to get Vee"""

    def __init__(self, res: AngularOptimizationResult):
        self.res: AngularOptimizationResult = AngularOptimizationResult.to_numpy(res)

        # Get some things we often need from the result
        self.N_grid: int = self.res.f.shape[0]
        Nel: int = self.res.f.shape[1]
        self.even: bool = Nel % 2 == 0
        self.N_grid_end: int = self.N_grid - 1 if self.even else self.N_grid

        # Get the subtraction for regularization of the integral with 1/f_N-1(r)

        self.f_reg = (Nel - 1) / self.res.f[:, Nel - 2]
        self.df_reg = (
            -(Nel - 1)
            * self.res.df[: self.N_grid_end, Nel - 2]
            / (self.res.f[: self.N_grid_end, Nel - 2]) ** 2
        )
        self.d2f_reg = (Nel - 1) * (
            2
            * self.res.df[: self.N_grid_end, Nel - 2] ** 2
            / (self.res.f[: self.N_grid_end, Nel - 2] ** 3)
            - self.res.d2f[: self.N_grid_end, Nel - 2]
            / (self.res.f[: self.N_grid_end, Nel - 2] ** 2)
        )

        # Get the analytical integral to add back later
        self.reg_integral = (Nel - 1) * self.res.coNe[Nel - 1]

        # Get grid, integrand and derivatives
        # Regardless of grid we use x and dx for the grid
        # and y, dy and d2y for the integrand and its derivatives
        # The y_reg, dy_reg and d2y_reg are the same but with the -1/f_N-1(r) subtraction
        if self.res.opt.grid == "r":
            self.x = self.res.r
            self.dx = self.res.r[1] - self.res.r[0]

            # Transform to float to prevent bpoly from complaining
            self.a1 = float(self.res.r[0])
            self.a2 = float(self.res.r[-1])

            self.y = integrand_r(self.res.r, self.res.rho, self.res.Vee)
            self.dy = dintegrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.Vee[: self.N_grid_end],
                self.res.dVee,
            )
            self.d2y = d2integrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.d2rho[: self.N_grid_end],
                self.res.Vee[: self.N_grid_end],
                self.res.dVee,
                self.res.d2Vee,
            )

            self.f_reg_sph = integrand_r(self.res.r, self.res.rho, self.f_reg)
            self.df_reg_sph = dintegrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.f_reg[: self.N_grid_end],
                self.df_reg,
            )
            self.d2f_reg_sph = d2integrand_r(
                self.res.r[: self.N_grid_end],
                self.res.rho[: self.N_grid_end],
                self.res.drho[: self.N_grid_end],
                self.res.d2rho[: self.N_grid_end],
                self.f_reg[: self.N_grid_end],
                self.df_reg,
                self.d2f_reg,
            )
            self.y_reg = self.y - self.f_reg_sph
            if self.even:
                self.y_reg[self.N_grid_end] = self.y[self.N_grid_end]
            self.dy_reg = self.dy - self.df_reg_sph
            self.d2y_reg = self.d2y - self.d2f_reg_sph

        elif self.res.opt.grid == "Ne":
            self.x = self.res.Ne
            self.dx = self.res.Ne[1] - self.res.Ne[0]
            self.a1 = 1.0
            self.a2 = 2.0

            self.y = self.res.Vee
            self.dy = self.res.dVee
            self.d2y = self.res.d2Vee

            self.y_reg = self.y - self.f_reg
            if self.even:
                self.y_reg[self.N_grid_end] = self.y[self.N_grid_end]
            self.dy_reg = self.dy - self.df_reg
            self.d2y_reg = self.d2y - self.d2f_reg
        else:
            raise ValueError(f"Unknown grid type {self.res.grid}")

        # Initialize empty dictionary for interpolators
        self.interpolators: Dict[
            str, List[Union[scipy.interpolate.BPoly, scipy.interpolate.PPoly]]
        ] = {}

    def scipy_interpolator(self, name: str, reg: bool = False, sub_div: int = 1):
        """
        Add a scipy interpolator to the dictionary of interpolators.

        Parameters
        ----------
        name : str
            Name of the interpolator. Must be one of "linear", "cubic", "akima" or "pchip".
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        if name in scipy_interpolators:
            if name not in self.interpolators:
                self.interpolators[name] = [{}, {}]
            self.interpolators[name][int(reg)][sub_div] = scipy_interpolators[name](
                np.copy(self.x[::sub_div]),
                np.copy(self.y_reg[::sub_div]) if reg else np.copy(self.y[::sub_div]),
            )
        else:
            raise ValueError(f"Unknown interpolator {name}")

    def bpoly_0(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we only include the integrand and no derivatives.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_0"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]

        if reg:
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), np.copy(self.y_reg[::sub_div]).reshape(-1, 1)
            )
        else:
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), np.copy(self.y[::sub_div]).reshape(-1, 1)
            )

    def bpoly_1(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we only include the integrand and the first derivative.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_1"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]
        if reg:
            fold_y_dy_reg = np.stack(
                (self.y_reg[: self.N_grid_end : sub_div], self.dy_reg[::sub_div]), axis=1
            ).tolist()
            if self.even:
                fold_y_dy_reg.append([self.y_reg[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_reg
            )
        else:
            fold_y_dy = np.stack(
                (self.y[: self.N_grid_end : sub_div], self.dy[::sub_div]), axis=1
            ).tolist()
            if self.even:
                fold_y_dy.append([self.y[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy
            )

    def bpoly_2(self, reg: bool = False, sub_div: int = 1):
        """
        Add a BPoly interpolator to the dictionary of interpolators,
        where we include the integrand and the first and second derivative.

        Parameters
        ----------
        reg : bool, optional
            Whether to use the regularization subtraction, by default False
        sub_div : int, optional
            Subdivision of the grid, by default 1, must be a divisor of the grid size-1
        """
        name = "bpoly_2"
        if name not in self.interpolators:
            self.interpolators[name] = [{}, {}]
        if reg:
            fold_y_dy_d2y_reg = np.stack(
                (
                    self.y_reg[: self.N_grid_end : sub_div],
                    self.dy_reg[::sub_div],
                    self.d2y_reg[::sub_div],
                ),
                axis=1,
            ).tolist()
            if self.even:
                fold_y_dy_d2y_reg.append([self.y_reg[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_d2y_reg
            )
        else:
            fold_y_dy_d2y = np.stack(
                (self.y[: self.N_grid_end : sub_div], self.dy[::sub_div], self.d2y[::sub_div]),
                axis=1,
            ).tolist()
            if self.even:
                fold_y_dy_d2y.append([self.y[-1]])
            self.interpolators[name][int(reg)][sub_div] = scipy.interpolate.BPoly.from_derivatives(
                np.copy(self.x[::sub_div]), fold_y_dy_d2y
            )

    def interpolate_all(self, max_sub_divs: int = 2):
        """Interpolate all the interpolators for all the possible combinations
        of regularization and subdivision.

        Parameters
        ----------
        max_sub_divs : int
            Maximum subdivision of the grid. 0 means no subdivision, 1 means once, etc.
            By default 2
        """
        sub_divs = (2 ** np.arange(0, max_sub_divs)).tolist()

        # Loop over all the possible combinations of regularization and subdivision
        for reg in [True, False]:
            for sub_div in sub_divs:
                # Initialize the different possible interpolators
                self.bpoly_0(reg, sub_div)
                self.bpoly_1(reg, sub_div)
                self.bpoly_2(reg, sub_div)
                for name in scipy_interpolators:
                    self.scipy_interpolator(name, reg, sub_div)

    def integrate(
        self, max_sub_divs: int = 2
    ) -> Tuple[Dict[str, List[Dict[int, float]]], Dict[str, List[float]]]:
        """
        Integrate all the interpolators for all the possible combinations of
        regularization and subdivision.
        Also compute the integrals using scipy.integrate instead.

        Parameters
        ----------
        max_sub_divs : int
            Maximum subdivision of the grid. 0 means no subdivision, 1 means once, etc.
            By default 2

        Returns
        -------
        integrals : Dict[str, List[Dict[int, float]]]
            Dictionary of the integrals for all the possible combinations
            of regularization, subdivision and integration method.
        """
        integrals: Dict[str, List[Dict[int, float]]] = {}
        for name, reg_list in self.interpolators.items():
            if name not in integrals:
                integrals[name] = [{}, {}]
            for reg in [True, False]:
                if self.interpolators[name][int(reg)] == {}:
                    continue
                for sub_div, interpolation in reg_list[int(reg)].items():
                    if name in scipy_interpolators:
                        integrals[name][int(reg)][sub_div] = float(
                            interpolation.integrate(self.a1, self.a2)
                        )
                    else:
                        integrals[name][int(reg)][sub_div] = interpolation.integrate(
                            self.a1, self.a2
                        )

        integrals["trapezoid"] = [{}, {}]
        integrals["simpson"] = [{}, {}]
        integrals["romb"] = [{}, {}]

        for reg in [True, False]:
            for sub_div in (2 ** np.arange(0, max_sub_divs)).tolist():
                integrals["trapezoid"][int(reg)][sub_div] = scipy.integrate.trapezoid(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.x[::sub_div]
                )
                integrals["simpson"][int(reg)][sub_div] = scipy.integrate.simpson(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.x[::sub_div]
                )
                integrals["romb"][int(reg)][sub_div] = scipy.integrate.romb(
                    self.y_reg[::sub_div] if reg else self.y[::sub_div], self.dx * sub_div
                )

        for name in integrals:
            for sub_div in integrals[name][1]:
                integrals[name][1][sub_div] += self.reg_integral
        if max_sub_divs > 1:
            extrapolated_integrals = self.richardson_extrapolation(integrals)
        else:
            extrapolated_integrals = None
        return integrals, extrapolated_integrals

    def sub_divs_to_grid_points(self, max_sub_div: int = 1):
        """
        Compute the number of grid points for each subdivision.

        Parameters
        ----------
        max_sub_div : int, optional
            Maximum subdivision of the grid, by default 1, must be a divisor of the grid size-1

        Returns
        -------
        grid_points : Dict[int, int]
            Dictionary of the number of grid points for each subdivision.
        """
        grid_points: Dict[int, int] = {}
        for sub_div in 2 ** np.arange(0, max_sub_div):
            grid_points[sub_div] = (self.N_grid - 1) // sub_div + 1
        return grid_points

    def richardson_extrapolation(
        self, integrals: Dict[str, List[Dict[int, float]]]
    ) -> Dict[str, List[float]]:
        """
        Compute the Richardson extrapolation of the integrals.

        Parameters
        ----------
        integrals : Dict[str, List[Dict[int, float]]]
            Dictionary of the integrals for all the possible combinations
            of regularization, subdivision and integration method.

        Returns
        -------
        extrapolated_integrals : Dict[str, List[float]]
            Dictionary of the integrals for all the possible combinations
            of regularization and integration method,
            with the Richardson extrapolation.
        """
        extrapolated_integrals: Dict[str, List[float]] = {}
        for name, ints in integrals.items():
            extrapolated_integrals[name] = [None, None]
            for reg in [True, False]:
                extrapolated_integrals[name][int(reg)] = (
                    ints[int(reg)][1] * ints[int(reg)][4] - ints[int(reg)][2] ** 2
                ) / (ints[int(reg)][1] - 2 * ints[int(reg)][2] + ints[int(reg)][4])
        return extrapolated_integrals
