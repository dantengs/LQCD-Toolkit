"""Convention-explicit numerical kernels; no legacy data or phase inference.

Spectral kernels use lattice units (a_t E, t/a_t), relativistic state
normalization, and Z_n=<0|O|n>. They describe diagonal meson two-point
functions and a forward three-point contribution, respectively. They do not
include oscillating states, thermal transition sums, contact terms or spin
projection factors. Inputs are plain finite NumPy-compatible numbers.

Form-factor extraction accepts an ALREADY renormalized, real, contravariant
Minkowski current (V^0,V^x,V^y,V^z), metric (+---). It cannot convert an
unidentified Euclidean correlator, decide flavors, or extract P->V / V->P
form factors. See docs/SCIENTIFIC_VALIDATION.md for the source audit.
"""
from dataclasses import dataclass

import numpy as np


def _array(value, name, *, real=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iufc":
        raise ValueError(f"{name} must contain numerical values")
    if real and np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real; no phase conversion is inferred")
    result = np.asarray(raw, dtype=float if real else complex)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, positive=False):
    value = _array(value, name)
    if value.ndim != 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}scalar")
    return float(value)


def _states(energies, overlaps):
    energies = _array(energies, "energies")
    overlaps = _array(overlaps, "overlaps", real=False)
    if energies.ndim != 1 or not energies.size or np.any(energies <= 0):
        raise ValueError("energies must be a nonempty positive vector")
    if overlaps.shape != energies.shape:
        raise ValueError("one overlap is required for each energy")
    return energies, overlaps


def _extent(nt):
    nt = _scalar(nt, "nt", positive=True)
    if nt != int(nt):
        raise ValueError("nt must be an integer")
    return nt


def c2_periodic(t, nt, energies, overlaps):
    """Sum |Z_n|²/(2E_n) [exp(-E_n t)+exp(-E_n(NT-t))].

    Same operator at both ends; equal forward/backward amplitudes. t may
    be scalar or an array, with 0<=t<=NT. No time folding is performed.
    """
    nt = _extent(nt)
    t = _array(t, "t")
    if np.any((t < 0) | (t > nt)):
        raise ValueError("require 0 <= t <= nt")
    e, z = _states(energies, overlaps)
    return np.sum(abs(z)**2 / (2 * e) * (
        np.exp(-t[..., None] * e) + np.exp(-(nt-t[..., None]) * e)), axis=-1)


def c3_forward(tau, T, nt, energies_initial, energies_final,
               overlaps_initial, overlaps_final, matrix_elements):
    """Forward C3=<O_f(T) J(tau) O_i^dagger(0)> spectral contribution.

    Sum_{m,n} Zf_m conj(Zi_n) M[m,n]/(4 Ef_m Ei_n)
       * exp(-Ef_m*(T-tau)-Ei_n*tau), M[m,n]=<f_m|J|i_n>.
    M has shape (n_final,n_initial); complex phases are retained. tau is
    scalar/array and 0<=tau<=T<=NT. Endpoints are algebraically allowed;
    data fits must independently exclude contact terms. NT only validates
    the domain: this model has NO wrapping/thermal contribution, including
    at T=NT. M is in the supplied current convention, not auto-Minkowski.
    """
    nt = _extent(nt)
    T = _scalar(T, "T")
    tau = _array(tau, "tau")
    if T < 0 or T > nt or np.any((tau < 0) | (tau > T)):
        raise ValueError("require 0 <= tau <= T <= nt")
    ei, zi = _states(energies_initial, overlaps_initial)
    ef, zf = _states(energies_final, overlaps_final)
    matrix = _array(matrix_elements, "matrix_elements", real=False)
    if matrix.shape != (ef.size, ei.size):
        raise ValueError("matrix_elements must have shape (n_final,n_initial)")
    amplitude = zf[:, None] * zi.conj()[None, :] * matrix / (4*ef[:, None]*ei[None, :])
    exponent = -(T-tau[..., None, None])*ef[:, None] - tau[..., None, None]*ei[None, :]
    return np.sum(amplitude * np.exp(exponent), axis=(-2, -1))


def _unit(unit):
    if unit not in ("lattice", "GeV"):
        raise ValueError("unit must be explicitly 'lattice' or 'GeV'")


def lattice_momentum(modes, spatial_extent, *, unit, spacing=None):
    """Continuum Fourier momentum 2*pi*n/N, NOT improved 2*sin(p/2).

    modes ends in length 3 and contains integers; spatial_extent is a
    positive integer or length-3 vector. unit='lattice' returns a_s p and
    requires spacing=None. unit='GeV' requires scalar a_s in GeV^-1.
    Anisotropic a_s/a_t conversion is the caller's responsibility.
    """
    _unit(unit)
    n = _array(modes, "modes")
    size = _array(spatial_extent, "spatial_extent")
    if n.ndim < 1 or n.shape[-1] != 3 or np.any(n != np.floor(n)):
        raise ValueError("modes must be integer triples")
    if size.shape not in ((), (3,)) or np.any(size <= 0) or np.any(size != np.floor(size)):
        raise ValueError("spatial_extent must be positive integer(s), scalar or length 3")
    momentum = 2*np.pi*n/size
    if unit == "GeV":
        return momentum / _scalar(spacing, "spacing (GeV^-1)", positive=True)
    if spacing is not None:
        raise ValueError("spacing must be None for lattice momentum")
    return momentum


def _kinematics(energy, momentum, name):
    energy = _scalar(energy, f"{name} energy", positive=True)
    momentum = _array(momentum, f"{name} momentum")
    if momentum.shape != (3,):
        raise ValueError("momentum must be a length-3 vector")
    return energy, momentum


def q_squared(energy_initial, momentum_initial, energy_final, momentum_final, *, unit):
    """q²=(Ei-Ef)²-|pi-pf|²; returns lattice a²q² or GeV².

    Energies/momenta must already share the declared unit (and lattice
    spacing for lattice units). No dispersion relation is imposed here.
    """
    _unit(unit)
    ei, pi = _kinematics(energy_initial, momentum_initial, "initial")
    ef, pf = _kinematics(energy_final, momentum_final, "final")
    return float((ei-ef)**2 - np.dot(pi-pf, pi-pf))


@dataclass(frozen=True)
class PPFormFactors:
    fplus: float
    fminus: float
    fzero: float
    q2: float
    unit: str
    condition_number: float
    residual_norm: float


def pp_form_factors(mass_initial, mass_final, energy_initial, momentum_initial,
                    energy_final, momentum_final, vector_current, *, unit,
                    rtol=1e-9, atol=1e-12, max_condition=1e10):
    """Solve V^mu=(pi+pf)^mu*fplus+(pi-pf)^mu*fminus for P->P.

    fzero=fplus+q²*fminus/(Mi²-Mf²), so fzero(0)=fplus(0) without 1/q².
    Accepts a real, renormalized contravariant Minkowski vector in the
    declared unit; requires continuum on-shell input within rtol/atol.
    All four components must be consistent with the two-form-factor
    decomposition. This is algebraic extraction, NOT a statistical fit:
    noisy inconsistent currents require a separate covariance-aware fit.
    Rejects equal masses and ill-conditioned/zero-recoil kinematics.
    """
    _unit(unit)
    rtol = _scalar(rtol, "rtol", positive=True)
    atol = _scalar(atol, "atol")
    max_condition = _scalar(max_condition, "max_condition", positive=True)
    if atol < 0 or max_condition < 1:
        raise ValueError("require atol >= 0 and max_condition >= 1")
    mi = _scalar(mass_initial, "mass_initial", positive=True)
    mf = _scalar(mass_final, "mass_final", positive=True)
    ei, pi = _kinematics(energy_initial, momentum_initial, "initial")
    ef, pf = _kinematics(energy_final, momentum_final, "final")
    for mass, energy, momentum in ((mi, ei, pi), (mf, ef, pf)):
        if not np.isclose(energy**2-np.dot(momentum, momentum), mass**2, rtol=rtol, atol=atol):
            raise ValueError("on-shell continuum kinematics required; supply an explicit lattice treatment otherwise")
    delta = mi**2-mf**2
    if abs(delta) <= atol + rtol*max(mi**2, mf**2):
        raise ValueError("equal/near-equal masses make this fzero definition singular")
    vi = np.r_[ei, pi]
    vf = np.r_[ef, pf]
    design = np.column_stack((vi+vf, vi-vf))
    singular_values = np.linalg.svd(design, compute_uv=False)
    if singular_values[-1] <= singular_values[0]/max_condition:
        raise ValueError("form factors are not separately identifiable at these kinematics")
    current = _array(vector_current, "Minkowski vector_current")
    if current.shape != (4,):
        raise ValueError("vector_current must have shape (4,) ordered (0,x,y,z)")
    fp, fm = np.linalg.lstsq(design, current, rcond=None)[0]
    residual = float(np.linalg.norm(design @ [fp, fm]-current))
    if residual > atol + rtol*np.linalg.norm(current):
        raise ValueError("current is inconsistent with the P->P decomposition")
    q2 = q_squared(ei, pi, ef, pf, unit=unit)
    return PPFormFactors(float(fp), float(fm), float(fp+q2*fm/delta), q2,
                         unit, float(singular_values[0]/singular_values[-1]), residual)


def z_map(q2, *, tplus, t0):
    """Real z=(sqrt(tplus-q²)-sqrt(tplus-t0))/(sum of roots).

    q2,tplus,t0 must have identical squared-energy units. Requires
    tplus>0, q2<tplus and t0<tplus; branch cut/threshold is excluded.
    The physical channel's branch point is supplied, never guessed.
    """
    q2 = _array(q2, "q2")
    tp = _scalar(tplus, "tplus", positive=True)
    t0 = _scalar(t0, "t0")
    if t0 >= tp or np.any(q2 >= tp):
        raise ValueError("require t0 < tplus and q2 < tplus")
    root = np.sqrt(tp-q2)
    reference = np.sqrt(tp-t0)
    return (root-reference)/(root+reference)


def pole_z_series(q2, coefficients, *, tplus, t0, pole_mass):
    """sum_k a_k*z**k / (1-q²/Mpole²), coefficients in ascending order.

    A simple pole-factored polynomial, NOT a default BCL or BGL model:
    no threshold derivative constraint, outer function or unitarity bound.
    All scales use one consistent unit. pole_mass=None explicitly selects
    no pole. Independent fplus/fzero series do NOT enforce equality at
    q²=0; that constraint must be imposed on their coefficients jointly.
    """
    q2 = _array(q2, "q2")
    coefficients = _array(coefficients, "coefficients")
    if coefficients.ndim != 1 or not coefficients.size:
        raise ValueError("coefficients must be a nonempty vector")
    z = z_map(q2, tplus=tplus, t0=t0)
    value = np.polynomial.polynomial.polyval(z, coefficients)
    if pole_mass is None:
        return value
    mass = _scalar(pole_mass, "pole_mass", positive=True)
    denominator = 1-q2/mass**2
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("evaluation at/too near the explicit pole")
    return value/denominator
