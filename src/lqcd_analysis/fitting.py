"""Correlated real-valued least squares with explicit covariance and diagnostics.

The caller supplies the covariance of the *mean*, e.g. mean_covariance() or the
jackknife covariance, never an unscaled covariance of individual configurations.
Models have signature model(x, parameters) -> one-dimensional prediction.
No priors, x-error model, physical sign convention or parameter bounds are guessed.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution
from .statistics import JackknifeResult, jackknife_covariance, _real_finite, _readonly


class FitError(RuntimeError):
    """A failed fit, including non-finite model evaluations or nonconvergence."""


class JackknifeFitError(FitError):
    def __init__(self, sample_index, config_id, reason):
        self.sample_index = sample_index
        self.config_id = config_id
        super().__init__(f"jackknife sample {sample_index} (deleted ID {config_id!r}) failed: {reason}")


@dataclass(frozen=True)
class FitResult:
    parameters: np.ndarray
    parameter_covariance: Optional[np.ndarray]
    rank: int
    parameter_rank: int
    dof: int
    chi2: float
    pvalue: Optional[float]
    pvalue_note: str
    residuals: np.ndarray
    covariance_eigenvalues: np.ndarray
    svd_cutoff: float
    active_bounds: np.ndarray
    nfev: int
    message: str


def _whitener(covariance, size, rcond):
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("svd_rcond must be finite and strictly between 0 and 1")
    cov = _real_finite(covariance, "covariance of the mean")
    if cov.shape != (size, size):
        raise ValueError("covariance must be square and match y")
    scale = float(np.max(np.abs(cov)))
    if scale == 0:
        raise ValueError("covariance has zero rank")
    if np.max(np.abs(cov - cov.T)) > 1e-12 * scale:
        raise ValueError("covariance must be symmetric")
    # For a symmetric positive-semidefinite matrix, eigenvalues equal singular
    # values. Eigh additionally reveals invalid negative variance directions.
    eigenvalues, vectors = np.linalg.eigh((cov + cov.T) / 2)
    if eigenvalues[0] < -1e-12 * max(scale, abs(eigenvalues[-1])):
        raise ValueError("covariance is not positive semidefinite")
    cutoff = rcond * eigenvalues[-1]
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError("covariance has no retained positive modes")
    whitener = (vectors[:, keep] / np.sqrt(eigenvalues[keep])).T
    return whitener, eigenvalues, float(cutoff)


def correlated_fit(x, y, covariance, model, p0, *, bounds, svd_rcond=1e-12,
                   max_nfev=None, model_jac=None) -> FitResult:
    """Fit in retained covariance modes; dof = data rank - Jacobian rank.

    bounds is required: use (-np.inf, np.inf) explicitly for unconstrained fits.
    Parameter covariance is the unscaled local inverse information matrix, only
    when all parameters are identifiable and no constraint is active. P-values
    are asymptotic chi-square diagnostics, not exact finite-ensemble p-values for
    an estimated covariance. They are omitted at active bounds, zero dof or an
    unidentifiable parameterization. Covariance-null modes are explicitly removed.
    """
    x = _real_finite(x, "x")
    y = _real_finite(y, "y")
    p0 = _real_finite(p0, "p0")
    if y.ndim != 1 or y.size == 0 or x.ndim < 1 or x.shape[0] != y.size:
        raise ValueError("y must be a nonempty vector and x must have matching leading length")
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError("p0 must be a nonempty parameter vector")
    if bounds is None:
        raise ValueError("bounds must be explicit; use (-np.inf, np.inf) for unbounded parameters")
    whitening, eigenvalues, cutoff = _whitener(covariance, y.size, svd_rcond)

    def residual(parameters):
        prediction = _real_finite(model(x, parameters), "model prediction")
        if prediction.shape != y.shape:
            raise ValueError("model prediction shape must exactly match y")
        result = whitening @ (prediction - y)
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite whitened residual")
        return result

    def derivative(parameters):
        jac = _real_finite(model_jac(x, parameters), "model Jacobian")
        if jac.shape != (y.size, p0.size):
            raise ValueError("model Jacobian shape must be (data_count, parameter_count)")
        return whitening @ jac

    try:
        fit = least_squares(residual, p0, bounds=bounds, max_nfev=max_nfev,
                            jac=derivative if model_jac is not None else "2-point")
    except Exception as exc:
        raise FitError(f"least-squares evaluation failed: {exc}") from exc
    if not fit.success or not np.all(np.isfinite(fit.x)) or not np.all(np.isfinite(fit.jac)):
        raise FitError(f"least-squares did not converge to a finite solution: {fit.message}")
    _, singular, vh = np.linalg.svd(fit.jac, full_matrices=False)
    jcut = svd_rcond * singular[0] if singular.size else 0.0
    retained = singular > jcut
    parameter_rank = int(np.count_nonzero(retained))
    rank = whitening.shape[0]
    dof = rank - parameter_rank
    chi2 = float(fit.fun @ fit.fun)
    if not np.isfinite(chi2):
        raise FitError("non-finite chi-square")
    constrained = bool(np.any(fit.active_mask != 0))
    identifiable = parameter_rank == p0.size
    param_cov = None
    if identifiable and not constrained:
        param_cov = (vh.T / singular**2) @ vh
        if not np.all(np.isfinite(param_cov)):
            raise FitError("non-finite parameter covariance")
        param_cov = _readonly(param_cov)
    if dof <= 0:
        pvalue, note = None, "No positive residual degrees of freedom."
    elif constrained:
        pvalue, note = None, "Active bounds invalidate the ordinary chi-square reference distribution."
    elif not identifiable:
        pvalue, note = None, "Parameterization is not locally identifiable."
    else:
        pvalue = float(chi2_distribution.sf(chi2, dof))
        note = "Asymptotic chi-square diagnostic; covariance estimation and window selection are not calibrated."
    return FitResult(_readonly(fit.x.copy()), param_cov, rank, parameter_rank, dof, chi2,
                     pvalue, note, _readonly(fit.fun.copy()), _readonly(eigenvalues), cutoff,
                     _readonly(fit.active_mask.copy()), int(fit.nfev), str(fit.message))


@dataclass(frozen=True)
class JackknifeFitResult:
    full_fit: FitResult
    sample_fits: tuple[FitResult, ...]
    config_ids: tuple
    window: np.ndarray
    parameter_samples: np.ndarray
    parameter_covariance: np.ndarray
    jackknife_mean: np.ndarray
    bias_corrected_parameters: np.ndarray


def fit_jackknife(x, data: JackknifeResult, model, p0, *, window, bounds,
                  svd_rcond=1e-12, max_nfev=None, model_jac=None) -> JackknifeFitResult:
    """Fixed-window fits using the same full-sample covariance for every replicate.

    window is an explicit slice or integer index vector selected before this
    function. No automatic window search or failed-sample filtering is performed.
    The central estimate is full_fit.parameters; a bias-corrected estimate is
    returned separately. This routine conditions on supplied (fixed) x values.
    """
    x = _real_finite(x, "x")
    if data.mean.ndim != 1 or x.ndim < 1 or x.shape[0] != data.mean.size:
        raise ValueError("fit_jackknife needs vector observables and matching x")
    if isinstance(window, slice):
        indices = np.arange(data.mean.size)[window]
    else:
        indices = np.asarray(window)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("window must be a slice or integer index vector")
    if indices.size == 0 or len(set(indices.tolist())) != indices.size:
        raise ValueError("window must be nonempty with unique indices")
    if np.any(indices < 0) or np.any(indices >= data.mean.size):
        raise ValueError("window index outside data range")
    covariance = data.covariance[np.ix_(indices, indices)]
    settings = dict(bounds=bounds, svd_rcond=svd_rcond, max_nfev=max_nfev, model_jac=model_jac)
    full = correlated_fit(x[indices], data.mean[indices], covariance, model, p0, **settings)
    fits = []
    for i, identity in enumerate(data.config_ids):
        try:
            result = correlated_fit(x[indices], data.samples[i, indices], covariance,
                                    model, full.parameters, **settings)
        except Exception as exc:
            raise JackknifeFitError(i, identity, str(exc)) from exc
        fits.append(result)
    params = np.stack([result.parameters for result in fits])
    jk_mean = params.mean(axis=0)
    n = len(fits)
    corrected = n * full.parameters - (n - 1) * jk_mean
    return JackknifeFitResult(full, tuple(fits), data.config_ids, _readonly(indices.copy()),
                             _readonly(params), _readonly(jackknife_covariance(params)),
                             _readonly(jk_mean), _readonly(corrected))
