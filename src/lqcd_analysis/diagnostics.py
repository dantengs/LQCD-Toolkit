"""Explicit positive-real effective-energy diagnostics, with complete jackknife.

The diagnostic idea is retained from the original repository's effective-mass
workflow. This implementation is independent: no implicit projection, absolute
value, domain clipping, path convention or legacy dependency is imported.
"""
from dataclasses import dataclass
import numpy as np
from .statistics import JackknifeResult, jackknife_covariance, _real_finite, _readonly


@dataclass(frozen=True)
class EffectiveEnergy:
    times: np.ndarray
    values: np.ndarray
    method: str
    spacing: float


def effective_energy(correlator, *, method, spacing=1.0, time_origin=0.0):
    """Return a diagnostic for a regularly sampled, explicitly positive real C.

    log: [log C(t)-log C(t+dt)]/dt, at midpoint t+dt/2. Exact for a
    forward single exponential; it is NOT a periodic-cosh energy estimator.
    cosh: acosh((C(t-dt)+C(t+dt))/(2 C(t)))/dt, at interior t. Exact for
    a single periodic equal-amplitude forward/backward exponential.

    dt is in the caller's time units, so output energies have inverse units.
    No wrap across the supplied endpoints, sign flip or real-part extraction
    is performed. Every supplied C must be finite and strictly positive.
    """
    data=_real_finite(correlator,"explicitly projected real correlator")
    if data.ndim!=1 or data.size<2 or np.any(data<=0):
        raise ValueError("effective energy requires a vector of at least two strictly positive real values")
    if isinstance(spacing,(bool,np.bool_)) or not np.isscalar(spacing) or not np.isfinite(spacing) or spacing<=0:
        raise ValueError("spacing must be a positive finite scalar")
    if not np.isscalar(time_origin) or not np.isfinite(time_origin):
        raise ValueError("time_origin must be a finite scalar")
    if method=="log":
        # A difference of logarithms avoids overflow in C(t)/C(t+dt).
        values=(np.log(data[:-1])-np.log(data[1:]))/spacing
        times=time_origin+(np.arange(data.size-1)+.5)*spacing
    elif method=="cosh":
        if data.size<3:
            raise ValueError("cosh effective energy requires at least three times")
        with np.errstate(over="ignore",divide="ignore",invalid="ignore"):
            argument=.5*(data[:-2]/data[1:-1]+data[2:]/data[1:-1])
        if not np.all(np.isfinite(argument)) or np.any(argument<1):
            bad=np.flatnonzero(~np.isfinite(argument)|(argument<1))+1
            raise ValueError(f"cosh effective energy is undefined at input time indices {bad.tolist()}; no clipping is applied")
        values=np.arccosh(argument)/spacing
        times=time_origin+np.arange(1,data.size-1)*spacing
    else:
        raise ValueError("method must be explicitly 'log' or 'cosh'")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("effective-energy output is not finite")
    return EffectiveEnergy(_readonly(times),_readonly(values),method,float(spacing))


class EffectiveEnergySampleError(ValueError):
    """A nonlinear diagnostic failed on one identified delete-one sample."""
    def __init__(self,sample_index,config_id,reason):
        self.sample_index=sample_index
        self.config_id=config_id
        super().__init__(f"effective energy failed for jackknife sample {sample_index} (deleted ID {config_id!r}): {reason}")


@dataclass(frozen=True)
class EffectiveEnergyJackknife:
    times: np.ndarray
    mean: np.ndarray
    samples: np.ndarray
    covariance: np.ndarray
    config_ids: tuple
    bias_corrected_mean: np.ndarray
    method: str
    spacing: float


def effective_energy_jackknife(data: JackknifeResult, *, method, spacing=1.0, time_origin=0.0):
    """Transform the full estimate and every JK replicate; never drop failures.

    First obtain data with source averaging, ID alignment and optional equal-size
    binning, then statistics.jackknife(). mean is the full-sample nonlinear
    estimate; the bias-corrected value is separate. Covariance includes all
    cross-time correlations and the (B-1)/B jackknife factor. This does not select
    a fit window or certify a plateau/state identification.
    """
    if not isinstance(data,JackknifeResult) or data.mean.ndim!=1:
        raise ValueError("provide a JackknifeResult with one time vector per replicate")
    options=dict(method=method,spacing=spacing,time_origin=time_origin)
    full=effective_energy(data.mean,**options)
    values=[]
    for index,(identity,replicate) in enumerate(zip(data.config_ids,data.samples)):
        try:
            values.append(effective_energy(replicate,**options).values)
        except ValueError as exc:
            raise EffectiveEnergySampleError(index,identity,str(exc)) from exc
    samples=np.stack(values)
    n=len(samples)
    corrected=n*full.values-(n-1)*samples.mean(axis=0)
    covariance=jackknife_covariance(samples)
    if not np.all(np.isfinite(corrected)) or not np.all(np.isfinite(covariance)):
        raise ValueError("non-finite effective-energy jackknife statistic")
    return EffectiveEnergyJackknife(full.times,full.values,_readonly(samples),_readonly(covariance),
        data.config_ids,_readonly(corrected),method,float(spacing))
