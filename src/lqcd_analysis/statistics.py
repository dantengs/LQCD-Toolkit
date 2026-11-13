"""Identity-aware, equal-configuration statistics for real observables.

Rows must be independent configurations (or equal-size ordered bins). Source
measurements are averaged *within* each configuration before resampling. Neither
configuration independence nor a sufficient bin size can be inferred here.
Complex data must be explicitly projected or stacked into real/imaginary columns.
"""
from dataclasses import dataclass
from collections.abc import Hashable, Mapping, Sequence
import numpy as np


def _real_finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real numeric data; project complex data explicitly")
    arr = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _ids(values, name, *, unique=True):
    result = tuple(values)
    for item in result:
        if not isinstance(item, Hashable) or item is None:
            raise ValueError(f"{name} must contain non-null hashable identities")
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            raise ValueError(f"{name} contains a non-finite identity")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identities")
    return result


def _readonly(arr):
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ConfigSamples:
    """One row per explicit configuration ID, in caller-supplied chain order."""
    config_ids: tuple
    values: np.ndarray

    def __post_init__(self):
        ids = _ids(self.config_ids, "config_ids")
        values = _real_finite(self.values, "values")
        if not ids or values.ndim < 1 or values.shape[0] != len(ids):
            raise ValueError("values must have one row per nonempty configuration ID list")
        if values.size == 0:
            raise ValueError("empty observable axes are not allowed")
        object.__setattr__(self, "config_ids", ids)
        object.__setattr__(self, "values", _readonly(values))


def average_sources(config_ids: Sequence, values, source_ids: Sequence) -> ConfigSamples:
    """Average distinct sources per configuration; preserve first-seen config order.

    A source ID identifies the whole measurement within a configuration (include
    noise/replica labels if needed). Duplicate (configuration, source) pairs fail.
    Missing source counts are allowed, but never change configuration weights.
    """
    ids = _ids(config_ids, "config_ids", unique=False)
    sources = _ids(source_ids, "source_ids", unique=False)
    data = _real_finite(values, "values")
    if data.ndim < 1 or data.shape[0] != len(ids) or len(sources) != len(ids) or not ids:
        raise ValueError("measurement values, configuration IDs and source IDs must match")
    if len(set(zip(ids, sources))) != len(ids):
        raise ValueError("duplicate configuration/source measurement")
    groups = {}
    for i, identity in enumerate(ids):
        groups.setdefault(identity, []).append(i)
    return ConfigSamples(tuple(groups), np.stack([data[idx].mean(axis=0) for idx in groups.values()]))


@dataclass(frozen=True)
class AlignmentResult:
    channels: dict[str, ConfigSamples]
    dropped_ids: dict[str, tuple]


def align_channels(channels: Mapping[str, ConfigSamples], mode="exact") -> AlignmentResult:
    """Align by IDs, never position; exact requires equal sets, not equal order.

    Intersection is opt-in and returns every discarded ID. Both modes preserve
    the first channel's ordering, which must be meaningful for later binning.
    IDs should include ensemble/chain labels when those could otherwise collide.
    """
    if mode not in ("exact", "intersection") or not channels:
        raise ValueError("provide channels and mode='exact' or 'intersection'")
    first = next(iter(channels.values())).config_ids
    common = set(first)
    for name, samples in channels.items():
        present = set(samples.config_ids)
        if mode == "exact" and present != set(first):
            raise ValueError(f"configuration membership mismatch in channel {name!r}")
        common.intersection_update(present)
    order = tuple(identity for identity in first if identity in common)
    if not order:
        raise ValueError("channels have no common configurations")
    aligned, dropped = {}, {}
    for name, samples in channels.items():
        positions = {identity: i for i, identity in enumerate(samples.config_ids)}
        aligned[name] = ConfigSamples(order, samples.values[[positions[i] for i in order]])
        dropped[name] = tuple(i for i in samples.config_ids if i not in common)
    return AlignmentResult(aligned, dropped)


def bin_configurations(samples: ConfigSamples, bin_size: int) -> ConfigSamples:
    """Average consecutive equal-size bins without dropping a remainder.

    Supplied order is retained, not sorted. Do not concatenate independent chains
    or cross chain boundaries: bin each chain separately. A bin ID is a tuple of
    its constituent configuration IDs, preserving provenance for alignment.
    """
    if isinstance(bin_size, bool) or not isinstance(bin_size, (int, np.integer)) or bin_size < 1:
        raise ValueError("bin_size must be a positive integer")
    n = len(samples.config_ids)
    if n % bin_size:
        raise ValueError(f"{n} configurations cannot form equal bins of {bin_size}; remainder forbidden")
    bins = tuple(tuple(samples.config_ids[i:i + bin_size]) for i in range(0, n, bin_size))
    data = samples.values.reshape((n // bin_size, bin_size) + samples.values.shape[1:]).mean(axis=1)
    return ConfigSamples(bins, data)


def mean_covariance(samples: ConfigSamples) -> np.ndarray:
    """Covariance of the equal-configuration mean, S/N, flattened observables."""
    n = len(samples.config_ids)
    if n < 2:
        raise ValueError("at least two configurations/bins are needed")
    data = samples.values.reshape(n, -1)
    delta = data - data.mean(axis=0)
    return delta.T @ delta / (n * (n - 1))


def jackknife_covariance(replicates) -> np.ndarray:
    """(B-1)/B times centered outer products; exclude the full-sample value."""
    data = _real_finite(replicates, "jackknife replicates")
    if data.ndim < 1 or data.shape[0] < 2 or data.size == 0:
        raise ValueError("at least two nonempty jackknife replicates are needed")
    n = data.shape[0]
    flat = data.reshape(n, -1)
    delta = flat - flat.mean(axis=0)
    return (n - 1) / n * (delta.T @ delta)


@dataclass(frozen=True)
class JackknifeResult:
    config_ids: tuple
    mean: np.ndarray
    samples: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        ids = _ids(self.config_ids, "deleted configuration IDs")
        mean = _real_finite(self.mean, "mean")
        samples = _real_finite(self.samples, "jackknife samples")
        covariance = _real_finite(self.covariance, "covariance")
        if len(ids) < 2 or samples.shape != (len(ids),) + mean.shape:
            raise ValueError("one jackknife replicate per deleted configuration is required")
        expected = jackknife_covariance(samples)
        if covariance.shape != expected.shape or not np.allclose(covariance, expected, rtol=1e-10, atol=1e-14):
            raise ValueError("covariance does not match the jackknife replicates")
        object.__setattr__(self, "config_ids", ids)
        for key, value in (("mean", mean), ("samples", samples), ("covariance", covariance)):
            object.__setattr__(self, key, _readonly(value))


def jackknife(samples: ConfigSamples) -> JackknifeResult:
    """Delete one configuration/bin; the full mean is stored separately."""
    n = len(samples.config_ids)
    if n < 2:
        raise ValueError("at least two configurations/bins are needed")
    mean = samples.values.mean(axis=0)
    replicates = (samples.values.sum(axis=0) - samples.values) / (n - 1)
    return JackknifeResult(samples.config_ids, mean, replicates, jackknife_covariance(replicates))


@dataclass(frozen=True)
class BootstrapResult:
    config_ids: tuple
    indices: np.ndarray
    means: dict[str, np.ndarray]
    samples: dict[str, np.ndarray]
    covariance: np.ndarray
    channel_slices: dict[str, slice]
    seed: int


def bootstrap(channels: Mapping[str, ConfigSamples], n_resamples: int, *, seed: int) -> BootstrapResult:
    """Reproducible paired bootstrap with shared indices and joint covariance.

    Exact ID alignment is mandatory here; call align_channels(..., 'intersection')
    explicitly first if an intersection and its discarded-ID report are desired.
    """
    aligned = align_channels(channels).channels
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("provide an explicit nonnegative integer seed")
    ids = next(iter(aligned.values())).config_ids
    n = len(ids)
    if n < 2:
        raise ValueError("at least two configurations/bins are needed")
    indices = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    means, replicates, slices, columns = {}, {}, {}, []
    offset = 0
    for name, data in aligned.items():
        means[name] = _readonly(data.values.mean(axis=0))
        replicates[name] = _readonly(data.values[indices].mean(axis=1))
        flat = replicates[name].reshape(n_resamples, -1)
        slices[name] = slice(offset, offset + flat.shape[1])
        offset += flat.shape[1]
        columns.append(flat)
    joint = np.concatenate(columns, axis=1)
    delta = joint - joint.mean(axis=0)
    cov = delta.T @ delta / (n_resamples - 1)
    return BootstrapResult(ids, _readonly(indices), means, replicates, _readonly(cov), slices, int(seed))
