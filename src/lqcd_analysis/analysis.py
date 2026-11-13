"""An explicit SQLite -> source average -> binned JK -> C2 fit workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from .database import CorrelatorStore
from .fitting import FitError, fit_jackknife
from .physics import c2_periodic
from .selection import load_channel, load_selection
from .statistics import ConfigSamples, bin_configurations, jackknife, jackknife_covariance


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _physical_parameters(parameters, states):
    return np.r_[np.cumsum(np.exp(parameters[:states])), np.exp(parameters[states:])]


def fit_two_point(database, selection, settings, *, verify_database_hash=False):
    """Fit a declared positive diagonal periodic C2, with ordered energies.

    All model inputs, fit times, configuration order, bin size and SVD cutoff
    are explicit. Positive energy *gaps* and overlap magnitudes are log
    parameters. This eliminates state-label permutations but does not identify
    a physical state or establish a sufficient number of states/bin size.
    """
    _require(settings.get("version") == "lqcd.fit2.v1", "Unsupported fit settings")
    _require(settings.get("spectral_assumption") == "positive-diagonal-periodic",
             "Confirm the positive diagonal periodic spectral assumption explicitly")
    with CorrelatorStore(database, verify_database_hash=verify_database_hash,
                         expected_config_sha256=selection.get("config_sha256")) as store:
        channel = load_channel(store, selection)
    _require(channel.identity["kind"] == "2pt", "This workflow fits two-point data")
    _require(channel.identity["source_operator"] == channel.identity["sink_operator"],
             "Positive diagonal C2 requires the same literal operator at source and sink")
    _require(channel.provenance["output_time_convention"] == "source-forward", "C2 fit needs source-forward time")
    _require(selection["component"] == "real", "Positive diagonal C2 requires an explicitly justified real projection")
    samples = channel.config_samples()
    order = settings.get("configuration_order")
    _require(isinstance(order, list) and all(isinstance(x, str) for x in order) and
             len(order) == len(set(order)) and set(order) == set(samples.config_ids),
             "Provide the exact configuration order, including every selected configuration")
    positions = {identity: i for i, identity in enumerate(samples.config_ids)}
    samples = ConfigSamples(tuple(order), samples.values[[positions[x] for x in order]])
    bins = bin_configurations(samples, settings.get("bin_size"))
    data = jackknife(bins)
    model_spec = settings.get("model", {})
    states = model_spec.get("states")
    _require(isinstance(states, int) and not isinstance(states, bool) and 1 <= states <= 8, "Choose 1..8 spectral states")
    energies = np.asarray(model_spec.get("initial_energies"), dtype=float)
    overlaps = np.asarray(model_spec.get("initial_overlaps"), dtype=float)
    _require(energies.shape == (states,) and overlaps.shape == (states,) and
             np.all(np.isfinite(energies)) and np.all(np.isfinite(overlaps)), "Invalid initial state parameters")
    gaps = np.diff(np.r_[0.0, energies])
    _require(np.all(gaps > 0) and np.all(overlaps > 0), "Initial energies must increase; overlaps must be positive")
    limits = np.asarray(model_spec.get("energy_gap_bounds", []) + model_spec.get("overlap_bounds", []), dtype=float)
    _require(limits.shape == (2 * states, 2) and np.all(np.isfinite(limits)) and
             np.all(limits[:, 0] > 0) and np.all(limits[:, 1] > limits[:, 0]), "Explicit positive finite parameter bounds are required")
    initial = np.log(np.r_[gaps, overlaps])
    lower, upper = np.log(limits[:, 0]), np.log(limits[:, 1])
    _require(np.all(initial > lower) and np.all(initial < upper), "Initial parameters must lie strictly inside their bounds")
    window = np.asarray(settings.get("window"))
    _require(window.ndim == 1 and window.dtype.kind in "iu" and window.size > 2 * states and
             np.all(np.diff(window) > 0), "Choose an increasing fixed fit-time list longer than the parameter count")
    _require(np.all((window >= 0) & (window < channel.identity["nt"])), "Fit time outside 0..NT-1")
    _require("svd_rcond" in settings, "Choose the relative covariance cutoff explicitly")
    nt = channel.identity["nt"]

    def model(t, parameters):
        physical = _physical_parameters(parameters, states)
        return c2_periodic(t, nt, physical[:states], physical[states:])

    fit = fit_jackknife(channel.times, data, model, initial, window=window,
                        bounds=(lower, upper), svd_rcond=settings["svd_rcond"],
                        max_nfev=settings.get("max_nfev"))
    _require(fit.full_fit.parameter_rank == 2 * states and fit.full_fit.dof > 0,
             "Fit has insufficient independent information for the chosen state count")
    if any(item.parameter_rank < 2 * states for item in fit.sample_fits):
        raise FitError("A jackknife fit has unidentifiable parameters")
    physical = _physical_parameters(fit.full_fit.parameters, states)
    replicates = np.stack([_physical_parameters(p, states) for p in fit.parameter_samples])
    covariance = jackknife_covariance(replicates)
    errors = np.sqrt(np.diag(covariance))
    active = bool(np.any(fit.full_fit.active_bounds) or any(np.any(item.active_bounds) for item in fit.sample_fits))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"version": "lqcd.fit-result.v1", "status": "requires-bound-review" if active else "validated-numerics",
            "scientific_status": "correlator model only; physical channel and systematic errors require independent confirmation",
            "identity": channel.identity, "fit_settings": settings, "provenance": channel.provenance,
            "model": "sum |Z|^2/(2E) [exp(-Et)+exp(-E(NT-t))]",
            "parameter_order": [f"energy_{i}" for i in range(states)] + [f"overlap_{i}" for i in range(states)],
            "parameters": physical.tolist(), "standard_errors": errors.tolist(),
            "parameter_covariance": covariance.tolist(), "parameter_jackknife_samples": replicates.tolist(),
            "configuration_count": len(order), "measurement_count": len(channel.config_ids),
            "bin_memberships": [list(x) for x in bins.config_ids],
            "chi2": fit.full_fit.chi2, "dof": fit.full_fit.dof, "covariance_rank": fit.full_fit.rank,
            "parameter_rank": fit.full_fit.parameter_rank, "pvalue": fit.full_fit.pvalue,
            "pvalue_note": fit.full_fit.pvalue_note, "covariance_eigenvalues": fit.full_fit.covariance_eigenvalues.tolist(),
            "covariance_cutoff": fit.full_fit.svd_cutoff, "active_bounds": active,
            "software": {"numpy": np.__version__, "scipy": scipy.__version__, "code_sha256": hashes}}


def run_fit_file(path, *, verify_database_hash=False):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    database = Path(settings["database"])
    selection_path = Path(settings["selection"])
    if not database.is_absolute():
        database = path.parent / database
    if not selection_path.is_absolute():
        selection_path = path.parent / selection_path
    return fit_two_point(database, load_selection(selection_path), settings,
                         verify_database_hash=verify_database_hash)
