"""Load labeled raw measurements using a versioned, explicit sidecar selection.

The preservation database deliberately contains no guessed scientific labels.
This layer records those labels and validates the identities needed for an
analysis, without changing the database or treating text/bin copies as samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from .database import CorrelatorStore, StoreError


SELECTION_VERSION = "lqcd.selection.v1"
_CONVENTIONS = {"source-forward", "sink-backward", "sink-forward", "raw-lattice", "initial-forward"}
_MASS = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
_TWO_NAME = re.compile(r"^2pt_([^_]+)_t([0-9]+)_m" + _MASS + r"_m" + _MASS + r"_mom([-0-9]+)\.(?:bin|txt|npy|npz)$")
_THREE_NAME = re.compile(r"^3pt_([^_]+)_t([0-9]+)_T([0-9]+)_m" + _MASS + r"_m" + _MASS + r"_([^_]+)_m" + _MASS + r"_([^_]+)_mom([-0-9]+)_([-0-9]+)\.(?:bin|txt|npy|npz)$")


def _require(ok, message):
    if not ok:
        raise StoreError(message)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def parse_momentum_token(token: str, allowed_momenta):
    """Resolve a historical delimiter-free token against an explicit mode table.

    No single-digit assumption is imposed. Multiple possible triples are an
    error (e.g. 1110 can represent [1,1,10] or [11,1,0]).
    """
    _require(isinstance(token, str), "Momentum token must be text")
    candidates = set()
    for vector in allowed_momenta:
        _require(len(vector) == 3 and all(_integer(v) for v in vector), "Momentum table must contain integer triples")
        if "".join(map(str, vector)) == token:
            candidates.add(tuple(vector))
    _require(len(candidates) == 1, f"Unknown or ambiguous momentum token: {token}")
    return next(iter(candidates))


@dataclass(frozen=True)
class MeasurementChannel:
    config_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    values: np.ndarray
    times: np.ndarray
    identity: dict
    provenance: dict

    def config_samples(self):
        """Equal-weight configurations, after averaging sources within each one."""
        from .statistics import average_sources
        return average_sources(self.config_ids, self.values, self.source_ids)


def load_selection(path):
    target = Path(path)
    _require(target.stat().st_size <= 16 * 1024**2, "Selection manifest exceeds size bound")
    return json.loads(target.read_text(encoding="utf-8"))


def _validate_selection(spec):
    _require(isinstance(spec, dict) and spec.get("version") == SELECTION_VERSION, "Unsupported selection manifest")
    _require(spec.get("template") is not True, "Fill the selection template before loading data")
    identity = spec.get("identity", {})
    _require(isinstance(identity, dict), "Missing channel identity")
    nt = identity.get("nt")
    _require(_integer(nt) and nt > 1, "Explicit NT is required")
    _require(isinstance(identity.get("ensemble"), str) and identity["ensemble"].strip(), "Explicit ensemble identity is required")
    _require(identity.get("kind") in {"2pt", "3pt"}, "Explicit 2pt/3pt kind is required")
    masses = identity.get("ordered_masses", [])
    _require(len(masses) == (2 if identity["kind"] == "2pt" else 3), "Keep all masses in writer order")
    try:
        _require(all(isinstance(m, str) and Decimal(m).is_finite() and Decimal(m) >= 0 for m in masses),
                 "Masses must be finite nonnegative decimal strings")
    except InvalidOperation as exc:
        raise StoreError("Invalid literal mass") from exc
    for name in ("source_operator", "sink_operator"):
        _require(isinstance(identity.get(name), str) and identity[name], f"Explicit {name} is required")
    for name in ("momentum_initial", "momentum_final"):
        v = identity.get(name)
        _require(isinstance(v, list) and len(v) == 3 and all(_integer(x) for x in v), f"Explicit {name} triple is required")
    _require(identity.get("physics_status") in {"confirmed", "unconfirmed"}, "Label the physical channel confirmation status")
    if identity["kind"] == "3pt":
        _require(isinstance(identity.get("current"), str) and identity["current"], "Explicit current is required")
        separation = identity.get("separation")
        _require(_integer(separation) and 0 <= separation < nt, "Explicit 0 <= T < NT is required")
    _require(spec.get("stage") in {"raw-measurement", "synthetic-validation"},
             "Aggregates without exact measurement memberships cannot enter resampling")
    _require(spec.get("time_convention") in _CONVENTIONS, "Known stored time convention is required")
    _require(spec.get("time_action") in {"as-stored", "to-source-forward"}, "Choose the time action explicitly")
    _require(not (spec["time_convention"] == "initial-forward" and spec["time_action"] != "as-stored"),
             "An initial-forward correlator already has its physical tau axis; do not reinterpret its raw origin")
    _require(isinstance(spec.get("convention_evidence"), str) and spec["convention_evidence"].strip(),
             "Provide writer/run evidence for the time convention")
    _require(spec.get("component") in {"real", "imag", "complex"}, "Choose real/imag/complex explicitly")
    _require(spec.get("storage") in {"split-ri", "complex", "real-only"}, "Declare numeric axes/storage explicitly")
    _require(isinstance(spec.get("array_key"), str), "Explicit decoded array key is required")
    _require(isinstance(spec.get("profile"), str), "Explicit decoder profile is required")
    modes = spec.get("allowed_momenta")
    _require(isinstance(modes, list) and modes, "Provide the generating run's allowed momentum table")
    for vector in modes:
        _require(isinstance(vector, list) and len(vector) == 3 and all(_integer(x) for x in vector), "Momentum table must contain integer triples")
    rows = spec.get("measurements")
    _require(isinstance(rows, list) and rows, "No measurements selected")
    _require(len(rows) <= 100000, "Selection has too many measurements")
    return identity, nt, rows


def load_channel(store: CorrelatorStore, spec: dict) -> MeasurementChannel:
    if isinstance(spec, dict) and spec.get("version") == "lqcd.legacy-cf.selection.v1":
        return load_cf_selection(store, spec)
    identity, nt, rows = _validate_selection(spec)
    _require(spec.get("config_sha256") == store.config_sha256, "Selection belongs to a different migration")
    values, configs, sources, lineage = [], [], [], []
    seen_measurements, seen_files, seen_writer_measurements = set(), set(), set()
    time_notes = set()
    for row in rows:
        config, source = row.get("config_id"), row.get("source_id")
        _require(isinstance(config, str) and config and isinstance(source, str) and source,
                 "Every measurement needs explicit configuration and source IDs")
        measurement_id = (config, source)
        _require(measurement_id not in seen_measurements, "Duplicate measurement; text/bin copies are not independent samples")
        seen_measurements.add(measurement_id)
        record = store.find_file(row.get("root_id"), row.get("path"))
        _require(record.sha256 == row.get("sha256"), "Selected original hash differs from database")
        _require(record.id not in seen_files, "The same source file was selected twice")
        seen_files.add(record.id)
        _require(not {"All", "Ave", "mean", "comb"}.intersection(PurePosixPath(record.path).parts),
                 "Historical aggregates cannot be relabeled as raw measurements")
        source_time = row.get("source_time")
        _require(_integer(source_time) and 0 <= source_time < nt, "Source time must be in 0..NT-1")
        writer_measurement = (config, source_time)
        _require(writer_measurement not in seen_writer_measurements,
                 "Duplicate writer measurement across files/roots; a changed source_id does not create a new source")
        seen_writer_measurements.add(writer_measurement)
        # This scientific adapter supports established per-measurement names.
        # Other containers remain inspectable without invented memberships.
        basename = PurePosixPath(record.path).name
        if identity["kind"] == "2pt":
            match = _TWO_NAME.fullmatch(basename)
            _require(match is not None, "Not an established per-measurement two-point basename")
            _require(match[1] == config and int(match[2]) == source_time, "Selection configuration/source disagrees with basename")
            _require(list(match.group(3, 4)) == identity["ordered_masses"], "Ordered literal masses disagree with basename")
            mode = parse_momentum_token(match[5], spec["allowed_momenta"])
            _require(list(mode) == identity["momentum_initial"] == identity["momentum_final"], "Two-point momentum disagrees with basename")
            # CPU2 axes (source_spin,sink_spin,...): ga1 contracts sink
            # indices m,k and ga2 source indices l,n; key is ga1_ga2.
            _require(spec["array_key"] == identity["sink_operator"] + "_" + identity["source_operator"], "Two-point sink_source operator pair disagrees with array key")
        else:
            match = _THREE_NAME.fullmatch(basename)
            _require(match is not None, "Not an established per-measurement three-point basename")
            _require(match[1] == config and int(match[2]) == source_time and int(match[3]) == identity["separation"],
                     "Selection configuration/source/T disagrees with basename")
            _require(list(match.group(4, 5, 7)) == identity["ordered_masses"], "Ordered literal masses disagree with basename")
            _require(match[6] == identity["sink_operator"] and match[8] == identity["source_operator"], "Operator labels disagree with basename")
            _require(list(parse_momentum_token(match[9], spec["allowed_momenta"])) == identity["momentum_initial"] and
                     list(parse_momentum_token(match[10], spec["allowed_momenta"])) == identity["momentum_final"], "Three-point momenta disagree with basename")
            _require(spec["array_key"] == identity["current"], "Current label disagrees with array key")
        decoded = store.decode(record.id, profile=spec["profile"])
        _require(spec["array_key"] in decoded.arrays, "Selected array key is absent")
        data = decoded.arrays[spec["array_key"]]
        storage = spec["storage"]
        if storage == "split-ri":
            _require(data.shape == (2, nt) and data.dtype.kind in "fiu", "Expected numeric (real/imag,time) array")
            series = data[0] + 1j * data[1] if spec["component"] == "complex" else data[0 if spec["component"] == "real" else 1]
        elif storage == "complex":
            _require(data.shape == (nt,) and data.dtype.kind == "c", "Expected declared complex time vector")
            series = data if spec["component"] == "complex" else (data.real if spec["component"] == "real" else data.imag)
        else:
            _require(data.shape == (nt,) and data.dtype.kind in "fiu" and spec["component"] == "real",
                     "A real-only file cannot provide imaginary or complex data")
            series = data
        _require(np.all(np.isfinite(series)), "Non-finite correlator")
        convention = spec["time_convention"]
        if spec["time_action"] == "to-source-forward":
            tau = np.arange(nt)
            if convention == "source-forward":
                indices = tau
            elif convention == "raw-lattice":
                indices = (source_time + tau) % nt
            elif convention == "sink-backward":
                _require(identity["kind"] == "3pt", "Sink conventions require a three-point separation")
                indices = (identity["separation"] - tau) % nt
            else:
                _require(identity["kind"] == "3pt", "Sink conventions require a three-point separation")
                indices = (tau - identity["separation"]) % nt
            series = series[indices]
            time_notes.add(f"{convention}->source-forward")
        values.append(np.asarray(series))
        configs.append(config)
        sources.append(source)
        lineage.append({"file_id": record.id, "root_id": record.root_id, "path": record.path,
                        "sha256": record.sha256, "config_id": config, "source_id": source,
                        "source_time": source_time, "decode": decoded.metadata})
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    provenance = {"selection_version": SELECTION_VERSION, "selection_sha256": spec_hash,
                  "database_config_sha256": store.config_sha256, "database_schema": store.schema_version,
                  "database_hash_verified": store.database_hash_verified,
                  "component": spec["component"], "time_action": spec["time_action"],
                  "output_time_convention": "source-forward" if spec["time_action"] == "to-source-forward" else spec["time_convention"],
                  "transformations": sorted(time_notes), "convention_evidence": spec["convention_evidence"],
                  "files": lineage}
    return MeasurementChannel(tuple(configs), tuple(sources), np.stack(values), np.arange(nt), dict(identity), provenance)


def load_cf_selection(store: CorrelatorStore, spec: dict) -> MeasurementChannel:
    """Route the explicit CF manifest to its single shared adapter."""
    from .legacy_profiles import load_cf_channel
    channel = load_cf_channel(store, spec)
    _require(spec.get("time_convention") == channel.provenance["output_time_convention"],
             "CF time convention disagrees with its writer profile")
    return channel
