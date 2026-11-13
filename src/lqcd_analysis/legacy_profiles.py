"""Explicit writer adapters for legacy split-real/imaginary correlators.

Container decoding, database access, statistics and fitting belong to the
existing core. This module validates writer labels and preserves stored order.
It never discovers a machine path or infers a physical channel from masses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import PurePosixPath
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .decoders import DEFAULT_LIMITS, DecodeResult, Limits, decode_numeric
from .database import CorrelatorStore

if TYPE_CHECKING:
    from .selection import MeasurementChannel


class LegacyProfileError(ValueError):
    """Writer conventions or measurement identities are absent or inconsistent."""


WRITER_PROFILES = (
    "two-point.forward.v1",
    "three-point.source-backward.v1",
    "three-point.sink-backward.v1",
)
PICKLE_PROFILE = "savedata.pickle.numeric.v1"
LEGACY_SELECTION_VERSION = "lqcd.legacy-cf.selection.v1"
DEFAULT_OPERATORS = frozenset(
    ["Id", *[f"g{i}" for i in range(1, 6)],
     *[f"g5.g{i}" for i in range(1, 5)],
     *[f"sigma{i}{j}" for i in range(1, 5) for j in range(i + 1, 5)]]
)
_MASS = r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
_NAME = re.compile(
    r"^cf(?P<config>[0-9]+)_t(?P<source>[0-9]+)"
    r"(?:_T(?P<separation>[0-9]+))?_"
    r"(?P<m1>" + _MASS + r")_(?P<m2>" + _MASS + r")"
    r"(?:_(?P<m3>" + _MASS + r"))?\.bin$"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise LegacyProfileError(message)


def _integer(value):
    return type(value) is int


def _text(value, name):
    _require(type(value) is str and bool(value.strip()) and len(value) <= 4096
             and not any(ord(c) < 32 or ord(c) == 127 for c in value),
             f"Explicit nonempty {name} text is required")
    return value


def _mass(value):
    _require(type(value) in (str, Decimal), "Masses must be decimal strings or Decimal values")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise LegacyProfileError("Invalid decimal mass") from exc
    _require(result.is_finite() and result > 0, "Masses must be finite and positive")
    return result


def _sequence(value, name):
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
             f"Explicit {name} sequence is required")
    return tuple(value)


def _time(value, nt, name):
    _require(_integer(value) and 0 <= value < nt, f"{name} must be an integer in 0..NT-1")
    return value


@dataclass(frozen=True)
class WriterEvidence:
    """A caller-provided version and source description for the actual writer."""

    version: str
    source: str

    def __post_init__(self):
        _text(self.version, "writer evidence version")
        _text(self.source, "writer evidence source")


@dataclass(frozen=True)
class WriterProfile:
    name: str
    nt: int
    writer_evidence: WriterEvidence

    def __post_init__(self):
        _require(type(self.name) is str and self.name in WRITER_PROFILES, "Unknown explicit writer profile")
        _require(_integer(self.nt) and 0 < self.nt <= DEFAULT_LIMITS.max_dimension,
                 "NT must be a positive integer within the numeric decoder extent limit")
        _require(type(self.writer_evidence) is WriterEvidence,
                 "Provide structured WriterEvidence with version and source")

    @property
    def kind(self):
        return "2pt" if self.name == WRITER_PROFILES[0] else "3pt"


@dataclass(frozen=True)
class CFIdentity:
    basename: str
    kind: str
    config_id: str
    source_time: int
    separation: int | None
    ordered_masses: tuple[Decimal, ...]


@dataclass(frozen=True)
class ChannelKey:
    kind: str
    ordered_masses: tuple[Decimal, ...]
    ordered_operators: tuple[str, ...]
    momenta: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class LegacyMeasurement:
    config_id: str
    source_id: str
    values: np.ndarray
    absolute_times: np.ndarray
    file_identity: CFIdentity
    channel_key: ChannelKey
    identity: dict
    provenance: dict


def parse_cf_filename(basename: str) -> CFIdentity:
    """Parse writer labels only; the name never selects a time convention."""
    _text(basename, "CF basename")
    _require("/" not in basename and "\\" not in basename, "Pass a basename, not a machine path")
    match = _NAME.fullmatch(basename)
    _require(match is not None, "Not a supported per-configuration CF basename")
    third = match["m3"]
    separation = match["separation"]
    _require((third is None) == (separation is None), "A three-point basename needs T and three masses")
    masses = tuple(_mass(match[name]) for name in ("m1", "m2", "m3") if match[name] is not None)
    return CFIdentity(basename, "2pt" if third is None else "3pt", match["config"],
                      int(match["source"]), int(separation) if separation is not None else None, masses)


def parse_array_key(key: str, kind: str, allowed_momenta, *, allowed_operators=None) -> ChannelKey:
    """Parse literal operator slots and resolve momentum tokens unambiguously.

    Explicit custom operator labels are permitted as literals; none are eval'd.
    Physical source/sink/current roles are supplied separately by the caller.
    """
    from .selection import parse_momentum_token

    _text(key, "array key")
    _require(kind in {"2pt", "3pt"}, "Choose 2pt or 3pt explicitly")
    parts = key.split("_")
    count = 2 if kind == "2pt" else 3
    _require(len(parts) == (5 if count == 2 else 8), "Array key does not match the declared point count")
    masses = tuple(_mass(parts[2 * i]) for i in range(count))
    operators = tuple(parts[2 * i + 1] for i in range(count))
    if allowed_operators is None:
        allowed = DEFAULT_OPERATORS
    else:
        allowed = _sequence(allowed_operators, "allowed operator")
        _require(bool(allowed), "Operator table must be nonempty and unique")
        for operator in allowed:
            _text(operator, "allowed operator")
            _require("_" not in operator, "Operator labels cannot contain the key delimiter")
        _require(len(allowed) == len(set(allowed)), "Operator table must be nonempty and unique")
    _require(all(operator in allowed for operator in operators), "Unknown literal gamma/operator label")
    modes = _sequence(allowed_momenta, "allowed momentum")
    _require(bool(modes), "Provide the generating run's allowed momentum table")
    try:
        momenta = tuple(parse_momentum_token(token, modes) for token in parts[2 * count:])
    except (TypeError, ValueError) as exc:
        raise LegacyProfileError("Unknown, malformed or ambiguous momentum token") from exc
    return ChannelKey(kind, masses, operators, momenta)


def stored_to_absolute_times(profile: WriterProfile, source_time: int,
                             separation: int | None = None) -> np.ndarray:
    """Absolute lattice index of each stored entry; no data are reordered."""
    _require(type(profile) is WriterProfile, "Provide an explicit WriterProfile")
    source_time = _time(source_time, profile.nt, "Source time")
    t = np.arange(profile.nt, dtype=np.int64)
    if profile.kind == "2pt":
        _require(separation is None, "A two-point profile has no three-point separation")
        result = (source_time + t) % profile.nt
    else:
        separation = _time(separation, profile.nt, "Separation")
        origin = source_time if profile.name == WRITER_PROFILES[1] else source_time + separation
        result = (origin - t) % profile.nt
    result.setflags(write=False)
    return result


def _vector(value, name):
    value = _sequence(value, name)
    _require(len(value) == 3 and all(_integer(x) for x in value), f"{name} must be an integer triple")
    return value


def _validate_identity(identity, file, key, profile):
    _require(type(identity) is dict, "Provide an explicit scientific identity object")
    _require(identity.get("kind") == file.kind == profile.kind, "Point count disagrees with writer/profile")
    _require(identity.get("nt") == profile.nt and _integer(identity.get("nt")), "Identity NT disagrees with profile")
    _text(identity.get("ensemble"), "ensemble identity")
    _require(type(identity.get("physics_status")) is str
             and identity["physics_status"] in {"confirmed", "synthetic-validation"},
             "Confirm physical conventions or declare synthetic-validation explicitly")
    _text(identity.get("physics_evidence"), "physical convention evidence")
    _text(identity.get("config_id"), "configuration ID")
    _require(identity["config_id"] == file.config_id, "Configuration ID disagrees with basename (including leading zeros)")
    _text(identity.get("source_id"), "source ID")
    _require(_integer(identity.get("source_time")) and identity["source_time"] == file.source_time,
             "Source time disagrees with basename")
    masses = tuple(_mass(x) for x in _sequence(identity.get("ordered_masses"), "ordered mass"))
    _require(masses == file.ordered_masses == key.ordered_masses, "Filename, key and identity masses disagree")
    roles = _sequence(identity.get("mass_roles"), "mass role")
    _require(len(roles) == len(masses), "Label each mass slot explicitly")
    for role in roles:
        _text(role, "mass role")
    _require(len(set(roles)) == len(roles), "Mass roles must be distinct labels in writer order")
    operators = _sequence(identity.get("ordered_operators"), "ordered operator")
    _require(operators == key.ordered_operators, "Explicit operator slots disagree with array key")
    required_roles = {"source", "sink"} if file.kind == "2pt" else {"source", "sink", "current"}
    mapping = identity.get("operator_roles")
    _require(type(mapping) is dict and set(mapping) == required_roles,
             "Map source/sink/current roles to explicit writer operator slots")
    _require(all(_integer(i) for i in mapping.values())
             and set(mapping.values()) == set(range(len(operators))), "Operator roles must cover each slot exactly once")
    for role, i in mapping.items():
        field = "current" if role == "current" else role + "_operator"
        _require(identity.get(field) == operators[i], "Physical operator role disagrees with its writer slot")
    initial = _vector(identity.get("momentum_initial"), "Initial momentum")
    final = _vector(identity.get("momentum_final"), "Final momentum")
    if file.kind == "2pt":
        _require(initial == final == key.momenta[0], "Two-point momentum disagrees with key")
        _require(identity.get("separation") is None, "Two-point identity has no separation")
    else:
        momentum_roles = identity.get("momentum_roles")
        _require(type(momentum_roles) is dict and set(momentum_roles) == {"initial", "final"}
                 and all(_integer(i) for i in momentum_roles.values())
                 and set(momentum_roles.values()) == {0, 1}, "Map both momentum slots explicitly")
        _require(initial == key.momenta[momentum_roles["initial"]]
                 and final == key.momenta[momentum_roles["final"]], "Physical momenta disagree with explicit writer slots")
        _require(_integer(identity.get("separation")) and identity["separation"] == file.separation,
                 "Separation disagrees with basename")
    try:
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LegacyProfileError("Scientific identity must be a finite JSON-compatible object") from exc
    return json.loads(canonical), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adapt_cf(decoded: DecodeResult, *, filename: str, profile: WriterProfile,
             array_key: str, identity: dict, allowed_momenta, component: str,
             stage: str = "configuration", allowed_operators=None) -> LegacyMeasurement:
    """Validate one CF observation using already-decoded, hash-checked bytes.

    All/Ave and other aggregates cannot enter this scientific adapter. Even an
    exact membership declaration does not define a derived-key averaging rule.
    """
    _require(type(stage) is str and stage in {"configuration", "synthetic-validation"},
             "Aggregates need exact memberships and a derived-observable contract; scientific loading is unsupported")
    _require(type(decoded) is DecodeResult and decoded.profile == PICKLE_PROFILE,
             "Reuse the core's explicit numeric pickle DecodeResult")
    _require(type(profile) is WriterProfile, "Provide an explicit WriterProfile")
    file = parse_cf_filename(filename)
    _require(file.kind == profile.kind, "Filename point count disagrees with writer profile")
    _text(array_key, "selected array key")
    _require(type(decoded.arrays) is dict and bool(decoded.arrays) and array_key in decoded.arrays,
             "Selected array key is absent")
    parsed = {}
    semantic_keys = set()
    for label, data in decoded.arrays.items():
        key = parse_array_key(label, file.kind, allowed_momenta, allowed_operators=allowed_operators)
        _require(key.ordered_masses == file.ordered_masses, "A key's mass slots disagree with the CF basename")
        _require(key not in semantic_keys, "Duplicate semantic key with differently formatted mass labels")
        semantic_keys.add(key)
        _require(type(data) is np.ndarray and data.shape == (2, profile.nt) and data.dtype.kind in "fiu",
                 "Every CF key must have a real numeric (2,NT) array")
        _require(np.all(np.isfinite(data)), "Non-finite CF array")
        parsed[label] = key
    key = parsed[array_key]
    normalized, identity_hash = _validate_identity(identity, file, key, profile)
    absolute = stored_to_absolute_times(profile, file.source_time, file.separation)
    _require(type(component) is str and component in {"real", "imag", "complex"},
             "Choose real, imag or complex explicitly")
    data = decoded.arrays[array_key]
    values = data[0] + 1j * data[1] if component == "complex" else data[0 if component == "real" else 1]
    values = np.array(values, copy=True)
    values.setflags(write=False)
    provenance = {
        "writer_profile": profile.name,
        "writer_evidence": {"version": profile.writer_evidence.version, "source": profile.writer_evidence.source},
        "numeric_decoder": decoded.profile, "decode": dict(decoded.metadata),
        "array_key": array_key, "identity_sha256": identity_hash,
        "time_action": "as-stored", "time_mapping": "stored index -> absolute lattice index",
        "component": component, "stage": stage,
        "scientific_scope": "Declared identity and writer layout validated; no fitted or inferred physical result",
    }
    return LegacyMeasurement(file.config_id, normalized["source_id"], values, absolute,
                             file, key, normalized, provenance)


def decode_cf(payload: bytes, *, limits: Limits = DEFAULT_LIMITS, **kwargs) -> LegacyMeasurement:
    """Thin standalone wrapper; database callers should reuse store.decode."""
    decoded = decode_numeric(payload, PICKLE_PROFILE, limits=limits)
    result = adapt_cf(decoded, **kwargs)
    return replace(result, provenance={**result.provenance, "payload_sha256": hashlib.sha256(payload).hexdigest()})


def _common_identity(identity):
    common = {k: v for k, v in identity.items() if k not in {"config_id", "source_id", "source_time"}}
    comparison = dict(common)
    # Compare decimal labels exactly without a floating conversion or a
    # Decimal.normalize operation that could round using the current context.
    masses = []
    for literal in common["ordered_masses"]:
        value = _mass(literal).as_tuple()
        digits, exponent = value.digits, value.exponent
        end = len(digits)
        while end > 1 and digits[end - 1] == 0:
            end -= 1
            exponent += 1
        masses.append("".join(map(str, digits[:end])) + "e" + str(exponent))
    comparison["ordered_masses"] = masses
    canonical = json.dumps(comparison, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return common, canonical


def to_measurement_channel(measurements: Sequence[LegacyMeasurement], *,
                           time_action: str) -> MeasurementChannel:
    """Bridge explicit CF observations into the shared statistics interface.

    Values retain their stored tau axis. Each observation's absolute lattice
    mapping remains in provenance; those source-dependent maps are not channel
    times. At least two declared configurations are required. This does not
    prove their independence or choose a bin size or covariance model.
    """
    from .selection import MeasurementChannel

    _require(type(time_action) is str and time_action == "as-stored",
             "Choose time_action='as-stored'; this bridge does not transform time axes")
    items = _sequence(measurements, "legacy measurement")
    _require(0 < len(items) <= 100000, "Provide a nonempty bounded observation sequence")
    profiles, components, stages = [], [], []
    configs, sources, values, lineage = [], [], [], []
    seen_ids, seen_writer_sources = set(), set()
    common_identity = common_key = comparison = None
    nt = None
    for item in items:
        _require(type(item) is LegacyMeasurement, "Bridge only validated LegacyMeasurement objects")
        _require(type(item.provenance) is dict, "Missing writer adaptation provenance")
        evidence = item.provenance.get("writer_evidence")
        _require(type(evidence) is dict and set(evidence) == {"version", "source"}, "Missing structured writer evidence")
        _require(type(item.identity) is dict, "Missing observation identity")
        profile = WriterProfile(item.provenance.get("writer_profile"), item.identity.get("nt"),
                                WriterEvidence(evidence["version"], evidence["source"]))
        _require(type(item.file_identity) is CFIdentity and type(item.channel_key) is ChannelKey,
                 "Missing parsed writer identity")
        _require(parse_cf_filename(item.file_identity.basename) == item.file_identity,
                 "Parsed CF identity disagrees with its basename")
        normalized, identity_hash = _validate_identity(item.identity, item.file_identity, item.channel_key, profile)
        _require(item.provenance.get("identity_sha256") == identity_hash,
                 "Observation identity changed after writer validation")
        _require(item.config_id == normalized["config_id"] and item.source_id == normalized["source_id"],
                 "Observation IDs disagree with the validated identity")
        _require(item.provenance.get("time_action") == "as-stored"
                 and item.provenance.get("numeric_decoder") == PICKLE_PROFILE,
                 "Observation lost the shared decoder or stored-time contract")
        stage = item.provenance.get("stage")
        _require(type(stage) is str and stage in {"configuration", "synthetic-validation"},
                 "Aggregates cannot enter the CF statistics bridge")
        component = item.provenance.get("component")
        _require(type(component) is str and component in {"real", "imag", "complex"}, "Missing explicit component")
        if nt is None:
            nt = profile.nt
            _require(len(items) * nt <= DEFAULT_LIMITS.max_elements, "Bridged observation element limit")
        else:
            _require(profile.nt == nt, "Cannot mix time extents in one channel")
        expected_times = stored_to_absolute_times(profile, item.file_identity.source_time, item.file_identity.separation)
        _require(type(item.absolute_times) is np.ndarray and item.absolute_times.dtype.kind in "iu"
                 and item.absolute_times.shape == (nt,)
                 and np.array_equal(item.absolute_times, expected_times), "Absolute-time mapping disagrees with writer profile")
        _require(type(item.values) is np.ndarray and item.values.shape == (nt,)
                 and item.values.dtype.kind in ("c" if component == "complex" else "fiu")
                 and np.all(np.isfinite(item.values)), "Observation values disagree with component or time extent")
        current_identity, current_comparison = _common_identity(normalized)
        if common_identity is None:
            common_identity, comparison, common_key = current_identity, current_comparison, item.channel_key
        else:
            _require(current_comparison == comparison and item.channel_key == common_key,
                     "Cannot mix physical channel identities or writer key slots")
            _require(profile == profiles[0] and component == components[0] and stage == stages[0],
                     "Cannot mix writer profiles/evidence, components or observation stages")
        measurement_id = (item.config_id, item.source_id)
        writer_id = (item.config_id, item.file_identity.source_time)
        _require(measurement_id not in seen_ids, "Duplicate configuration/source observation")
        _require(writer_id not in seen_writer_sources,
                 "Duplicate writer source time; a changed source ID does not create an observation")
        seen_ids.add(measurement_id)
        seen_writer_sources.add(writer_id)
        profiles.append(profile)
        components.append(component)
        stages.append(stage)
        configs.append(item.config_id)
        sources.append(item.source_id)
        values.append(item.values)
        lineage.append({"config_id": item.config_id, "source_id": item.source_id,
                        "source_time": item.file_identity.source_time,
                        "basename": item.file_identity.basename,
                        "absolute_lattice_times": expected_times.tolist(),
                        "adaptation": dict(item.provenance)})
    configuration_order = tuple(dict.fromkeys(configs))
    _require(len(configuration_order) >= 2,
             "A statistics channel needs at least two declared configurations; multiple sources are not configurations")
    convention = {
        WRITER_PROFILES[0]: "source-forward",
        WRITER_PROFILES[1]: "source-backward",
        WRITER_PROFILES[2]: "sink-backward",
    }[profiles[0].name]
    stacked = np.stack(values)
    stacked.setflags(write=False)
    times = np.arange(nt, dtype=np.int64)
    times.setflags(write=False)
    provenance = {
        "adapter_version": "lqcd.legacy-cf-channel.v1", "writer_profile": profiles[0].name,
        "component": components[0], "time_action": "as-stored",
        "output_time_convention": convention, "time_axis": "stored tau indices",
        "configuration_count": len(configuration_order), "measurement_count": len(items),
        "configuration_order": list(configuration_order), "files": lineage,
        "statistical_scope": "Declared configurations; independence, binning and fit model require separate validation",
    }
    return MeasurementChannel(tuple(configs), tuple(sources), stacked, times, common_identity, provenance)


def _reject_aggregate_path(path):
    reserved = {"all", "ave", "mean", "comb", "source-average", "source_average", "aggregates"}
    _require(not reserved.intersection(part.casefold() for part in PurePosixPath(path).parts),
             "Historical aggregate paths cannot be relabeled as raw CF observations")


def load_cf_channel(store: CorrelatorStore, spec: dict) -> MeasurementChannel:
    """Load an explicit CF sidecar through the shared, hash-checking store.

    No database is opened here. The caller owns the store's seal/hash policy.
    This is the legacy writer dispatch target for the shared selection layer.
    """
    _require(type(spec) is dict and spec.get("version") == LEGACY_SELECTION_VERSION,
             "Unsupported legacy CF selection manifest")
    _require(spec.get("template") is not True, "Fill the selection template before loading data")
    config_hash = spec.get("config_sha256")
    _require(type(config_hash) is str and _SHA256.fullmatch(config_hash) is not None
             and config_hash == store.config_sha256, "Selection belongs to a different migration configuration")
    _require(store.schema_version in {"lqcd.opaque-chunks.v1", "lqcd.opaque-chunks.v2"},
             "Unsupported shared store schema")
    _require(spec.get("profile") == PICKLE_PROFILE, "Select the existing numeric pickle decoder explicitly")
    raw_profile = spec.get("writer_profile")
    _require(type(raw_profile) is dict and set(raw_profile) == {"name", "nt", "writer_evidence"},
             "Explicit writer_profile fields are required")
    evidence = raw_profile["writer_evidence"]
    _require(type(evidence) is dict and set(evidence) == {"version", "source"}, "Provide structured writer evidence")
    profile = WriterProfile(raw_profile["name"], raw_profile["nt"],
                            WriterEvidence(evidence["version"], evidence["source"]))
    identity = spec.get("identity")
    _require(type(identity) is dict and not {"config_id", "source_id", "source_time"}.intersection(identity),
             "Common identity must keep observation configuration/source fields in measurements")
    _require(identity.get("nt") == profile.nt and _integer(identity.get("nt")), "Identity NT disagrees with writer profile")
    _require(identity.get("kind") == profile.kind, "Identity point count disagrees with writer profile")
    _require(spec.get("time_action") == "as-stored", "Choose time_action='as-stored' explicitly")
    stage = spec.get("stage")
    _require(type(stage) is str and stage in {"raw-measurement", "synthetic-validation"},
             "Aggregate stages have no supported CF scientific loading contract")
    _require(identity.get("physics_status") == ("confirmed" if stage == "raw-measurement" else "synthetic-validation"),
             "Selection stage and physical confirmation declaration disagree")
    component = spec.get("component")
    _require(type(component) is str and component in {"real", "imag", "complex"}, "Choose an explicit component")
    array_key = _text(spec.get("array_key"), "selected array key")
    allowed_momenta = _sequence(spec.get("allowed_momenta"), "allowed momentum")
    selected_key = parse_array_key(array_key, profile.kind, allowed_momenta,
                                    allowed_operators=spec.get("allowed_operators"))
    rows = _sequence(spec.get("measurements"), "measurement")
    _require(0 < len(rows) <= 100000 and len(rows) * profile.nt <= DEFAULT_LIMITS.max_elements,
             "Selection exceeds the observation or numeric element limit")
    try:
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise LegacyProfileError("Selection must be a finite JSON-compatible object") from exc
    _require(len(canonical) <= 16 * 1024**2, "Selection manifest exceeds the shared sidecar size bound")
    observations, lineage = [], []
    seen_files, seen_ids, seen_writer_sources = set(), set(), set()
    source_rows = {}
    required = {"root_id", "path", "sha256", "config_id", "source_id", "source_time"}
    for row in rows:
        _require(type(row) is dict and set(row) == required, "Every observation needs exactly the declared source/identity fields")
        root_id = _text(row["root_id"], "source root ID")
        path = _text(row["path"], "store-relative path")
        _require("\\" not in path and not path.startswith("/")
                 and all(part not in {"", ".", ".."} for part in path.split("/")), "Invalid store-relative source path")
        _reject_aggregate_path(path)
        row_hash = row["sha256"]
        _require(type(row_hash) is str and _SHA256.fullmatch(row_hash) is not None, "Invalid selected source SHA256")
        config = _text(row["config_id"], "configuration ID")
        source_id = _text(row["source_id"], "source ID")
        source_time = _time(row["source_time"], profile.nt, "Source time")
        measurement_id, writer_id = (config, source_id), (config, source_time)
        _require(measurement_id not in seen_ids, "Duplicate configuration/source selection")
        _require(writer_id not in seen_writer_sources, "Duplicate writer source; a changed source ID does not create a measurement")
        file = parse_cf_filename(PurePosixPath(path).name)
        row_identity = {**identity, "config_id": config, "source_id": source_id, "source_time": source_time}
        _validate_identity(row_identity, file, selected_key, profile)
        record = store.find_file(root_id, path)
        _require(record.root_id == root_id and record.path == path, "Shared store returned a different source identity")
        _require(record.sha256 == row_hash, "Selected original hash differs from database")
        _require(_integer(record.id) and record.id > 0 and record.id not in seen_files, "The same stored source file was selected twice")
        if store.schema_version == "lqcd.opaque-chunks.v2":
            if root_id not in source_rows:
                source = store.source_provenance(root_id)
                _require(type(source) is dict and source.get("root_id") == root_id
                         and type(source.get("path")) is str, "Missing validated source provenance")
                _reject_aggregate_path(source["path"])
                source_rows[root_id] = source
            source = dict(source_rows[root_id])
        else:
            source = None
        decoded = store.decode(record.id, profile=PICKLE_PROFILE)
        adapted = adapt_cf(decoded, filename=file.basename, profile=profile, array_key=array_key,
                           identity=row_identity, allowed_momenta=allowed_momenta, component=component,
                           stage="configuration" if stage == "raw-measurement" else stage,
                           allowed_operators=spec.get("allowed_operators"))
        adapted = replace(adapted, provenance={**adapted.provenance, "payload_sha256": record.sha256})
        observations.append(adapted)
        lineage.append({"file_id": record.id, "root_id": root_id, "path": path,
                        "sha256": record.sha256, "config_id": config, "source_id": source_id,
                        "source_time": source_time, "decode": dict(decoded.metadata),
                        "absolute_lattice_times": adapted.absolute_times.tolist(),
                        "source_provenance": source})
        seen_files.add(record.id)
        seen_ids.add(measurement_id)
        seen_writer_sources.add(writer_id)
    channel = to_measurement_channel(observations, time_action=spec["time_action"])
    provenance = {
        **channel.provenance, "selection_version": LEGACY_SELECTION_VERSION,
        "selection_sha256": hashlib.sha256(canonical).hexdigest(),
        "database_config_sha256": store.config_sha256, "database_schema": store.schema_version,
        "database_hash_verified": store.database_hash_verified,
        "stage": stage, "transformations": [], "files": lineage,
        "writer_evidence": {"version": profile.writer_evidence.version, "source": profile.writer_evidence.source},
        "convention_evidence": profile.writer_evidence.version + ": " + profile.writer_evidence.source,
        "source_provenance_status": "validated-v2-source-table" if source_rows else "not-available-in-v1",
    }
    return replace(channel, provenance=provenance)


@dataclass(frozen=True)
class AggregateLayout:
    """A validated declaration, not authenticated or scientifically loaded data."""

    config_ids: tuple[str, ...]
    keys: tuple[str, ...]
    source_groups: tuple
    scientific_loading_supported: bool = field(default=False, init=False)


def validate_aggregate_membership(arrays: Mapping, *, nt: int, exact_members,
                                  expected_keys) -> AggregateLayout:
    """Check per-key row membership, common order, source groups and shape.

    exact_members[key][row] is {config_id, sources: [{source_id, source_time,
    sha256}, ...]}. Every key needs its own explicit complete declaration.
    No declaration can prove the legacy writer actually followed that order.
    """
    _require(_integer(nt) and 0 < nt <= DEFAULT_LIMITS.max_dimension,
             "NT must be a positive integer within the numeric decoder extent limit")
    keys = _sequence(expected_keys, "expected key")
    _require(bool(keys) and len(keys) <= DEFAULT_LIMITS.max_arrays,
             "Declare every expected key within the numeric decoder array limit")
    for key in keys:
        _text(key, "expected key")
    _require(len(keys) == len(set(keys)), "Expected keys must be unique")
    _require(isinstance(arrays, Mapping) and set(arrays) == set(keys), "Aggregate key set is incomplete or unexpected")
    _require(type(exact_members) is dict and set(exact_members) == set(keys),
             "Exact per-key memberships are required; counts and directory names are insufficient")
    common = None
    for key in keys:
        rows = _sequence(exact_members[key], "per-key member")
        _require(bool(rows) and len(rows) <= 100000, "Provide a nonempty bounded configuration order")
        groups = []
        seen_configs = set()
        for row in rows:
            _require(type(row) is dict and set(row) == {"config_id", "sources"}, "Malformed configuration member")
            config = _text(row["config_id"], "aggregate configuration ID")
            _require(config not in seen_configs, "Duplicate aggregate configuration ID")
            seen_configs.add(config)
            sources = _sequence(row["sources"], "source member")
            _require(bool(sources), "An output configuration needs exact source memberships")
            group = []
            seen_ids, seen_times = set(), set()
            for source in sources:
                _require(type(source) is dict and set(source) == {"source_id", "source_time", "sha256"},
                         "A source member needs an ID, time and original SHA256")
                sid = _text(source["source_id"], "aggregate source ID")
                time = _time(source["source_time"], nt, "Aggregate source time")
                sha = source["sha256"]
                _require(type(sha) is str and _SHA256.fullmatch(sha) is not None, "Invalid source SHA256")
                _require(sid not in seen_ids and time not in seen_times,
                         "Duplicate writer source; relabeling source IDs does not create an observation")
                seen_ids.add(sid)
                seen_times.add(time)
                group.append((sid, time, sha))
            groups.append((config, tuple(group)))
        groups = tuple(groups)
        if common is None:
            common = groups
        else:
            _require(groups == common, "Per-key configuration/source memberships or their order disagree")
        array = arrays[key]
        _require(type(array) is np.ndarray and array.dtype.kind in "fiu"
                 and array.shape == (2, len(groups) * nt), "Aggregate array length disagrees with exact per-key memberships")
        _require(np.all(np.isfinite(array)), "Non-finite aggregate array")
    return AggregateLayout(tuple(group[0] for group in common), keys, common)
