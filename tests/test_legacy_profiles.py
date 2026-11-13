"""Synthetic writer-format tests; no real measurement or machine configuration."""
import copy
from decimal import Decimal
import hashlib
import pickle
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

from lqcd_analysis.decoders import DEFAULT_LIMITS, decode_numeric
from lqcd_analysis.database import StoredFile, StoreError
from lqcd_analysis.legacy_profiles import (
    LEGACY_SELECTION_VERSION, LegacyProfileError, PICKLE_PROFILE, WRITER_PROFILES, WriterEvidence,
    WriterProfile, adapt_cf, decode_cf, parse_array_key, parse_cf_filename,
    stored_to_absolute_times, to_measurement_channel, load_cf_channel, validate_aggregate_membership,
)
from lqcd_analysis.selection import MeasurementChannel


EVIDENCE = WriterEvidence("synthetic-writer.v1", "In-memory fixture specification")
MODES = [[0, 0, -1], [1, 0, 0], [0, 0, 0]]


def fixture(kind="2pt", profile_name=None):
    nt = 8
    if kind == "2pt":
        filename = "cf0007_t6_0.125000_0.250000.bin"
        key = "0.125_g5_0.25_g5_00-1"
        name = WRITER_PROFILES[0]
        masses, operators = ["0.125000", "0.250000"], ["g5", "g5"]
        roles = {"sink": 0, "source": 1}
    else:
        filename = "cf0007_t6_T3_0.125000_0.250000_0.500000.bin"
        key = "0.125_g4_0.25_g5_0.5_g5_00-1_100"
        name = profile_name or WRITER_PROFILES[1]
        masses, operators = ["0.125000", "0.250000", "0.500000"], ["g4", "g5", "g5"]
        roles = {"current": 0, "sink": 1, "source": 2}
    profile = WriterProfile(name, nt, EVIDENCE)
    values = np.stack([np.arange(nt, dtype=float), 10.0 + np.arange(nt)])
    payload = pickle.dumps({key: values}, protocol=4)
    identity = {
        "ensemble": "synthetic-ensemble", "physics_status": "synthetic-validation",
        "physics_evidence": "Synthetic role assignment, no physical result claimed",
        "nt": nt, "kind": kind, "config_id": "0007", "source_id": "source-six",
        "source_time": 6, "ordered_masses": masses,
        "mass_roles": ["active", "spectator"] if kind == "2pt" else ["first-active", "second-active", "spectator"],
        "ordered_operators": operators, "operator_roles": roles,
        "source_operator": "g5", "sink_operator": "g5",
        "momentum_initial": [0, 0, -1],
        "momentum_final": [0, 0, -1] if kind == "2pt" else [1, 0, 0],
    }
    if kind == "3pt":
        identity.update(current="g4", separation=3,
                        momentum_roles={"initial": 0, "final": 1})
    args = dict(filename=filename, profile=profile, array_key=key, identity=identity,
                allowed_momenta=MODES, component="complex", stage="synthetic-validation")
    return payload, values, args


def member(config, source_time=2, digest="a" * 64):
    return {"config_id": config, "sources": [
        {"source_id": f"source-{source_time}", "source_time": source_time, "sha256": digest}
    ]}


def observation(config, source_time, value, *, kind="2pt", profile_name=None, component="real", source_id=None,
                identity_updates=None):
    _, _, args = fixture(kind, profile_name)
    args["filename"] = args["filename"].replace("cf0007_t6", f"cf{config}_t{source_time}", 1)
    args["identity"].update(config_id=config, source_time=source_time,
                             source_id=source_id or f"source-{source_time}")
    args["identity"].update(identity_updates or {})
    args["component"] = component
    array = np.stack([np.full(8, value, dtype=float), np.full(8, -value, dtype=float)])
    return decode_cf(pickle.dumps({args["array_key"]: array}, protocol=4), **args)


class ToyStore:
    """A hash-checking in-memory stand-in for the existing readonly store API."""

    config_sha256 = "1" * 64
    database_hash_verified = False

    def __init__(self, files, *, schema_version="lqcd.opaque-chunks.v2"):
        self.schema_version = schema_version
        self.records, self.payloads = {}, {}
        self.decode_calls, self.source_calls = [], []
        self.source_path = "/synthetic-source/config"
        for file_id, (root, path, payload) in enumerate(files, 1):
            record = StoredFile(file_id, root, path, {}, hashlib.sha256(payload).hexdigest(), len(payload))
            self.records[root, path] = record
            self.payloads[file_id] = payload

    def find_file(self, root_id, path):
        try:
            return self.records[root_id, path]
        except KeyError as exc:
            raise StoreError("Synthetic source path is absent") from exc

    def decode(self, file_id, *, profile):
        self.decode_calls.append((file_id, profile))
        record = next(r for r in self.records.values() if r.id == file_id)
        payload = self.payloads[file_id]
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise StoreError("Synthetic stored payload hash mismatch")
        return decode_numeric(payload, profile)

    def source_provenance(self, root_id):
        self.source_calls.append(root_id)
        if self.schema_version != "lqcd.opaque-chunks.v2":
            raise StoreError("No source table in a v1 store")
        return {"root_id": root_id, "path": self.source_path,
                "origin": "synthetic-generated", "storage_host": "synthetic-store"}


def loader_fixture(*, kind="2pt", schema_version="lqcd.opaque-chunks.v2"):
    _, _, args = fixture(kind)
    files, rows = [], []
    for config, source_time, value in (("0007", 1, 10), ("0007", 5, 14),
                                       ("0008", 1, 30), ("0008", 5, 34)):
        name = args["filename"].replace("cf0007_t6", f"cf{config}_t{source_time}", 1)
        payload = pickle.dumps({args["array_key"]: np.stack([
            np.full(8, value, dtype=float), np.full(8, -value, dtype=float)])}, protocol=4)
        path = "config/" + name
        files.append(("synthetic", path, payload))
        rows.append({"root_id": "synthetic", "path": path,
                     "sha256": hashlib.sha256(payload).hexdigest(), "config_id": config,
                     "source_id": f"source-{source_time}", "source_time": source_time})
    common = {k: v for k, v in args["identity"].items() if k not in {"config_id", "source_id", "source_time"}}
    store = ToyStore(files, schema_version=schema_version)
    profile = args["profile"]
    spec = {
        "version": LEGACY_SELECTION_VERSION, "config_sha256": store.config_sha256,
        "profile": PICKLE_PROFILE,
        "writer_profile": {"name": profile.name, "nt": profile.nt,
                           "writer_evidence": {"version": EVIDENCE.version, "source": EVIDENCE.source}},
        "identity": common, "array_key": args["array_key"], "allowed_momenta": MODES,
        "component": "real", "time_action": "as-stored", "stage": "synthetic-validation",
        "measurements": rows,
    }
    return store, spec


class LegacyProfileTests(unittest.TestCase):
    def test_filename_preserves_identity_and_decimal_mass_equality(self):
        parsed = parse_cf_filename("cf0007_t6_0.125000_0.250000.bin")
        self.assertEqual(parsed.config_id, "0007")
        self.assertEqual(parsed.ordered_masses, (Decimal("0.125"), Decimal("0.25")))
        self.assertIsNone(parsed.separation)
        triple = parse_cf_filename("cf0007_t6_T3_0.125000_0.250000_0.500000.bin")
        self.assertEqual((triple.kind, triple.separation), ("3pt", 3))
        for bad in ("cf7_t6_T3_0.1_0.2.bin", "cf7_t6_0.1_0.2_0.3.bin",
                    "folder/cf7_t6_0.1_0.2.bin", "cf7_t6_0.0_0.2.bin",
                    "cf7_t6_-0.1_0.2.bin", "cf7_t6_nan_0.2.bin"):
            with self.subTest(bad=bad), self.assertRaises(LegacyProfileError):
                parse_cf_filename(bad)

    def test_positive_decimal_is_label_validation_not_a_physical_mass_cut(self):
        parsed = parse_cf_filename("cf01_t0_2.57e-316_0.25.bin")
        self.assertEqual(parsed.ordered_masses[0], Decimal("2.57e-316"))
        payload, _, args = fixture()
        args["identity"]["ordered_masses"][0] = "2.57e-316"
        with self.assertRaises(LegacyProfileError):
            decode_cf(payload, **args)

    def test_gamma_literals_and_negative_or_ambiguous_momentum(self):
        key = parse_array_key("0.1_g5.g1_0.2_sigma12_00-1", "2pt", MODES)
        self.assertEqual(key.ordered_operators, ("g5.g1", "sigma12"))
        self.assertEqual(key.momenta, ((0, 0, -1),))
        with self.assertRaises(LegacyProfileError):
            parse_array_key("0.1_g5_0.2_g5_1110", "2pt", [[1, 1, 10], [11, 1, 0]])
        with self.assertRaises(LegacyProfileError):
            parse_array_key("0.1_unknown_0.2_g5_000", "2pt", [[0, 0, 0]])
        custom = parse_array_key("0.1_custom(1)_0.2_g5_000", "2pt", [[0, 0, 0]],
                                 allowed_operators=["custom(1)", "g5"])
        self.assertEqual(custom.ordered_operators[0], "custom(1)")

    def test_time_maps_are_explicit_wrap_and_differ_by_separation(self):
        forward = WriterProfile(WRITER_PROFILES[0], 8, EVIDENCE)
        source = WriterProfile(WRITER_PROFILES[1], 8, EVIDENCE)
        sink = WriterProfile(WRITER_PROFILES[2], 8, EVIDENCE)
        assert_array_equal(stored_to_absolute_times(forward, 6), [6, 7, 0, 1, 2, 3, 4, 5])
        assert_array_equal(stored_to_absolute_times(source, 6, 3), [6, 5, 4, 3, 2, 1, 0, 7])
        assert_array_equal(stored_to_absolute_times(sink, 6, 3), [1, 0, 7, 6, 5, 4, 3, 2])
        # T=0 is valid axis arithmetic, without a physical contact-term claim.
        assert_array_equal(stored_to_absolute_times(source, 6, 0), stored_to_absolute_times(sink, 6, 0))
        self.assertFalse(stored_to_absolute_times(sink, 6, 3).flags.writeable)
        for value in (True, -1, 8, 2.0, None):
            with self.subTest(value=value), self.assertRaises(LegacyProfileError):
                stored_to_absolute_times(source, 6, value)
        with self.assertRaises(LegacyProfileError):
            stored_to_absolute_times(forward, True)
        with self.assertRaises(LegacyProfileError):
            stored_to_absolute_times(forward, 0, 1)

    def test_writer_profile_and_structured_evidence_are_mandatory(self):
        for name, nt, evidence in (("guess-from-name", 8, EVIDENCE),
                                   (WRITER_PROFILES[0], True, EVIDENCE),
                                   (WRITER_PROFILES[0], 0, EVIDENCE),
                                   (WRITER_PROFILES[0], 8, "old writer")):
            with self.subTest(name=name, nt=nt), self.assertRaises(LegacyProfileError):
                WriterProfile(name, nt, evidence)
        for version, source in (("", "fixture"), ("v1", "")):
            with self.assertRaises(LegacyProfileError):
                WriterEvidence(version, source)

    def test_oversized_time_extent_fails_before_any_time_array_allocation(self):
        with mock.patch("lqcd_analysis.legacy_profiles.np.arange",
                        side_effect=AssertionError("must reject before allocation")):
            with self.assertRaises(LegacyProfileError):
                WriterProfile(WRITER_PROFILES[0], DEFAULT_LIMITS.max_dimension + 1, EVIDENCE)
            with self.assertRaises(LegacyProfileError):
                validate_aggregate_membership({}, nt=DEFAULT_LIMITS.max_dimension + 1,
                                                exact_members={}, expected_keys=[])

    def test_core_decoder_is_reused_and_stored_values_are_never_reordered(self):
        for kind in ("2pt", "3pt"):
            payload, values, args = fixture(kind)
            with mock.patch("pickle.loads", side_effect=AssertionError("must not execute pickle")):
                adapted = decode_cf(payload, **args)
            assert_array_equal(adapted.values, values[0] + 1j * values[1])
            self.assertFalse(adapted.values.flags.writeable)
            self.assertEqual(adapted.provenance["time_action"], "as-stored")
            self.assertEqual(adapted.provenance["payload_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(adapted.config_id, "0007")

    def test_explicit_identity_roles_are_checked_without_guessing_slot_meaning(self):
        payload, _, args = fixture("3pt")
        adapted = decode_cf(payload, **args)
        self.assertEqual(adapted.identity["operator_roles"], {"current": 0, "sink": 1, "source": 2})
        changes = [
            {"config_id": "7"}, {"source_time": True}, {"nt": True},
            {"ordered_masses": ["0.125", "0.250", "0.501"]},
            {"operator_roles": {"current": 0, "sink": 0, "source": 2}},
            {"current": "g1"}, {"mass_roles": ["active", "active", "spectator"]},
            {"momentum_roles": {"initial": 0, "final": 0}},
            {"separation": True}, {"physics_status": "unconfirmed"}, {"physics_evidence": ""},
        ]
        for change in changes:
            altered = copy.deepcopy(args)
            altered["identity"].update(change)
            with self.subTest(change=change), self.assertRaises(LegacyProfileError):
                decode_cf(payload, **altered)
        for missing in ("ensemble", "source_id", "operator_roles", "mass_roles", "ordered_operators"):
            altered = copy.deepcopy(args)
            del altered["identity"][missing]
            with self.subTest(missing=missing), self.assertRaises(LegacyProfileError):
                decode_cf(payload, **altered)

    def test_actual_three_point_profile_is_not_chosen_from_filename(self):
        payload, values, args = fixture("3pt")
        first = decode_cf(payload, **args)
        args["profile"] = WriterProfile(WRITER_PROFILES[2], 8, EVIDENCE)
        second = decode_cf(payload, **args)
        assert_array_equal(first.values, second.values)
        assert_array_equal((second.absolute_times - first.absolute_times) % 8, np.full(8, 3))
        args["profile"] = WriterProfile(WRITER_PROFILES[0], 8, EVIDENCE)
        with self.assertRaises(LegacyProfileError):
            decode_cf(payload, **args)

    def test_every_cf_key_must_have_correct_shape_mass_identity_and_finite_values(self):
        payload, values, args = fixture()
        key = args["array_key"]
        bad_arrays = [
            {key: np.zeros((2, 16))}, {key: np.zeros((8, 2))},
            {key: np.full((2, 8), np.nan)},
            {key: values, "0.126_g1_0.25_g1_00-1": values},
            {key: values, "0.125000_g5_0.250000_g5_00-1": values},
            {key: values, "0.125_g1_0.25_g1_00-1": np.zeros((2, 7))},
        ]
        for arrays in bad_arrays:
            with self.subTest(keys=list(arrays)), self.assertRaises(LegacyProfileError):
                decode_cf(pickle.dumps(arrays, protocol=4), **args)
        decoded = decode_numeric(payload, PICKLE_PROFILE)
        for component, expected in (("real", values[0]), ("imag", values[1])):
            actual = adapt_cf(decoded, **{**args, "component": component})
            assert_array_equal(actual.values, expected)

    def test_aggregates_are_diagnostic_only_even_with_declared_members(self):
        payload, _, args = fixture()
        for stage in ("All", "Ave", "source-average", "concatenated"):
            with self.subTest(stage=stage), self.assertRaises(LegacyProfileError):
                decode_cf(payload, **{**args, "stage": stage})
        with self.assertRaises(LegacyProfileError):
            decode_cf(payload, **{**args, "filename": "2pt_t0_4_0.125000_0.250000.bin"})
        arrays = {"a": np.zeros((2, 16)), "b": np.ones((2, 16))}
        rows = [member("0007"), member("0008")]
        exact = {"a": copy.deepcopy(rows), "b": copy.deepcopy(rows)}
        layout = validate_aggregate_membership(arrays, nt=8, exact_members=exact, expected_keys=["a", "b"])
        self.assertEqual(layout.config_ids, ("0007", "0008"))
        self.assertFalse(layout.scientific_loading_supported)
        # Source averages remain one output configuration per row, never one
        # independent row per contributing source.
        for values in exact.values():
            values[0]["sources"].append({"source_id": "source-4", "source_time": 4, "sha256": "b" * 64})
        averaged = validate_aggregate_membership(arrays, nt=8, exact_members=exact, expected_keys=["a", "b"])
        self.assertEqual(len(averaged.config_ids), 2)

    def test_aggregate_memberships_reject_missing_keys_rows_order_and_relabeled_sources(self):
        arrays = {"a": np.zeros((2, 16)), "b": np.zeros((2, 16))}
        rows = [member("0007"), member("0008")]
        exact = {"a": copy.deepcopy(rows), "b": copy.deepcopy(rows)}
        cases = [None, {"a": rows}, {"a": rows, "b": rows[::-1]},
                 {"a": rows, "b": rows[:1]},
                 {"a": [member("0007"), member("0007")], "b": rows},
                 {"a": [{"config_id": None, "sources": []}], "b": rows}]
        duplicate = copy.deepcopy(exact)
        duplicate["a"][0]["sources"].append({"source_id": "different-label", "source_time": 2, "sha256": "a" * 64})
        cases.append(duplicate)
        for members in cases:
            with self.subTest(members=members), self.assertRaises(LegacyProfileError):
                validate_aggregate_membership(arrays, nt=8, exact_members=members, expected_keys=["a", "b"])
        for broken in ({"a": arrays["a"], "b": np.zeros((2, 8))}, {"a": arrays["a"]}):
            with self.assertRaises(LegacyProfileError):
                validate_aggregate_membership(broken, nt=8, exact_members=exact, expected_keys=["a", "b"])

    def test_bridge_reuses_shared_channel_and_equal_configuration_source_average(self):
        observations = [observation("0007", 1, 10), observation("0007", 5, 14),
                        observation("0008", 1, 30), observation("0008", 5, 34)]
        channel = to_measurement_channel(observations, time_action="as-stored")
        self.assertIsInstance(channel, MeasurementChannel)
        self.assertEqual(channel.config_ids, ("0007", "0007", "0008", "0008"))
        self.assertNotIn("config_id", channel.identity)
        self.assertNotIn("source_id", channel.identity)
        self.assertNotIn("source_time", channel.identity)
        assert_array_equal(channel.times, np.arange(8))
        assert_array_equal(channel.provenance["files"][0]["absolute_lattice_times"], [1, 2, 3, 4, 5, 6, 7, 0])
        assert_array_equal(channel.provenance["files"][1]["absolute_lattice_times"], [5, 6, 7, 0, 1, 2, 3, 4])
        samples = channel.config_samples()  # Calls the existing statistics core.
        self.assertEqual(samples.config_ids, ("0007", "0008"))
        assert_array_equal(samples.values, np.array([[12.0] * 8, [32.0] * 8]))
        # Unequal source counts retain one equal-weight row per configuration.
        unbalanced = to_measurement_channel(observations[:3], time_action="as-stored").config_samples()
        assert_array_equal(unbalanced.values, np.array([[12.0] * 8, [30.0] * 8]))
        # Equivalent decimal spelling does not split a physical channel.
        alternate = observation("0008", 1, 30,
                                identity_updates={"ordered_masses": ["0.125", "0.2500000"]})
        equivalent = to_measurement_channel([observations[0], alternate], time_action="as-stored")
        self.assertEqual(equivalent.identity["ordered_masses"], ["0.125000", "0.250000"])

    def test_bridge_rejects_repeated_or_relabeled_sources_and_single_configuration(self):
        first = observation("0007", 1, 10)
        second_config = observation("0008", 1, 30)
        relabeled = observation("0007", 1, 10, source_id="different-source-label")
        for observations in ([first, first, second_config], [first, relabeled, second_config],
                             [first], [first, observation("0007", 5, 14)]):
            with self.subTest(count=len(observations)), self.assertRaises(LegacyProfileError):
                to_measurement_channel(observations, time_action="as-stored")

    def test_bridge_requires_one_component_profile_physical_identity_and_explicit_time_action(self):
        first = observation("0007", 1, 10)
        second = observation("0008", 1, 30)
        with self.assertRaises(LegacyProfileError):
            to_measurement_channel([first, second], time_action="to-source-forward")
        with self.assertRaises(LegacyProfileError):
            to_measurement_channel([first, observation("0008", 1, 30, component="imag")], time_action="as-stored")
        source = observation("0007", 1, 10, kind="3pt", profile_name=WRITER_PROFILES[1])
        sink = observation("0008", 1, 30, kind="3pt", profile_name=WRITER_PROFILES[2])
        with self.assertRaises(LegacyProfileError):
            to_measurement_channel([source, sink], time_action="as-stored")
        different_channel = observation("0008", 1, 30,
                                        identity_updates={"ensemble": "another-synthetic-ensemble"})
        with self.assertRaises(LegacyProfileError):
            to_measurement_channel([first, different_channel], time_action="as-stored")
        changed = copy.deepcopy(second)
        changed.identity["physics_evidence"] = "Different physical interpretation"
        with self.assertRaises(LegacyProfileError):
            to_measurement_channel([first, changed], time_action="as-stored")

    def test_bridge_preserves_complex_data_and_shared_statistics_requires_projection(self):
        observations = [observation("0007", 1, 10, component="complex"),
                        observation("0008", 1, 30, component="complex")]
        channel = to_measurement_channel(observations, time_action="as-stored")
        self.assertEqual(channel.provenance["component"], "complex")
        self.assertTrue(np.iscomplexobj(channel.values))
        with self.assertRaisesRegex(ValueError, "project complex data explicitly"):
            channel.config_samples()

    def test_selection_loader_uses_shared_store_decoder_channel_and_v2_source_provenance(self):
        store, spec = loader_fixture()
        with mock.patch("pickle.loads", side_effect=AssertionError("must not execute pickle")):
            channel = load_cf_channel(store, spec)
        self.assertIsInstance(channel, MeasurementChannel)
        assert_array_equal(channel.config_samples().values, np.array([[12.0] * 8, [32.0] * 8]))
        self.assertEqual(store.decode_calls, [(i, PICKLE_PROFILE) for i in range(1, 5)])
        self.assertEqual(store.source_calls, ["synthetic"])
        provenance = channel.provenance
        self.assertEqual(provenance["selection_version"], LEGACY_SELECTION_VERSION)
        self.assertEqual(provenance["database_config_sha256"], store.config_sha256)
        self.assertEqual(provenance["database_schema"], "lqcd.opaque-chunks.v2")
        self.assertFalse(provenance["database_hash_verified"])
        self.assertEqual(provenance["output_time_convention"], "source-forward")
        self.assertEqual(provenance["transformations"], [])
        self.assertEqual(provenance["files"][0]["sha256"], spec["measurements"][0]["sha256"])
        self.assertEqual(provenance["files"][0]["source_provenance"]["origin"], "synthetic-generated")
        self.assertEqual(len(provenance["selection_sha256"]), 64)
        self.assertNotIn("config_id", channel.identity)

    def test_selection_loader_v1_and_three_point_time_conventions_remain_explicit(self):
        store, spec = loader_fixture(schema_version="lqcd.opaque-chunks.v1")
        channel = load_cf_channel(store, spec)
        self.assertEqual(store.source_calls, [])
        self.assertEqual(channel.provenance["source_provenance_status"], "not-available-in-v1")
        self.assertTrue(all(row["source_provenance"] is None for row in channel.provenance["files"]))
        for name, expected in ((WRITER_PROFILES[1], "source-backward"),
                               (WRITER_PROFILES[2], "sink-backward")):
            three_store, three_spec = loader_fixture(kind="3pt")
            three_spec["writer_profile"]["name"] = name
            with self.subTest(profile=name):
                result = load_cf_channel(three_store, three_spec)
                self.assertEqual(result.provenance["output_time_convention"], expected)
                self.assertNotEqual(result.provenance["output_time_convention"], "initial-forward")
                assert_array_equal(result.times, np.arange(8))

    def test_selection_loader_rejects_config_or_payload_hash_before_decode(self):
        for kind in ("config", "payload"):
            store, spec = loader_fixture()
            if kind == "config":
                spec["config_sha256"] = "2" * 64
            else:
                spec["measurements"][0]["sha256"] = "2" * 64
            with self.subTest(kind=kind), self.assertRaises(LegacyProfileError):
                load_cf_channel(store, spec)
            self.assertEqual(store.decode_calls, [])
        store, spec = loader_fixture()
        store.payloads[1] = b"altered stored bytes"
        with self.assertRaises(StoreError):
            load_cf_channel(store, spec)  # The shared store owns payload verification.

    def test_selection_loader_rejects_aggregate_paths_stages_and_missing_or_duplicate_members(self):
        for path in ("All/cf0007_t1_0.125000_0.250000.bin", "Ave/cf0007_t1_0.125000_0.250000.bin"):
            store, spec = loader_fixture()
            spec["measurements"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(LegacyProfileError):
                load_cf_channel(store, spec)
            self.assertEqual(store.decode_calls, [])
        store, spec = loader_fixture()
        store.source_path = "/synthetic-source/Ave"
        with self.assertRaises(LegacyProfileError):
            load_cf_channel(store, spec)
        for mode in ("stage", "missing", "duplicate", "renamed", "identity", "decoder"):
            store, spec = loader_fixture()
            if mode == "stage":
                spec.update(stage="All", exact_members={})
            elif mode == "missing":
                spec["measurements"] = None
            elif mode in {"duplicate", "renamed"}:
                row = copy.deepcopy(spec["measurements"][0])
                if mode == "renamed":
                    row["source_id"] = "a-different-label"
                spec["measurements"].append(row)
            elif mode == "identity":
                spec["measurements"][0]["config_id"] = "7"
            else:
                spec["profile"] = "guess-from-extension"
            with self.subTest(mode=mode), self.assertRaises(LegacyProfileError):
                load_cf_channel(store, spec)


if __name__ == "__main__":
    unittest.main()
