from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import sqlite3
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from lqcd_analysis.analysis import fit_two_point
from lqcd_analysis.cli import main as cli_main
from lqcd_analysis.database import CorrelatorStore, StoreError
from lqcd_analysis.physics import c2_periodic
from lqcd_analysis.selection import load_channel, parse_momentum_token
from fixture_store import create_store, new_artifact_dir


REAL = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "correlators"


def synthetic_selection(directory, *, nt=32, nconfig=48, pair="g5_g5"):
    rng = np.random.default_rng(82911)
    truth = c2_periodic(np.arange(nt), nt, [0.31], [1.3])
    scales = 0.003 * truth + 1e-8
    covariance = np.exp(-abs(np.arange(nt)[:, None] - np.arange(nt)[None, :]) / 2.0)
    correlated_noise = rng.multivariate_normal(np.zeros(nt), covariance, size=nconfig)
    files, selected = [], []
    for i in range(nconfig):
        config = str(2000 + 10 * i)
        for source in (0, 4):
            data = truth + scales * (correlated_noise[i] + 0.25 * rng.normal(size=nt))
            payload = pickle.dumps({pair: np.stack([data, np.zeros(nt)])}, protocol=4)
            name = f"configs/{config}/t{source}/2pt_{config}_t{source}_m0.100000_m0.200000_mom000.bin"
            files.append(("synthetic", name, payload))
            selected.append({"root_id": "synthetic", "path": name, "sha256": hashlib.sha256(payload).hexdigest(),
                             "config_id": config, "source_id": f"t{source}", "source_time": source})
    database, digest, _ = create_store(directory, files)
    spec = {"version": "lqcd.selection.v1", "config_sha256": digest,
            "identity": {"ensemble": "SYNTHETIC-format-and-fit-validation", "nt": nt, "kind": "2pt",
                         "ordered_masses": ["0.100000", "0.200000"], "source_operator": pair.split("_")[1],
                         "sink_operator": pair.split("_")[0], "momentum_initial": [0, 0, 0],
                         "momentum_final": [0, 0, 0], "physics_status": "unconfirmed"},
            "stage": "synthetic-validation", "time_convention": "source-forward", "time_action": "as-stored",
            "convention_evidence": "Synthetic c2_periodic fixture: no physical ensemble claim",
            "component": "real", "storage": "split-ri", "array_key": pair,
            "profile": "savedata.pickle.numeric.v1", "allowed_momenta": [[0, 0, 0]], "measurements": selected}
    settings = {"version": "lqcd.fit2.v1", "spectral_assumption": "positive-diagonal-periodic",
                "configuration_order": [str(2000 + 10 * i) for i in range(nconfig)], "bin_size": 2,
                "window": list(range(4, 12)), "svd_rcond": 1e-10, "max_nfev": 1000,
                "model": {"states": 1, "initial_energies": [0.3], "initial_overlaps": [1.2],
                          "energy_gap_bounds": [[0.01, 2.0]], "overlap_bounds": [[0.01, 10.0]]}}
    return database, spec, settings


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = new_artifact_dir("real-database")
        cls.provenance = json.loads((REAL / "provenance.json").read_text(encoding="utf-8"))
        inputs = [("read-only-sample", "samples/" + r["local_name"], (REAL / r["local_name"]).read_bytes())
                  for r in cls.provenance["files"]]
        cls.database, cls.digest, cls.manifest = create_store(cls.directory, inputs)

    def test_format_originals_survive_exact_database_roundtrip(self):
        with CorrelatorStore(self.database, verify_database_hash=True) as store:
            self.assertTrue(store.database_hash_verified)
            for item, record in zip(self.provenance["files"], store.iter_files(limit=10)):
                data = store.read_original(record.id)
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                self.assertEqual(data, (REAL / item["local_name"]).read_bytes())

    def test_format_database_decodes_both_point_counts(self):
        with CorrelatorStore(self.database) as store:
            two = store.decode(1, profile="savedata.pickle.numeric.v1")
            three = store.decode(3, profile="savedata.pickle.numeric.v1")
            self.assertEqual(len(two.arrays), 70)
            self.assertEqual(len(three.arrays), 16)
            for data in [*two.arrays.values(), *three.arrays.values()]:
                self.assertEqual(data.shape, (2, 64))

    def test_read_only_and_wrong_configuration(self):
        with CorrelatorStore(self.database) as store:
            with self.assertRaises(sqlite3.OperationalError):
                store._con.execute("DELETE FROM files")
        with self.assertRaises(StoreError):
            CorrelatorStore(self.database, expected_config_sha256="0" * 64)

    def test_missing_seal_is_rejected(self):
        directory = new_artifact_dir("unsealed")
        database = directory / "data.sqlite"
        with database.open("xb") as output:
            output.write(self.database.read_bytes())
        with self.assertRaises(StoreError):
            CorrelatorStore(database)

    def test_chunk_gap_and_payload_corruption_fail(self):
        for mode in ("sequence", "bytes"):
            directory = new_artifact_dir("corrupt-" + mode)
            database, _, _ = create_store(directory, [("fixture", "tiny.bin", b"abc")])
            con = sqlite3.connect(database)
            try:
                con.execute("PRAGMA journal_mode=MEMORY")
                if mode == "sequence":
                    con.execute("UPDATE chunks SET seq=4")
                else:
                    con.execute("UPDATE chunks SET payload=?", (b"abd",))
                con.commit()
            finally:
                con.close()
            with CorrelatorStore(database, require_sealed=False) as store:
                with self.assertRaises(StoreError):
                    store.read_original(1)
            with self.assertRaises(StoreError):
                CorrelatorStore(database, verify_database_hash=True)

    def test_empty_and_multichunk_originals(self):
        directory = new_artifact_dir("chunks")
        data = b"AB" * 180000
        database, _, _ = create_store(directory, [("fixture", "empty.bin", b""), ("fixture", "large.bin", data)])
        with CorrelatorStore(database, verify_database_hash=True) as store:
            self.assertEqual(store.read_original(1), b"")
            self.assertEqual(store.read_original(2), data)
            self.assertEqual(len(list(store.iter_files(path_glob="large*"))), 1)
        with CorrelatorStore(database, max_file_bytes=100) as store:
            with self.assertRaises(StoreError):
                store.read_original(2)


class SelectionAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = new_artifact_dir("synthetic-c2")
        cls.database, cls.selection, cls.settings = synthetic_selection(cls.directory)

    def load(self, selection):
        with CorrelatorStore(self.database) as store:
            return load_channel(store, selection)

    def test_explicit_source_averaging(self):
        channel = self.load(self.selection)
        averaged = channel.config_samples()
        self.assertEqual(channel.values.shape, (96, 32))
        self.assertEqual(averaged.values.shape, (48, 32))
        assert_allclose(averaged.values[0], channel.values[:2].mean(axis=0))

    def test_wrong_hash_identity_or_unknown_time_fails(self):
        for field, value in (("sha256", "0" * 64), ("config_id", "9999"), ("source_time", 5)):
            spec = copy.deepcopy(self.selection)
            spec["measurements"][0][field] = value
            with self.assertRaises(StoreError):
                self.load(spec)
        spec = copy.deepcopy(self.selection)
        spec["time_convention"] = "unknown"
        with self.assertRaises(StoreError):
            self.load(spec)

    def test_mass_operator_and_momentum_mismatch_fail(self):
        for field, value in (("ordered_masses", ["9.000000", "8.000000"]),
                             ("source_operator", "g4"), ("momentum_initial", [1, 0, 0])):
            spec = copy.deepcopy(self.selection)
            spec["identity"][field] = value
            with self.assertRaises(StoreError):
                self.load(spec)

    def test_duplicates_not_independent_samples(self):
        for rename_source in (False, True):
            spec = copy.deepcopy(self.selection)
            row = dict(spec["measurements"][0])
            if rename_source:
                row["source_id"] = "invented-replica"
            spec["measurements"].append(row)
            with self.assertRaises(StoreError):
                self.load(spec)

    def test_ambiguous_momentum_token_fails(self):
        self.assertEqual(parse_momentum_token("-1-11", [[-1, -1, 1], [1, 1, 1]]), (-1, -1, 1))
        with self.assertRaises(StoreError):
            parse_momentum_token("1110", [[1, 1, 10], [11, 1, 0]])

    def test_reverse_and_forward_sink_time_conventions(self):
        directory = new_artifact_dir("three-point-time")
        nt, source, separation = 8, 6, 3
        source_forward = np.arange(nt, dtype=float)
        raw = np.roll(source_forward, source)
        files, entries = [], []
        for root, direction in (("backward", -1), ("forward", 1)):
            stored = raw[(source + separation + direction * np.arange(nt)) % nt]
            payload = pickle.dumps({"g4": np.stack([stored, np.zeros(nt)])}, protocol=4)
            name = "configs/100/t6/3pt_100_t6_T3_m0.500000_m0.100000_g5_m0.200000_g5_mom000_100.bin"
            files.append((root, name, payload))
            entries.append({"root_id": root, "path": name, "sha256": hashlib.sha256(payload).hexdigest(),
                            "config_id": "100", "source_id": "t6", "source_time": 6})
        database, digest, _ = create_store(directory, files)
        spec = copy.deepcopy(self.selection)
        spec.update(config_sha256=digest, array_key="g4", allowed_momenta=[[0, 0, 0], [1, 0, 0]], time_action="to-source-forward")
        spec["identity"].update(kind="3pt", nt=nt, current="g4", separation=separation,
                                ordered_masses=["0.500000", "0.100000", "0.200000"], momentum_final=[1, 0, 0])
        with CorrelatorStore(database) as store:
            for row, convention in zip(entries, ("sink-backward", "sink-forward")):
                spec.update(measurements=[row], time_convention=convention)
                assert_array_equal(load_channel(store, spec).values[0], source_forward)

    def test_full_sqlite_to_jackknife_fit_recovers_known_parameters(self):
        result = fit_two_point(self.database, self.selection, self.settings, verify_database_hash=True)
        self.assertEqual(result["status"], "validated-numerics")
        self.assertEqual(result["measurement_count"], 96)
        self.assertEqual(result["configuration_count"], 48)
        self.assertEqual(len(result["bin_memberships"]), 24)
        self.assertEqual(result["covariance_rank"], 8)
        self.assertEqual(result["dof"], 6)
        assert_allclose(result["parameters"], [0.31, 1.3], rtol=0, atol=0.002)
        self.assertTrue(np.all(abs(np.array(result["parameters"]) - [0.31, 1.3]) < 5 * np.array(result["standard_errors"])))
        for name, value in (("selection.json", self.selection), ("fit.json", self.settings), ("result.json", result)):
            with (self.directory / name).open("x", encoding="utf-8") as output:
                json.dump(value, output, ensure_ascii=True, indent=2, allow_nan=False)
        self.assertTrue(result["provenance"]["database_hash_verified"])

    def test_non_diagonal_and_missing_chain_order_rejected(self):
        directory = new_artifact_dir("non-diagonal")
        database, spec, settings = synthetic_selection(directory, pair="g5_g4")
        with self.assertRaisesRegex(ValueError, "same literal operator"):
            fit_two_point(database, spec, settings)
        bad = copy.deepcopy(self.settings)
        bad["configuration_order"] = bad["configuration_order"][:-1]
        with self.assertRaises(ValueError):
            fit_two_point(self.database, self.selection, bad)


if __name__ == "__main__":
    unittest.main()
