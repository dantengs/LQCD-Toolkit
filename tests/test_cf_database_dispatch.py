import copy
import hashlib
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose

from fixture_store import create_store, new_artifact_dir
from lqcd_analysis.analysis import fit_two_point
from lqcd_analysis.database import CorrelatorStore
from lqcd_analysis.physics import c2_periodic
from lqcd_analysis.selection import load_channel


class CFDatabaseDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = new_artifact_dir("cf-database-dispatch")
        rng = np.random.default_rng(9342)
        nt = 16
        truth = c2_periodic(np.arange(nt), nt, [.31], [1.3])
        key, files, measurements = "0.1_g5_0.2_g5_000", [], []
        for cfg in map(str, range(1, 41)):
            data = truth + rng.normal(size=nt) * .002 * truth
            payload = pickle.dumps({key: np.stack([data, np.zeros(nt)])}, protocol=4)
            path = f"configuration/cf{cfg}_t0_0.100000_0.200000.bin"
            files.append(("synthetic-root", path, payload))
            measurements.append({"root_id": "synthetic-root", "path": path, "sha256": hashlib.sha256(payload).hexdigest(),
                                 "config_id": cfg, "source_id": "t0", "source_time": 0})
        cls.database, digest, _ = create_store(folder, files)
        cls.spec = {"version": "lqcd.legacy-cf.selection.v1", "config_sha256": digest,
                    "profile": "savedata.pickle.numeric.v1", "writer_profile": {"name": "two-point.forward.v1", "nt": nt, "writer_evidence": {"version": "synthetic.v1", "source": "Synthetic format specification"}},
                    "identity": {"kind": "2pt", "nt": nt, "ensemble": "synthetic", "physics_status": "synthetic-validation",
                                 "physics_evidence": "Synthetic source and operator routing", "ordered_masses": ["0.100000", "0.200000"],
                                 "mass_roles": ["spectator", "active"], "ordered_operators": ["g5", "g5"],
                                 "operator_roles": {"sink": 0, "source": 1}, "source_operator": "g5", "sink_operator": "g5",
                                 "momentum_initial": [0, 0, 0], "momentum_final": [0, 0, 0]},
                    "array_key": key, "allowed_momenta": [[0, 0, 0]], "component": "real", "stage": "synthetic-validation",
                    "time_action": "as-stored", "time_convention": "source-forward", "measurements": measurements}

    def test_database_manifest_bridges_without_copying_or_reordering_core(self):
        with CorrelatorStore(self.database, verify_database_hash=True) as store:
            channel = load_channel(store, self.spec)
            self.assertEqual(channel.values.shape, (40, 16))
            self.assertEqual(channel.config_ids, tuple(map(str, range(1, 41))))
            self.assertEqual(channel.provenance["database_schema"], "lqcd.opaque-chunks.v1")
            self.assertTrue(channel.provenance["database_hash_verified"])
            self.assertEqual(len(channel.provenance["files"]), 40)

    def test_existing_c2_workflow_consumes_cf_selection(self):
        settings = {"version": "lqcd.fit2.v1", "spectral_assumption": "positive-diagonal-periodic",
                    "configuration_order": list(map(str, range(1, 41))), "bin_size": 2, "window": list(range(2, 9)),
                    "svd_rcond": 1e-10, "model": {"states": 1, "initial_energies": [.3], "initial_overlaps": [1.2],
                                                  "energy_gap_bounds": [[.01, 2]], "overlap_bounds": [[.01, 10]]}}
        result = fit_two_point(self.database, self.spec, settings)
        self.assertEqual(result["status"], "validated-numerics")
        assert_allclose(result["parameters"], [.31, 1.3], atol=.002, rtol=0)

    def test_wrong_original_or_time_profile_fails_before_statistics(self):
        for field, value in (("config_sha256", "0" * 64), ("time_convention", "sink-backward")):
            spec = copy.deepcopy(self.spec)
            spec[field] = value
            with CorrelatorStore(self.database) as store, self.assertRaises(ValueError):
                load_channel(store, spec)
        spec = copy.deepcopy(self.spec)
        spec["measurements"][0]["sha256"] = "0" * 64
        with CorrelatorStore(self.database) as store, self.assertRaises(ValueError):
            load_channel(store, spec)


if __name__ == "__main__":
    unittest.main()
