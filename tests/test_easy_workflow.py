import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from fixture_store import new_artifact_dir
from lqcd_analysis.cli import main as cli_main
from lqcd_analysis.database import CorrelatorStore, StoreError
from lqcd_analysis.generation import Geometry, run_generation, correlator_records
from lqcd_analysis.selection import load_channel
from lqcd_analysis.workflow import plan_workflow, run_workflow
from lqcd_analysis.writer import create_generated_store


def easy_fixture(label="easy"):
    folder = new_artifact_dir(label)
    operators = {}
    for role, mass in (("initial", .4), ("final", .2), ("spectator", .1)):
        temporal = (2 + mass) * np.eye(4) - .25 * np.roll(np.eye(4), 1, axis=1) - .25 * np.roll(np.eye(4), -1, axis=1)
        path = folder / (role + ".npy")
        with path.open("xb") as stream:
            np.save(stream, np.kron(np.eye(12), temporal))
        operators[role] = path.name
    config = {"version": "lqcd.easy.v1", "template": False, "output": "output", "ensemble": "mathematical-fixture",
              "scientific_mode": "mathematical-validation", "geometry": {"nt": 4, "nz": 1, "ny": 1, "nx": 1},
              "quark_masses": {"c": "0.400000", "q": "0.200000", "s": "0.100000"},
              "process": {"initial_quark": "c", "final_quark": "q", "spectator_quark": "s", "initial_operator": "g5",
                          "final_operator": "g5", "current": "g4", "momentum_initial": [0, 0, 0], "momentum_final": [0, 0, 0], "separation": 2},
              "sources": {"kind": "point", "positions": [[0, 0, 0, 0], [0, 0, 0, 1]]},
              "configurations": [{"id": "1"}],
              "backend": {"profile": "explicit-matrix-reference", "operators": operators, "residual_tolerance": 1e-12},
              "max_memory_bytes": 16 * 1024**2, "analysis": None}
    path = folder / "physics.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return folder, path, config


def replace_configuration(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")


class EasyWorkflowTests(unittest.TestCase):
    def test_check_is_read_only_and_batches_physical_configuration(self):
        folder, path, _ = easy_fixture("easy-check")
        with patch("lqcd_analysis.workflow.run_generation") as producer:
            plan = plan_workflow(path)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(cli_main(["check", str(path)]), 0)
            producer.assert_not_called()
        self.assertEqual(len(plan["tasks"]), 2)
        self.assertEqual([t["config"]["source_id"] for t in plan["tasks"]], ["t0", "t1"])
        self.assertFalse((folder / "output").exists())
        self.assertIn("no-solver-started", buffer.getvalue())

    def test_one_command_reference_pipeline_to_sqlite_and_sidecars(self):
        folder, path, _ = easy_fixture("easy-reference-run")
        result = run_workflow(path)
        with CorrelatorStore(result["database"]["database"], verify_database_hash=True) as store:
            self.assertEqual(store.summary()["file_count"], 6)
            self.assertEqual(len(result["database"]["selections"]), 3)
            for filename in result["database"]["selections"]:
                selection = json.loads(Path(filename).read_text(encoding="utf-8"))
                channel = load_channel(store, selection)
                self.assertEqual(channel.values.shape, (2, 4))
                self.assertEqual(selection["component"], "complex")
        self.assertFalse(list((folder / "output").rglob("*.bin")))
        self.assertTrue((folder / "output" / "workflow.json").is_file())
        self.assertEqual(result["products"][0]["backend_status"], "explicit-operator-reference")
        with self.assertRaises(ValueError):
            run_workflow(path)
        # Keep a compact, reusable demonstration pointer for the user-facing guide.
        pointer = Path(__file__).resolve().parents[1] / "validation" / "latest_easy_reference.json"
        pointer.write_text(json.dumps({"physics_configuration": str(path), "workflow_result": str(folder / "output" / "workflow.json"),
                                       "database": result["database"]["database"], "mode": "mathematical validation only"}, indent=2), encoding="utf-8")

    def test_equal_initial_final_products_keep_distinct_role_selections(self):
        _, path, config = easy_fixture("easy-identical-sectors")
        config["process"]["final_quark"] = "c"
        config["backend"]["operators"]["final"] = config["backend"]["operators"]["initial"]
        replace_configuration(path, config)
        result = run_workflow(path)
        self.assertEqual(len(result["database"]["selections"]), 3)
        with CorrelatorStore(result["database"]["database"]) as store:
            for filename in result["database"]["selections"]:
                load_channel(store, json.loads(Path(filename).read_text(encoding="utf-8")))

    def test_z3_estimator_is_validated_and_forwarded(self):
        _, path, config = easy_fixture("easy-grid")
        config["sources"] = {"kind": "z3-grid", "positions": [[0, 0, 0, 0]], "base_seed": 19, "grid_spacing": [1, 1, 1]}
        replace_configuration(path, config)
        with self.assertRaisesRegex(ValueError, "estimator"):
            plan_workflow(path)
        config["grid_estimator"] = "zero-momentum-source-average"
        replace_configuration(path, config)
        plan = plan_workflow(path)
        self.assertEqual(plan["tasks"][0]["config"]["grid_estimator"], config["grid_estimator"])
        result = run_workflow(path)
        self.assertEqual(result["database"]["file_count"], 3)

    def test_number_aliases_and_wrong_operator_shape_fail_before_generation(self):
        folder, path, config = easy_fixture("easy-preflight")
        config["configurations"] = [{"id": "1"}, {"id": "01"}]
        replace_configuration(path, config)
        with self.assertRaisesRegex(ValueError, "canonical"):
            plan_workflow(path)
        config["configurations"] = [{"id": "1"}]
        wrong = folder / "wrong.npy"
        with wrong.open("xb") as output:
            np.save(output, np.eye(3))
        config["backend"]["operators"]["final"] = wrong.name
        replace_configuration(path, config)
        with patch("lqcd_analysis.workflow.run_generation") as producer:
            with self.assertRaisesRegex(ValueError, "shape"):
                run_workflow(path)
            producer.assert_not_called()
        self.assertFalse((folder / "output").exists())

    def test_invalid_analysis_is_rejected_before_any_producer(self):
        folder, path, config = easy_fixture("easy-analysis-preflight")
        config["analysis"] = {"type": "two-point", "role": "initial", "settings": {"max_nfev": 0}}
        replace_configuration(path, config)
        with patch("lqcd_analysis.workflow.run_generation") as producer:
            with self.assertRaisesRegex(ValueError, "max_nfev"):
                run_workflow(path)
            producer.assert_not_called()
        self.assertFalse((folder / "output").exists())

    def test_malformed_external_template_and_log_contract_fail_before_launch(self):
        folder, path, config = easy_fixture("easy-external-preflight")
        gauge = folder / "not-a-physical-gauge.fixture"
        gauge.write_bytes(b"preflight-test-only")
        template = folder / "bad.template"
        template.write_text("this_is_not_a_valid_profile 1\n", encoding="utf-8")
        config.update(scientific_mode="external-production-action")
        config["configurations"] = [{"id": "1", "gauge": gauge.name}]
        config["backend"] = {"profile": "xqcd.overlap-inverter-stdin.v1", "argv": [sys.executable],
                             "input_template": template.name, "action_evidence": "preflight-only fixture",
                             "residual_pattern": "[", "residual_definition": "fixture", "residual_evidence": "fixture",
                             "residual_tolerance": 1e-10, "expected_residual_count": 12}
        replace_configuration(path, config)
        with patch("lqcd_analysis.workflow.run_generation") as producer:
            with self.assertRaisesRegex(ValueError, "regular expression"):
                run_workflow(path)
            config["backend"]["residual_pattern"] = r"TEST ([0-9.e+-]+)"
            replace_configuration(path, config)
            with self.assertRaises(ValueError):
                run_workflow(path)
            producer.assert_not_called()
        self.assertFalse((folder / "output").exists())

    def test_scientific_sidecar_failure_leaves_unsealed_store(self):
        folder, path, _ = easy_fixture("easy-incomplete-store")
        plan = plan_workflow(path)
        product = run_generation(plan["tasks"][0]["config"], folder / "product")
        records = correlator_records(product)
        records[0]["metadata"]["sink_operator"] = "g4"
        output = folder / "bad-store"
        with self.assertRaises(StoreError):
            create_generated_store(output, records, {"validation": "intentional wrong label"})
        self.assertFalse((output / "sealed.json").exists())
        with self.assertRaises(StoreError):
            CorrelatorStore(output / "data.sqlite")


if __name__ == "__main__":
    unittest.main()
