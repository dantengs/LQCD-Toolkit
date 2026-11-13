"""Run the published quick-start commands; retain every validation artifact."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from fixture_store import new_artifact_dir
from lqcd_analysis.database import CorrelatorStore
from lqcd_analysis.selection import load_channel


RELEASE = Path(__file__).resolve().parents[1]


class PublicExampleTests(unittest.TestCase):
    def test_published_commands_generate_sealed_sqlite_without_overwrite(self):
        artifact = new_artifact_dir("public-reference-example")
        example = artifact / "example"
        tool = RELEASE / "tools" / "create_reference_example.py"

        def command(*args):
            # Deliberately run outside the release/example directories: input
            # paths must be resolved against physics.json rather than cwd.
            return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=artifact,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace", timeout=60)

        made = command(tool, example)
        self.assertEqual(made.returncode, 0, made.stderr)
        physics = example / "physics.json"
        config = json.loads(physics.read_text(encoding="utf-8"))
        self.assertEqual(config["scientific_mode"], "mathematical-validation")
        self.assertIsNone(config["analysis"])
        self.assertNotIn("gauge", config["configurations"][0])
        for filename in config["backend"]["operators"].values():
            self.assertFalse(Path(filename).is_absolute())
            matrix = np.load(example / filename, allow_pickle=False)
            self.assertEqual(matrix.shape, (48, 48))
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)
        snapshots = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in example.iterdir()}

        checked = command(RELEASE / "run.py", "check", physics)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("no-solver-started", checked.stdout)
        self.assertFalse((example / "output").exists())

        ran = command(RELEASE / "run.py", "run", physics)
        self.assertEqual(ran.returncode, 0, ran.stderr)
        result = json.loads((example / "output" / "workflow.json").read_text(encoding="utf-8"))
        self.assertIsNone(result["analysis"])
        self.assertEqual(result["products"][0]["backend_status"], "explicit-operator-reference")
        with CorrelatorStore(result["database"]["database"], verify_database_hash=True) as store:
            self.assertEqual(store.summary()["file_count"], 3)
            for selection_path in result["database"]["selections"]:
                spec = json.loads(Path(selection_path).read_text(encoding="utf-8"))
                channel = load_channel(store, spec)
                self.assertEqual(channel.values.shape, (1, 4))
                self.assertTrue(np.all(np.isfinite(channel.values)))
                self.assertGreater(float(np.max(np.abs(channel.values))), 0)
        self.assertFalse(list((example / "output").rglob("*.bin")))

        # All ordinary and sequential solves preserve a true residual check.
        product_path = Path(result["products"][0]["artifacts"]["metadata"])
        product = json.loads(product_path.read_text(encoding="utf-8"))
        for name in ("solve_initial", "solve_final", "solve_spectator", "solve_sequential"):
            residuals = product["metadata"]["stages"][name]["relative_residuals"]
            self.assertEqual(len(residuals), 12)
            self.assertLessEqual(max(residuals), 1e-12)

        repeated_create = command(tool, example)
        self.assertNotEqual(repeated_create.returncode, 0)
        repeated_run = command(RELEASE / "run.py", "run", physics)
        self.assertNotEqual(repeated_run.returncode, 0)
        for name, digest in snapshots.items():
            self.assertEqual(hashlib.sha256((example / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
