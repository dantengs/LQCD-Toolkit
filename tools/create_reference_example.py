"""Create a tiny runnable mathematical example; never overwrite or delete.

Usage from the release directory:
    python tools/create_reference_example.py examples/my_reference
    python run.py check examples/my_reference/physics.json
    python run.py run examples/my_reference/physics.json

The explicit matrices exercise source generation, linear solves, sequential
sources, contractions and SQLite storage. They are not a lattice QCD action,
and there is no gauge field or production-backend validation in this example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_reference_example(output_dir):
    """Write physics.json and three 48x48 numeric NPY files in a new directory.

    All paths in physics.json are relative to that file, so moving the complete
    example directory preserves the input configuration. Generated run output
    is placed in its new 'output' child directory. One configuration suffices
    for this generation demonstration; analysis is explicitly disabled.
    """
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    nt = 4
    masses = {"initial": "0.400000", "final": "0.200000", "spectator": "0.100000"}
    operators = {}
    for role, mass in masses.items():
        # Positive-definite temporal hopping matrix, with no gauge links.
        # Ordering is [sink_spin, sink_color, t, z, y, x]; space has one site.
        temporal = ((2.0 + float(mass)) * np.eye(nt)
                    - 0.25 * np.roll(np.eye(nt), 1, axis=1)
                    - 0.25 * np.roll(np.eye(nt), -1, axis=1))
        matrix = np.kron(np.eye(12), temporal)
        filename = role + "_operator.npy"
        with (destination / filename).open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        operators[role] = filename
    configuration = {
        "version": "lqcd.easy.v1",
        "template": False,
        "scientific_mode": "mathematical-validation",
        "reference_description": (
            "Toy explicit positive-definite matrices only: spin/color identity "
            "times periodic temporal hopping. No gauge field, no production "
            "Dirac action, and no physical QCD result is represented."
        ),
        "ensemble": "mathematical-reference",
        "output": "output",
        "geometry": {"nt": nt, "nz": 1, "ny": 1, "nx": 1},
        "quark_masses": {
            "toy_initial": masses["initial"],
            "toy_final": masses["final"],
            "toy_spectator": masses["spectator"],
        },
        "process": {
            "initial_quark": "toy_initial",
            "final_quark": "toy_final",
            "spectator_quark": "toy_spectator",
            "initial_operator": "g5",
            "final_operator": "g5",
            "current": "Id",
            "momentum_initial": [0, 0, 0],
            "momentum_final": [0, 0, 0],
            "separation": 2,
        },
        "sources": {"kind": "point", "positions": [[0, 0, 0, 0]]},
        "configurations": [{"id": "1"}],
        "backend": {
            "profile": "explicit-matrix-reference",
            "operators": operators,
            "residual_tolerance": 1e-12,
        },
        "max_memory_bytes": 16 * 1024**2,
        "analysis": None,
    }
    path = destination / "physics.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(configuration, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a mathematical-only source-to-SQLite example in a new directory")
    parser.add_argument("output_dir", help="New directory; an existing destination is rejected")
    args = parser.parse_args(argv)
    try:
        path = create_reference_example(args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Example creation stopped: {exc}\n")
    print(json.dumps({"physics_configuration": str(path), "scientific_mode": "mathematical-validation",
                      "analysis": None, "note": "No gauge field or production physics is included."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
