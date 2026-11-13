"""Create deterministic, wholly synthetic legacy-format fixtures in a new folder."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def create(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(152347)
    records = []
    for stem, count, header in (("two_point", 70, "Gamma"), ("three_point", 16, "ga3ma")):
        arrays = {}
        for i in range(count):
            label = (f"operator{i:02}_operator{i:02}" if header == "Gamma" else f"current{i:02}")
            arrays[label] = rng.normal(size=(2, 64)) * np.exp(-np.arange(64)[None, :] / (10 + i % 5))
        payload = pickle.dumps(arrays, protocol=4)
        keys = sorted(arrays)
        labels = [f"({key})" if header == "Gamma" else key for key in keys]
        lines = ["Synthetic numeric format fixture; no research measurements.",
                 header + " " + " ".join(label + suffix for label in labels for suffix in ("_real", "_imag"))]
        for t in range(64):
            lines.append(f"t{t} " + " ".join(f"{arrays[key][ri,t]:.6e}" for key in keys for ri in (0, 1)))
        for extension, data in (("bin", payload), ("txt", ("\n".join(lines) + "\n").encode("ascii"))):
            name = stem + "." + extension
            with (destination / name).open("xb") as output:
                output.write(data)
            records.append({"local_name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    provenance = {"origin": "synthetic-format-fixtures", "seed": 152347, "files": records,
                  "purpose": "Parser/precision tests only; no physical ensemble, source or channel claim"}
    with (destination / "provenance.json").open("x", encoding="utf-8") as output:
        json.dump(provenance, output, indent=2)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    print(json.dumps(create(parser.parse_args().destination), indent=2))
