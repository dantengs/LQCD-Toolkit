"""Write newly generated correlators directly to the compatible SQLite store.

This is a producer, not a migration or cleanup program. The source objects are
in-memory numeric arrays. No small correlator files are materialized or deleted.
All generated serialized bytes are decoded and verified before sealing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np

from .database import CHUNK_BYTES, CorrelatorStore, SCHEMA_VERSION, _relative_path
from .decoders import decode_numeric


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _write(path, data):
    with path.open("xb") as output:
        output.write(data)
        output.flush()


def _file_digest(path):
    digest, length = hashlib.sha256(), 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
            length += len(block)
    return {"size": length, "sha256": digest.hexdigest()}


def _numeric_bytes(arrays):
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("Generated record needs a nonempty numeric array dictionary")
    prepared, logical_bytes = {}, 0
    for key, value in arrays.items():
        data = np.asarray(value)
        if not isinstance(key, str) or not key or data.dtype.kind not in "fiu" or data.ndim != 2 or data.shape[0] != 2:
            raise ValueError("Generated correlators must be keyed numeric (real/imag,time) arrays")
        if not np.all(np.isfinite(data)):
            raise ValueError("Non-finite generated correlator")
        logical_bytes += data.nbytes
        if logical_bytes > 60 * 1024**2:
            raise ValueError("Generated record exceeds the small-correlator byte budget")
        prepared[key] = np.array(data, copy=True)
    # Serialization is of our plain numeric objects, never of user classes.
    payload = pickle.dumps(prepared, protocol=4)
    decoded = decode_numeric(payload, "savedata.pickle.numeric.v1")
    if set(decoded.arrays) != set(prepared):
        raise ValueError("Producer roundtrip label mismatch")
    for key, original in prepared.items():
        restored = decoded.arrays[key]
        if restored.dtype != original.dtype or restored.shape != original.shape or not np.array_equal(restored, original):
            raise ValueError("Producer numeric serialization roundtrip failed")
    return payload


def create_generated_store(output_directory, records, run_metadata):
    """Consume records with `path`, `arrays`, `metadata`; return DB and selections.

    Destination must be new. Partial output is retained after an error and has
    no valid seal. The same byte schema is used as migration, with a distinct
    `lqcd.generated-store.v1` origin in config/format metadata. This store must
    never be passed to the migration cleanup pipeline.
    """
    target = Path(output_directory).absolute()
    target.mkdir(parents=True, exist_ok=False)
    config = {"version": "lqcd.generated-store.v1", "origin": "generated-in-memory-numeric-correlators",
              "output": str(target), "delete_after_verify": False, "roots": [], "run_metadata": run_metadata}
    digest = hashlib.sha256(_canonical(config)).hexdigest()
    database = target / "data.sqlite"
    con = sqlite3.connect(database)
    manifests, selections = [], {}
    try:
        if con.execute("PRAGMA journal_mode=PERSIST").fetchone()[0].lower() != "persist":
            raise ValueError("Persistent journal mode is required")
        con.execute("PRAGMA synchronous=FULL")
        con.executescript("CREATE TABLE info(version TEXT, config_sha256 TEXT); CREATE TABLE files(id INTEGER PRIMARY KEY, root_id TEXT, path TEXT, meta TEXT, sha256 TEXT, byte_length INTEGER, UNIQUE(root_id,path)); CREATE TABLE chunks(file_id INTEGER, seq INTEGER, payload BLOB NOT NULL, PRIMARY KEY(file_id,seq), FOREIGN KEY(file_id) REFERENCES files(id));")
        con.execute("INSERT INTO info VALUES (?,?)", (SCHEMA_VERSION, digest))
        con.commit()
        for record in records:
            path = _relative_path(record["path"])
            metadata = dict(record["metadata"])
            root_id = record.get("root_id", metadata["ensemble"])
            payload = _numeric_bytes(record["arrays"])
            sha = hashlib.sha256(payload).hexdigest()
            stored_meta = {"root_id": root_id, "path": path, "basename": path.rsplit("/", 1)[-1],
                           "identity": {"size": len(payload)},
                           "provenance": {"classification": "newly-generated", "origin": "numeric-array-serialization",
                                          "physics_decoded": False, "generation": metadata}}
            with con:
                cursor = con.execute("INSERT INTO files(root_id,path,meta,sha256,byte_length) VALUES (?,?,?,?,?)",
                                     (root_id, path, _canonical(stored_meta).decode(), sha, len(payload)))
                for seq, start in enumerate(range(0, len(payload), CHUNK_BYTES)):
                    con.execute("INSERT INTO chunks VALUES (?,?,?)", (cursor.lastrowid, seq, payload[start:start + CHUNK_BYTES]))
            manifests.append({"id": cursor.lastrowid, "metadata": stored_meta, "sha256": sha, "byte_length": len(payload)})
            identity = {key: metadata[key] for key in ("kind", "ensemble", "nt", "ordered_masses", "source_operator", "sink_operator",
                                                       "momentum_initial", "momentum_final")}
            identity["physics_status"] = metadata.get("physics_status", "unconfirmed")
            if identity["kind"] == "3pt":
                identity.update(current=metadata["current"], separation=metadata["separation"])
            for key in record["arrays"]:
                # One sidecar per literal channel. No fit projection is chosen.
                role = path.split("/", 1)[0] if identity["kind"] == "2pt" and path.startswith(("initial/", "final/")) else ""
                group = hashlib.sha256(_canonical({"identity": identity, "key": key, "product_role": role})).hexdigest()[:16]
                if group not in selections:
                    conventions = metadata["time_convention"]
                    selections[group] = {"version": "lqcd.selection.v1", "config_sha256": digest,
                                         "identity": identity, "stage": "raw-measurement", "storage": "split-ri",
                                         "array_key": key, "profile": "savedata.pickle.numeric.v1", "component": "complex",
                                         "time_convention": conventions, "time_action": "as-stored",
                                         "convention_evidence": metadata.get("convention_evidence", metadata.get("writer_profile", "")),
                                         "allowed_momenta": metadata.get("allowed_momenta", [identity["momentum_initial"], identity["momentum_final"]]),
                                         "measurements": []}
                sidecar = selections[group]
                if sidecar["time_convention"] != metadata["time_convention"]:
                    raise ValueError("Mixed generation time conventions")
                sidecar["measurements"].append({"root_id": root_id, "path": path, "sha256": sha,
                                                "config_id": str(metadata["config_id"]),
                                                "source_id": metadata["source_id"], "source_time": metadata["source_time"]})
        if not manifests:
            raise ValueError("No generated correlators")
    finally:
        con.close()
    _write(target / "config.json", _canonical(config) + b"\n")
    _write(target / "schema.json", _canonical({"version": SCHEMA_VERSION, "origin": "lqcd.generated-store.v1"}) + b"\n")
    _write(target / "format.json", _canonical({"version": "lqcd.generated-store.v1", "profile": "savedata.pickle.numeric.v1",
                                              "source": "in-memory arrays; no filesystem originals or deletion authorization",
                                              "backend_validation": "See run_metadata and each record's generation provenance"}) + b"\n")
    _write(target / "manifest.jsonl", b"".join(_canonical(row) + b"\n" for row in manifests))
    # Read back every row and validate every scientific sidecar before sealing.
    from .selection import load_channel
    selection_dir = target / "selections"
    selection_dir.mkdir(exist_ok=False)
    selection_paths = []
    with CorrelatorStore(database, require_sealed=False, expected_config_sha256=digest) as store:
        for row in manifests:
            store.read_original(row["id"])
            store.decode(row["id"], profile="savedata.pickle.numeric.v1")
        for group, sidecar in selections.items():
            load_channel(store, sidecar)
        for group, sidecar in selections.items():
            path = selection_dir / (sidecar["identity"]["kind"] + "-" + group + ".json")
            _write(path, _canonical(sidecar) + b"\n")
            selection_paths.append(str(path))
    members = {}
    for name in ("data.sqlite", "config.json", "schema.json", "format.json", "manifest.jsonl"):
        members[name] = _file_digest(target / name)
    _write(target / "sealed.json", _canonical({"config_sha256": digest, "members": members}) + b"\n")
    with CorrelatorStore(database, verify_database_hash=True) as store:
        store.summary()
    return {"database": str(database), "config_sha256": digest, "file_count": len(manifests),
            "selections": selection_paths, "origin": "lqcd.generated-store.v1"}
