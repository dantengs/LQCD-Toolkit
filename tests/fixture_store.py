"""Persistent test artifacts using the exact migration schema; never cleanup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


VALIDATION = Path(__file__).resolve().parents[1] / "validation"


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def new_artifact_dir(label):
    root = VALIDATION.resolve()
    target = (root / (label + "-" + uuid.uuid4().hex)).resolve()
    if root not in target.parents:
        raise ValueError("Fixture output must stay inside validation")
    target.mkdir(parents=True, exist_ok=False)
    return target


def create_store(directory, files):
    """files is [(root_id,root_relative_path,bytes)]; write a sealed fixture."""
    directory = Path(directory)
    database = directory / "data.sqlite"
    if database.exists():
        raise ValueError("Fixture destination already exists")
    roots = sorted({r for r, _, _ in files})
    config = {"fixture": True, "sources_quiescent": True, "delete_after_verify": False,
              "output": directory.as_posix(), "roots": [{"id": r, "path": "/synthetic-fixture/" + r} for r in roots]}
    digest = hashlib.sha256(canonical(config)).hexdigest()
    con = sqlite3.connect(database)
    try:
        # Keep journals in memory. No test creates and then deletes a journal.
        con.execute("PRAGMA journal_mode=MEMORY")
        con.executescript("CREATE TABLE info(version TEXT, config_sha256 TEXT); CREATE TABLE files(id INTEGER PRIMARY KEY, root_id TEXT, path TEXT, meta TEXT, sha256 TEXT, byte_length INTEGER, UNIQUE(root_id,path)); CREATE TABLE chunks(file_id INTEGER, seq INTEGER, payload BLOB NOT NULL, PRIMARY KEY(file_id,seq), FOREIGN KEY(file_id) REFERENCES files(id));")
        con.execute("INSERT INTO info VALUES (?,?)", ("lqcd.opaque-chunks.v1", digest))
        manifest = []
        for root_id, path, payload in files:
            sha = hashlib.sha256(payload).hexdigest()
            metadata = {"root_id": root_id, "path": path, "basename": path.rsplit("/", 1)[-1],
                        "identity": {"dev": "0", "ino": "0", "mode": 33188, "nlink": 1,
                                     "size": len(payload), "mtime_ns": 0, "ctime_ns": 0},
                        "provenance": {"classification": "fixture", "physics_decoded": False}}
            cursor = con.execute("INSERT INTO files(root_id,path,meta,sha256,byte_length) VALUES (?,?,?,?,?)",
                                 (root_id, path, canonical(metadata).decode(), sha, len(payload)))
            for seq, start in enumerate(range(0, len(payload), 262144)):
                con.execute("INSERT INTO chunks VALUES (?,?,?)", (cursor.lastrowid, seq, payload[start:start + 262144]))
            manifest.append({"id": cursor.lastrowid, "metadata": metadata, "sha256": sha, "byte_length": len(payload)})
        con.commit()
    finally:
        con.close()
    originals = {"config.json": canonical(config) + b"\n", "schema.json": canonical({"version": "lqcd.opaque-chunks.v1"}) + b"\n",
                 "format.json": canonical({"fixture": True, "payload": "original bytes"}) + b"\n",
                 "manifest.jsonl": b"".join(canonical(row) + b"\n" for row in manifest)}
    for name, data in originals.items():
        with (directory / name).open("xb") as output:
            output.write(data)
    originals["data.sqlite"] = database.read_bytes()
    members = {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in originals.items()}
    with (directory / "sealed.json").open("xb") as output:
        output.write(canonical({"config_sha256": digest, "members": members}) + b"\n")
    return database, digest, manifest
