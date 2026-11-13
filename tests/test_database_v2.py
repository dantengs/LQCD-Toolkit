"""Synthetic v1/v2 interoperability fixtures; all outputs are retained.

No remote source directory, production data, deletion, or cleanup fixture is used.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np

from fixture_store import canonical, create_store, new_artifact_dir
from lqcd_analysis.database import CHUNK_BYTES, SCHEMA_VERSION, SUPPORTED_VERSIONS, CorrelatorStore, StoreError


V2 = "lqcd.opaque-chunks.v2"


def _seal(folder):
    """Rehash every preserved member after intentional synthetic mutations."""
    config = json.loads((folder / "config.json").read_bytes())
    digest = hashlib.sha256(canonical(config)).hexdigest()
    members = {}
    for name in ("data.sqlite", "config.json", "schema.json", "format.json", "manifest.jsonl"):
        data = (folder / name).read_bytes()
        members[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (folder / "sealed.json").write_bytes(canonical({"config_sha256": digest, "members": members}) + b"\n")


def _fixture(*, defaults=False, source_fields=False, config_edit=None, sources_edit=None,
             metadata_edit=None, database_edit=None, schema_version=V2, large=False):
    folder = new_artifact_dir("numeric-store-v2")
    config = {
        "fixture": "generic-synthetic", "output": "/synthetic-fixture/control",
        "snapshot_mode": "observed-stable", "sources_quiescent": False, "delete_after_verify": False,
        "filename_patterns": ["*.npy", "*.bin"],
        "roots": [{"id": "source_a", "path": "/synthetic-fixture/input-a"},
                  {"id": "source_b", "path": "/synthetic-fixture/input-b"}],
    }
    if not defaults:
        config.update(source_origins={"source_a": "synthetic-origin-a", "source_b": "synthetic-origin-b"},
                      storage_host="synthetic-storage")
    sources = [(r["id"], r["path"], config.get("source_origins", {}).get(r["id"], "unspecified"),
                config.get("storage_host", "unspecified")) for r in config["roots"]]
    original_sources = {row[0]: dict(zip(("root_id", "path", "origin", "storage_host"), row)) for row in sources}
    if config_edit:
        config_edit(config)
    if sources_edit:
        sources = sources_edit(sources)
    digest = hashlib.sha256(canonical(config)).hexdigest()
    array = np.arange(100000 if large else 12, dtype=">f4").reshape(2, -1)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    payload = stream.getvalue()
    database = folder / "data.sqlite"
    manifest = []
    con = sqlite3.connect(database)
    try:
        con.execute("PRAGMA journal_mode=MEMORY")
        # Sources deliberately has no PK: malformed duplicate-source fixtures
        # must be rejected by the reader itself, not just by this test schema.
        con.executescript(
            "CREATE TABLE info(version TEXT,config_sha256 TEXT); "
            "CREATE TABLE sources(root_id TEXT,path TEXT,origin TEXT,storage_host TEXT); "
            "CREATE TABLE files(id INTEGER PRIMARY KEY,root_id TEXT,path TEXT,meta TEXT,sha256 TEXT,byte_length INTEGER); "
            "CREATE TABLE chunks(file_id INTEGER,seq INTEGER,payload BLOB);"
        )
        con.execute("INSERT INTO info VALUES (?,?)", (V2, digest))
        con.executemany("INSERT INTO sources VALUES (?,?,?,?)", sources)
        for fid, root_id in enumerate(("source_a", "source_b"), 1):
            metadata = {"root_id": root_id, "path": "records/sample.npy", "basename": "sample.npy",
                        "identity": {"dev": "0", "ino": str(fid), "mode": 33188, "nlink": 1,
                                     "size": len(payload), "mtime_ns": 0, "ctime_ns": 0},
                        "provenance": {"classification": "synthetic", "physics_decoded": False}}
            if source_fields:
                source = original_sources[root_id]
                metadata.update(origin=source["origin"], storage_host=source["storage_host"], root_path=source["path"])
            if metadata_edit:
                metadata_edit(metadata)
            sha = hashlib.sha256(payload).hexdigest()
            con.execute("INSERT INTO files VALUES (?,?,?,?,?,?)", (fid, root_id, "records/sample.npy",
                        canonical(metadata).decode("ascii"), sha, len(payload)))
            for seq, start in enumerate(range(0, len(payload), CHUNK_BYTES)):
                con.execute("INSERT INTO chunks VALUES (?,?,?)", (fid, seq, payload[start:start + CHUNK_BYTES]))
            manifest.append({"id": fid, "metadata": metadata, "sha256": sha, "byte_length": len(payload)})
        if database_edit:
            database_edit(con)
        con.commit()
    finally:
        con.close()
    for name, value in (("config.json", config), ("schema.json", {"version": schema_version}),
                        ("format.json", {"fixture": "generic-synthetic", "decoder": None})):
        (folder / name).write_bytes(canonical(value) + b"\n")
    (folder / "manifest.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in manifest))
    _seal(folder)
    return folder, database, digest, array, payload


class DatabaseV2Tests(unittest.TestCase):
    def assert_invalid(self, **kwargs):
        _, database, _, _, _ = _fixture(**kwargs)
        with self.assertRaises(StoreError):
            CorrelatorStore(database, verify_database_hash=True)

    def test_v1_default_writer_contract_is_unchanged(self):
        self.assertEqual(SCHEMA_VERSION, "lqcd.opaque-chunks.v1")
        self.assertEqual(SUPPORTED_VERSIONS, {SCHEMA_VERSION, V2})
        folder = new_artifact_dir("numeric-store-v1-compatibility")
        database, digest, _ = create_store(folder, [("synthetic", "sample.bin", b"unchanged synthetic bytes")])
        with CorrelatorStore(database, expected_config_sha256=digest, verify_database_hash=True) as store:
            self.assertEqual(store.schema_version, SCHEMA_VERSION)
            self.assertEqual(store.summary()["schema"], SCHEMA_VERSION)
            self.assertEqual(store.read_original(1), b"unchanged synthetic bytes")
            with self.assertRaises(StoreError):
                store.source_provenance("synthetic")

    def test_numeric_v2_preserves_bytes_dtype_and_distinct_source_occurrences(self):
        folder, database, digest, array, payload = _fixture(source_fields=True)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
        with CorrelatorStore(database, expected_config_sha256=digest, verify_database_hash=True) as store:
            self.assertEqual(store.schema_version, V2)
            self.assertEqual(store.summary()["schema"], V2)
            self.assertEqual(store.summary()["file_count"], 2)
            self.assertTrue(store.database_hash_verified)
            files = list(store.iter_files(limit=10))
            self.assertEqual([record.root_id for record in files], ["source_a", "source_b"])
            self.assertEqual(files[0].path, files[1].path)
            self.assertEqual(files[0].sha256, files[1].sha256)
            self.assertNotEqual(files[0].id, files[1].id)
            for record in files:
                self.assertEqual(store.read_original(record.id), payload)
                result = store.decode(record.id, profile="numpy.npy.numeric.v1")
                np.testing.assert_array_equal(result.arrays["data"], array)
                self.assertEqual(result.arrays["data"].dtype.str, ">f4")
                self.assertEqual(store.find_file(record.root_id, record.path).id, record.id)
            source = store.source_provenance("source_a")
            self.assertEqual(source, {"root_id": "source_a", "path": "/synthetic-fixture/input-a",
                                      "origin": "synthetic-origin-a", "storage_host": "synthetic-storage"})
            source["origin"] = "changed-local-copy"
            self.assertEqual(store.source_provenance("source_a")["origin"], "synthetic-origin-a")
            with self.assertRaises(StoreError):
                store.source_provenance("missing")
            with self.assertRaises(sqlite3.OperationalError):
                store._con.execute("UPDATE files SET path='changed'")
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_default_source_fields_and_offline_unsealed_fixture(self):
        _, database, digest, _, _ = _fixture(defaults=True)
        actual_open = Path.open

        def forbid_source_open(path, *args, **kwargs):
            if "synthetic-fixture" in path.parts:
                raise AssertionError("Original source paths must not be opened")
            return actual_open(path, *args, **kwargs)

        with patch.object(Path, "open", forbid_source_open):
            with CorrelatorStore(database, require_sealed=False, expected_config_sha256=digest) as store:
                source = store.source_provenance("source_b")
                self.assertEqual(source["origin"], "unspecified")
                self.assertEqual(source["storage_host"], "unspecified")
                self.assertFalse(store.summary()["sealed"])
                store.read_original(2)

    def test_source_membership_and_every_provenance_field_must_match(self):
        for index, wrong in ((0, "unknown_root"), (1, "/synthetic-fixture/wrong-root"),
                             (2, "wrong-origin"), (3, "wrong-storage")):
            def edit(rows, index=index, wrong=wrong):
                changed = list(rows[0])
                changed[index] = wrong
                return [tuple(changed), rows[1]]
            with self.subTest(field=index):
                self.assert_invalid(sources_edit=edit)
        for edit in (lambda rows: rows[:1], lambda rows: rows + [rows[0]], lambda rows: [rows[0], rows[0]]):
            self.assert_invalid(sources_edit=edit)

    def test_invalid_config_roots_origins_and_host_are_rejected_after_resealing(self):
        edits = (
            lambda c: c.update(roots=[]),
            lambda c: c["roots"].append(copy.deepcopy(c["roots"][0])),
            lambda c: c["roots"][0].update(extra="unrecognized"),
            lambda c: c["roots"][0].update(id="invalid root"),
            lambda c: c["roots"][0].update(path="relative/root"),
            lambda c: c["roots"][0].update(path="/synthetic-fixture/../other"),
            lambda c: c.update(source_origins={"source_a": "synthetic-origin-a"}),
            lambda c: c["source_origins"].update(extra="unexpected-origin"),
            lambda c: c["source_origins"].update(source_a=""),
            lambda c: c.update(storage_host=""),
        )
        for index, edit in enumerate(edits):
            with self.subTest(case=index):
                self.assert_invalid(config_edit=edit)

    def test_info_schema_and_config_binding_fail_closed(self):
        edits = (
            lambda con: con.execute("INSERT INTO info SELECT * FROM info"),
            lambda con: con.execute("UPDATE info SET version='lqcd.opaque-chunks.v99'"),
            lambda con: con.execute("UPDATE info SET config_sha256=?", ("0" * 64,)),
            lambda con: con.execute("ALTER TABLE sources ADD COLUMN extra TEXT"),
            lambda con: con.execute("ALTER TABLE sources RENAME COLUMN origin TO renamed_origin"),
            lambda con: con.execute("ALTER TABLE info ADD COLUMN extra TEXT"),
        )
        for index, edit in enumerate(edits):
            with self.subTest(case=index):
                self.assert_invalid(database_edit=edit)
        self.assert_invalid(schema_version=SCHEMA_VERSION)
        _, database, _, _, _ = _fixture()
        with self.assertRaises(StoreError):
            CorrelatorStore(database, expected_config_sha256="0" * 64)
        self.assert_invalid(database_edit=lambda con: con.execute("UPDATE files SET root_id='unknown_root'"))

    def test_all_file_metadata_is_bound_before_any_numeric_read(self):
        edits = (
            lambda m: m.update(root_id="unknown_root"),
            lambda m: m.update(path="different.npy"),
            lambda m: m.update(path="../outside.npy"),
            lambda m: m.update(basename="different.npy"),
            lambda m: m["identity"].update(size=0),
            lambda m: m.update(identity="not-an-object"),
            lambda m: m.update(origin="wrong-origin"),
            lambda m: m.update(storage_host="wrong-storage"),
            lambda m: m.update(root_path="/synthetic-fixture/wrong-root"),
        )
        for index, edit in enumerate(edits):
            with self.subTest(case=index):
                self.assert_invalid(source_fields=True, metadata_edit=edit)

    def test_chunk_sequence_length_type_and_hash_are_checked_even_with_fresh_seal(self):
        edits = (
            lambda con: con.execute("UPDATE chunks SET seq=9 WHERE file_id=1 AND seq=0"),
            lambda con: con.execute("INSERT INTO chunks SELECT * FROM chunks WHERE file_id=1 AND seq=0"),
            lambda con: con.execute("UPDATE chunks SET payload=? WHERE file_id=1 AND seq=0", (b"wrong bytes",)),
            lambda con: con.execute("UPDATE chunks SET payload=? WHERE file_id=1 AND seq=0", (b"",)),
            lambda con: con.execute("UPDATE chunks SET payload='not a blob' WHERE file_id=1 AND seq=0"),
            lambda con: con.execute("UPDATE files SET sha256=? WHERE id=1", ("0" * 64,)),
        )
        for index, edit in enumerate(edits):
            with self.subTest(case=index):
                _, database, _, _, _ = _fixture(database_edit=edit, large=True)
                with CorrelatorStore(database, verify_database_hash=True) as store:
                    with self.assertRaises(StoreError):
                        store.read_original(1)
        _, database, _, array, payload = _fixture(large=True)
        self.assertGreater(len(payload), CHUNK_BYTES)
        with CorrelatorStore(database, verify_database_hash=True) as store:
            self.assertEqual(store.read_original(1), payload)
            np.testing.assert_array_equal(store.decode(1, profile="numpy.npy.numeric.v1").arrays["data"], array)
        with CorrelatorStore(database, max_file_bytes=128) as store:
            with self.assertRaises(StoreError):
                store.read_original(1)

    def test_seal_and_config_tampering_remain_rejected(self):
        folder, database, _, _, _ = _fixture()
        config = json.loads((folder / "config.json").read_bytes())
        config["storage_host"] = "changed-storage"
        (folder / "config.json").write_bytes(canonical(config) + b"\n")
        with self.assertRaises(StoreError):
            CorrelatorStore(database)
        with self.assertRaises(StoreError):
            CorrelatorStore(database, require_sealed=False)
        _seal(folder)
        with self.assertRaises(StoreError):
            CorrelatorStore(database, verify_database_hash=True)


if __name__ == "__main__":
    unittest.main()
