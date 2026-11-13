"""Read the migration store without modifying it or executing serialized code.

Reads lqcd.opaque-chunks.v1 and v2; writers keep the v1 default.
Every payload returned to a decoder is length- and SHA256-verified. The
optional full seal check additionally hashes the complete database once.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Iterator


SCHEMA_VERSION = "lqcd.opaque-chunks.v1"
SUPPORTED_VERSIONS = frozenset((SCHEMA_VERSION, "lqcd.opaque-chunks.v2"))
CHUNK_BYTES = 262144
DEFAULT_MAX_FILE_BYTES = 64 * 1024**2
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class StoreError(ValueError):
    """Missing provenance, incompatible schema, or corrupt stored bytes."""


def _require(condition, message):
    if not condition:
        raise StoreError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _relative_path(path):
    _require(isinstance(path, str) and "\\" not in path and "\x00" not in path, "Invalid store-relative path")
    parts = path.split("/")
    _require(all(x not in ("", ".", "..") for x in parts), "Invalid store-relative path")
    return path


def _read_json(path, max_bytes=4 * 1024**2):
    _require(path.is_file() and not path.is_symlink(), f"Missing or linked provenance file: {path.name}")
    _require(path.stat().st_size <= max_bytes, f"Provenance file too large: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise StoreError(f"Invalid JSON provenance: {path.name}") from exc


@dataclass(frozen=True)
class StoredFile:
    id: int
    root_id: str
    path: str
    metadata: dict
    sha256: str
    byte_length: int

    @classmethod
    def from_row(cls, row):
        try:
            meta = json.loads(row["meta"])
            _require(isinstance(meta, dict), "File metadata must be an object")
            _relative_path(row["path"])
            _require(meta.get("root_id") == row["root_id"] and meta.get("path") == row["path"],
                     "File metadata disagrees with its identity")
            size = row["byte_length"]
            _require(isinstance(size, int) and size >= 0, "Invalid stored byte length")
            _require(isinstance(row["sha256"], str) and _SHA256.fullmatch(row["sha256"]), "Invalid payload SHA256")
            if "identity" in meta:
                _require(isinstance(meta["identity"], dict) and meta["identity"].get("size") == size,
                         "Metadata length disagreement")
            return cls(row["id"], row["root_id"], row["path"], meta, row["sha256"], size)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise StoreError("Malformed file record") from exc


class CorrelatorStore:
    """Read-only access to a sealed migration database.

    `require_sealed=False` exists for deliberately created test fixtures. It
    must not be used to read an in-progress migration. By default, config and
    schema seal hashes, DB size, config digest and the schema row are checked.
    `verify_database_hash=True` also checks the complete DB hash against the
    seal; this is a potentially large sequential read. Independently, each
    original payload is always checked before it is returned.
    """

    def __init__(self, path, *, max_file_bytes=DEFAULT_MAX_FILE_BYTES,
                 expected_config_sha256=None, require_sealed=True, verify_database_hash=False):
        original = Path(path).absolute()
        _require(original.is_file() and not original.is_symlink(), "Database must be an existing regular file")
        _require(isinstance(max_file_bytes, int) and 0 < max_file_bytes <= 1024**3, "Invalid file byte limit")
        self.path = original.resolve()
        self.max_file_bytes = max_file_bytes
        self.seal = None
        self._config = None
        self._schema_document = None
        self._sources = {}
        self.database_hash_verified = False
        if require_sealed:
            self.seal = _read_json(self.path.parent / "sealed.json")
            _require(isinstance(self.seal, dict), "Invalid seal document")
            members = self.seal.get("members", {})
            _require(isinstance(members, dict) and
                     set(members) == {"data.sqlite", "manifest.jsonl", "config.json", "schema.json", "format.json"} and
                     all(isinstance(entry, dict) for entry in members.values()),
                     "Unexpected seal membership")
            _require(members["data.sqlite"].get("size") == self.path.stat().st_size, "Database size differs from seal")
            for name in ("config.json", "schema.json", "format.json"):
                target = self.path.parent / name
                value = _read_json(target)
                _require(isinstance(value, dict), f"Invalid provenance document: {name}")
                data = target.read_bytes()
                _require(members[name] == {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                         f"Sealed component changed: {name}")
                if name == "config.json":
                    self._config = value
                    digest = hashlib.sha256(_canonical(value)).hexdigest()
                    _require(digest == self.seal.get("config_sha256"), "Config digest differs from seal")
                elif name == "schema.json":
                    self._schema_document = value
            if expected_config_sha256 is not None:
                _require(expected_config_sha256 == self.seal.get("config_sha256"), "Wrong migration configuration")
        self._con = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self._con.row_factory = sqlite3.Row
        try:
            self._con.execute("PRAGMA query_only=ON")
            self._con.execute("PRAGMA trusted_schema=OFF")
            info = self._con.execute("SELECT version, config_sha256 FROM info").fetchall()
            _require(len(info) == 1 and info[0]["version"] in SUPPORTED_VERSIONS, "Incompatible database schema")
            self.schema_version = info[0]["version"]
            self.config_sha256 = info[0]["config_sha256"]
            _require(isinstance(self.config_sha256, str) and _SHA256.fullmatch(self.config_sha256), "Invalid config SHA256")
            if self.seal:
                _require(self.config_sha256 == self.seal["config_sha256"], "Database and seal configuration disagree")
            if expected_config_sha256 is not None:
                _require(self.config_sha256 == expected_config_sha256, "Wrong migration configuration")
            # These are column contracts, not inferred alternative schemas.
            for table, expected in {
                "info": {"version", "config_sha256"},
                "files": {"id", "root_id", "path", "meta", "sha256", "byte_length"},
                "chunks": {"file_id", "seq", "payload"},
            }.items():
                actual = {r["name"] for r in self._con.execute(f"PRAGMA table_info({table})")}
                _require(actual == expected, f"Incompatible {table} columns")
            if self._schema_document is not None:
                _require(self._schema_document.get("version") == self.schema_version,
                         "Schema provenance differs from database version")
            if self.schema_version == "lqcd.opaque-chunks.v2":
                self._validate_sources()
            if verify_database_hash:
                self.verify_database_seal()
        except sqlite3.DatabaseError as exc:
            self._con.close()
            raise StoreError("Invalid database schema or contents") from exc
        except Exception:
            self._con.close()
            raise

    def _validate_sources(self):
        """Bind v2 source provenance to the immutable config without opening sources."""
        actual_columns = {r["name"] for r in self._con.execute("PRAGMA table_info(sources)")}
        _require(actual_columns == {"root_id", "path", "origin", "storage_host"}, "Incompatible sources columns")
        if self._config is None:
            # An unsealed v2 fixture may skip a seal, never its config binding.
            self._config = _read_json(self.path.parent / "config.json")
            self._schema_document = _read_json(self.path.parent / "schema.json")
        config = self._config
        _require(isinstance(config, dict) and isinstance(self._schema_document, dict),
                 "Invalid v2 provenance documents")
        _require(hashlib.sha256(_canonical(config)).hexdigest() == self.config_sha256,
                 "Source configuration digest differs from database")
        _require(self._schema_document.get("version") == self.schema_version,
                 "Schema provenance differs from database version")
        roots = config.get("roots")
        _require(isinstance(roots, list) and roots, "V2 requires explicit source roots")
        expected = {}
        for root in roots:
            _require(isinstance(root, dict) and set(root) == {"id", "path"}, "Source roots require exactly id and path")
            root_id, path = root["id"], root["path"]
            _require(isinstance(root_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", root_id) is not None,
                     "Invalid source root ID")
            _require(root_id not in expected, "Duplicate source root ID")
            # These are preserved POSIX source names, not local machine paths.
            _require(isinstance(path, str) and "\x00" not in path and
                     PurePosixPath(path).is_absolute() and str(PurePosixPath(path)) == path and
                     ".." not in PurePosixPath(path).parts, "Invalid preserved source root path")
            expected[root_id] = {"root_id": root_id, "path": path}
        origins = config.get("source_origins", {root_id: "unspecified" for root_id in expected})
        host = config.get("storage_host", "unspecified")
        _require(isinstance(origins, dict) and set(origins) == set(expected) and
                 all(isinstance(value, str) and value.strip() for value in origins.values()),
                 "Source origins must cover exactly the configured roots")
        _require(isinstance(host, str) and host.strip(), "Invalid source storage host")
        for root_id, source in expected.items():
            source.update(origin=origins[root_id], storage_host=host)
        rows = self._con.execute("SELECT root_id,path,origin,storage_host FROM sources ORDER BY root_id").fetchall()
        _require(len(rows) == len(expected), "Source table membership differs from configuration")
        seen = set()
        for row in rows:
            root_id = row["root_id"]
            _require(isinstance(root_id, str) and root_id in expected and root_id not in seen,
                     "Unknown or duplicate source table root")
            _require(dict(row) == expected[root_id], "Source provenance differs from configuration")
            seen.add(root_id)
        self._sources = expected
        # Check every file's provenance, including files not selected for decoding.
        # Numeric blobs stay lazy: length/sequence/hash are verified on every read.
        for row in self._con.execute("SELECT * FROM files ORDER BY id"):
            self._stored_file(row)

    def source_provenance(self, root_id):
        """Return a copy of a validated v2 source row; v1 has no source table."""
        _require(isinstance(root_id, str) and root_id in self._sources, "No validated source provenance for root")
        return dict(self._sources[root_id])

    def _stored_file(self, row):
        record = StoredFile.from_row(row)
        if self.schema_version == "lqcd.opaque-chunks.v2":
            _require(record.root_id in self._sources, "File references an unknown source root")
            meta = record.metadata
            _require(isinstance(meta.get("identity"), dict) and meta["identity"].get("size") == record.byte_length,
                     "V2 file identity/byte length mismatch")
            if "basename" in meta:
                _require(meta["basename"] == record.path.rsplit("/", 1)[-1], "File basename metadata mismatch")
            source = self._sources[record.root_id]
            # The basic v2 producer stores these fields in sources, not files.
            # If a producer additionally duplicates them at the metadata top
            # level, do not accept contradictions or reinterpret the raw blob.
            for field in ("origin", "storage_host"):
                if field in meta:
                    _require(meta[field] == source[field], "File source metadata differs from source table")
            if "root_path" in meta:
                _require(meta["root_path"] == source["path"], "File root path differs from source table")
        return record

    def close(self):
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def verify_database_seal(self):
        _require(self.seal is not None, "Full database verification requires a seal")
        digest, length = hashlib.sha256(), 0
        with self.path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
                length += len(chunk)
        _require(self.seal["members"]["data.sqlite"] == {"size": length, "sha256": digest.hexdigest()},
                 "Database hash differs from seal")
        self.database_hash_verified = True

    def summary(self):
        row = self._con.execute("SELECT COUNT(*), COALESCE(SUM(byte_length),0) FROM files").fetchone()
        return {"schema": self.schema_version, "config_sha256": self.config_sha256,
                "file_count": row[0], "original_bytes": row[1], "sealed": self.seal is not None,
                "database_hash_verified": self.database_hash_verified}

    def iter_files(self, *, root_id=None, path_glob=None, after_id=0, limit=100) -> Iterator[StoredFile]:
        _require(isinstance(limit, int) and 1 <= limit <= 10000, "Listing limit must be 1..10000")
        conditions, args = ["id > ?"], [after_id]
        if root_id is not None:
            conditions.append("root_id = ?")
            args.append(root_id)
        if path_glob is not None:
            conditions.append("path GLOB ?")
            args.append(path_glob)
        args.append(limit)
        sql = "SELECT * FROM files WHERE " + " AND ".join(conditions) + " ORDER BY id LIMIT ?"
        for row in self._con.execute(sql, args):
            yield self._stored_file(row)

    def file(self, file_id) -> StoredFile:
        _require(isinstance(file_id, int) and not isinstance(file_id, bool) and file_id > 0, "Invalid file ID")
        row = self._con.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
        _require(row is not None, f"Stored file not found: {file_id}")
        return self._stored_file(row)

    def find_file(self, root_id, path) -> StoredFile:
        _relative_path(path)
        row = self._con.execute("SELECT * FROM files WHERE root_id=? AND path=?", (root_id, path)).fetchone()
        _require(row is not None, "Selected source path is absent from the database")
        return self._stored_file(row)

    def read_original(self, file_id) -> bytes:
        record = self.file(file_id)
        _require(record.byte_length <= self.max_file_bytes, "Original exceeds configured read limit")
        output = bytearray()
        digest = hashlib.sha256()
        for expected_seq, row in enumerate(self._con.execute(
                "SELECT seq,payload FROM chunks WHERE file_id=? ORDER BY seq", (file_id,))):
            _require(type(row["seq"]) is int and row["seq"] == expected_seq, "Nonconsecutive payload chunk sequence")
            chunk = row["payload"]
            _require(isinstance(chunk, bytes) and 0 < len(chunk) <= CHUNK_BYTES, "Invalid payload chunk")
            _require(len(output) + len(chunk) <= record.byte_length, "Payload exceeds recorded length")
            digest.update(chunk)
            output.extend(chunk)
        _require(len(output) == record.byte_length, "Payload length mismatch")
        _require(digest.hexdigest() == record.sha256, "Payload SHA256 mismatch")
        return bytes(output)

    def decode(self, file_id, *, profile):
        from .decoders import decode_numeric
        return decode_numeric(self.read_original(file_id), profile)
