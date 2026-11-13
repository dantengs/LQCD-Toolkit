# Versioned correlator store and analysis selections

The container stores exact original or newly produced numeric-serialization
bytes. A successful byte-preservation check is separate from a scientific
interpretation. Analysis opens stores read-only and never migrates or deletes
source data.

## SQLite byte schema

Supported versions are `lqcd.opaque-chunks.v1` and
`lqcd.opaque-chunks.v2`. The generated writer produces v1; the reader preserves
version identity rather than silently changing a database.

```sql
info(version TEXT, config_sha256 TEXT);
files(id INTEGER PRIMARY KEY, root_id TEXT, path TEXT, meta TEXT,
      sha256 TEXT, byte_length INTEGER);
chunks(file_id INTEGER, seq INTEGER, payload BLOB,
       PRIMARY KEY(file_id, seq));
-- v2 additionally records source provenance:
sources(root_id TEXT PRIMARY KEY, path TEXT, origin TEXT, storage_host TEXT);
```

There must be exactly one compatible info row. Each file occurrence has its own
root-relative path, metadata, length and SHA256. Numeric/channel keys are not
unique filesystem identities. Chunk sequence starts at zero with no gaps;
chunks are nonempty and at most 262144 bytes. Concatenation must match the
recorded length and hash. Empty originals have no chunks. No pickle is executed
during storage or inspection.

The v2 source table must match its configuration exactly. Source root paths,
origin and storage host are provenance strings; the reader does not contact
those machines or open original source roots. The public repository distributes
no actual private source configuration. The generated v1 store uses an explicit
`lqcd.generated-store.v1` origin and must not be passed to a deletion/migration
program.

## Seal and readback

Adjacent `config.json`, `schema.json`, `format.json`, `manifest.jsonl` and
`sealed.json` describe the store. Config SHA256 uses canonical ASCII JSON with
sorted keys and compact separators, without a final newline. `sealed.json`
contains size/SHA256 entries for `data.sqlite` and the four metadata files.

The default reader requires a seal and validates the schema/config relationship,
small metadata component hashes and DB size. Every returned original is always
verified by ordered-chunk length/SHA256. A full database hash check is explicit
because it is a potentially large sequential read. This reader does not claim
to verify an external transport archive or remote deletion lifecycle.

The producer validates every generated numeric serialization, reads back every
stored payload and validates every scientific selection before writing the
seal. Partial output is retained without a valid seal. Test-only unsealed reads
must never be used to consume a running migration.

## Numeric profiles

`savedata.pickle.numeric.v1` is an inert opcode interpreter for bounded
dictionaries of numeric ndarrays. NumPy reconstruction symbols are declarative
tokens; serialized `GLOBAL`, `REDUCE` or `BUILD` never invokes a supplied
callable. Unsupported objects, cycles, external buffers, persistent IDs and
unbounded shapes are rejected. This is intentionally narrower than general
pickle compatibility.

`savedata.table-text.v1` reads labeled real/imaginary pairs with contiguous time
rows. It records `%.6e` rounding rather than claiming binary precision.
`numpy.npy.numeric.v1` and `numpy.npz.numeric.v1` use numeric-only loading and
bounded header/shape checks. NPZ currently requires the explicit `data` and
`coefficients` member contract; decompression validates actual size, stream end,
CRC and trailing bytes, not just ZIP directory declarations.

Decoder success establishes numeric content, not axes or sample identities.
Generic NPY/NPZ aggregates without exact membership are inspectable but are not
automatically promoted to raw measurements.

## Selection contract

`lqcd.selection.v1` binds one literal channel to a database config hash, an
explicit numeric profile, array key, storage axes, component and time convention.
Identity includes ensemble, NT, kind, ordered literal masses, endpoint operators,
integer momenta, confirmation status and C3 current/T when applicable. Each row
names root/path/hash, configuration ID, source ID and source time.

The per-measurement adapter checks basename fields and array keys. Two-point
CPU-reference keys are `sink_source`; three-point keys are current labels.
Ordered masses are retained; physical flavor names are never inferred from
numbers. Historical delimiter-free momentum tokens are resolved against an
explicit allowed momentum table and ambiguous tokens are rejected.

Source, sink and physical initial time are distinct concepts in the sequential
contraction. `initial-forward` identifies physical C3 tau directly. Historical
base-forward/raw-lattice/sink-backward transformations are explicit and recorded;
joint analysis additionally verifies the physical time profile. No automatic
sign, conjugation, real/imaginary selection, momentum average or mass sorting is
applied. Duplicate per-configuration/source-time copies are not independent
measurements. Aggregates need exact members and derivation metadata before
scientific use.

## Explicit CF manifest dispatch

The CF adapter is newly implemented format compatibility code in this repository. It reuses the single shared v1/v2 store reader, decoder, selection object, and statistical/fitting pipeline; historical writer scripts and source archives are not included.

`lqcd.legacy-cf.selection.v1` is accepted by the same `load_channel` entry point
and therefore by the existing two-point and joint-fit commands when their
physical model/time requirements are met. It binds `config_sha256`,
`profile=savedata.pickle.numeric.v1`, a structured `writer_profile` with
`name`, `nt` and `writer_evidence={version,source}`, a shared
scientific `identity`, explicit `array_key`, `allowed_momenta`, `component`,
`stage`, `time_action=as-stored`, `time_convention` and a `measurements` list.
Each measurement supplies the same root/path/hash and configuration/source
fields as the standard selection. A selection uses one explicit literal array
key; Decimal labels are compared numerically without float conversion. Identity
additionally declares ordered
operator and mass slots and their physical role maps. See
[LEGACY_FORMATS.md](LEGACY_FORMATS.md) for those writer contracts.

The dispatcher reads verified bytes through the shared store, applies the
writer adapter and returns the shared `MeasurementChannel`. It does not roll
time, infer a flavor or clone any statistical machinery. The source-backward
three-point profile remains a stored-time observation; it is not automatically
promoted to the sequential model's physical initial-to-current time. Unknown
physical time/phase conventions still prevent that joint fit.
