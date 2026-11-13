"""Bounded numeric decoders; no legacy writer or pickle callable is executed.

Profiles describe containers only, never measurement identity or time direction.
See ../review/DECODER_CONTRACT.md for supported grammar and resource limits.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import math
import pickletools
import re
import struct
import tokenize
import zipfile
import zlib

import numpy as np


class DecodeError(ValueError):
    """Invalid, unsupported, or over-budget payload; preserve it as opaque."""


@dataclass(frozen=True)
class Limits:
    max_payload_bytes: int = 64 * 1024 * 1024
    max_array_bytes: int = 64 * 1024 * 1024
    max_total_array_bytes: int = 64 * 1024 * 1024
    max_elements: int = 8 * 1024 * 1024
    max_dimension: int = 8 * 1024 * 1024
    max_ndim: int = 8
    max_arrays: int = 4096
    max_header_bytes: int = 16384
    max_label_chars: int = 4096
    max_text_lines: int = 65536
    max_text_line_chars: int = 262144
    max_stack: int = 4096
    max_memo: int = 65536
    max_opcodes: int = 1000000
    max_tuple_items: int = 32
    max_zip_ratio: float = 1000.0


DEFAULT_LIMITS = Limits()
PROFILES = (
    "savedata.table-text.v1",
    "numpy.npy.numeric.v1",
    "numpy.npz.numeric.v1",
    "savedata.pickle.numeric.v1",
)


@dataclass(frozen=True)
class DecodeResult:
    profile: str
    arrays: dict[str, np.ndarray]
    metadata: dict


class _Budget:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.bytes = 0
        self.count = 0

    def reserve(self, shape: tuple, dtype: np.dtype) -> int:
        nbytes = _shape_bytes(shape, dtype, self.limits)
        self.bytes += nbytes
        self.count += 1
        _require(self.bytes <= self.limits.max_total_array_bytes, "total array byte limit")
        _require(self.count <= self.limits.max_arrays, "array count limit")
        return nbytes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeError(message)


def _dtype(descr: object) -> np.dtype:
    # A string-only scalar numeric grammar excludes objects, records, subarrays,
    # strings, dates and dtype metadata before asking NumPy to parse anything.
    _require(type(descr) is str and len(descr) <= 8, "numeric dtype string required")
    _require(re.fullmatch(r"[<>=|]?[biufc][0-9]{1,2}", descr) is not None,
             "unsupported dtype descriptor")
    dtype = np.dtype(descr)
    _require(dtype.kind in "biufc" and not dtype.hasobject and
             dtype.fields is None and dtype.subdtype is None and
             0 < dtype.itemsize <= 32, "non-scalar or nonnumeric dtype")
    return dtype


def _shape_bytes(shape: object, dtype: np.dtype, limits: Limits) -> int:
    _require(type(shape) is tuple and len(shape) <= limits.max_ndim, "invalid shape/rank")
    for dim in shape:
        _require(type(dim) is int and 0 <= dim <= limits.max_dimension, "invalid dimension")
    count = math.prod(shape)
    _require(count <= limits.max_elements, "element count limit")
    nbytes = count * dtype.itemsize
    _require(nbytes <= limits.max_array_bytes, "array byte limit")
    return nbytes


def _label(value: object, limits: Limits) -> str:
    _require(type(value) is str and 0 < len(value) <= limits.max_label_chars,
             "invalid array label")
    _require(not any(ord(c) < 32 or ord(c) == 127 for c in value), "control character in label")
    return value


def _npy(payload: bytes, budget: _Budget) -> np.ndarray:
    limits = budget.limits
    _require(payload[:6] == b"\x93NUMPY" and len(payload) >= 10, "invalid NPY magic")
    version = tuple(payload[6:8])
    _require(version in ((1, 0), (2, 0), (3, 0)), "unsupported NPY version")
    length_size = 2 if version == (1, 0) else 4
    start = 8 + length_size
    _require(len(payload) >= start, "truncated NPY length")
    header_len = int.from_bytes(payload[8:start], "little")
    _require(0 < header_len <= limits.max_header_bytes and start + header_len <= len(payload),
             "invalid/oversized NPY header")
    syntax = ast.parse(payload[start:start + header_len].decode(
        "utf-8" if version == (3, 0) else "latin1").strip(), mode="eval")
    _require(isinstance(syntax.body, ast.Dict) and len(syntax.body.keys) == 3,
             "NPY header must have exactly three fields")
    header = ast.literal_eval(syntax)
    _require(type(header) is dict and set(header) == {"descr", "shape", "fortran_order"},
             "invalid NPY header keys")
    _require(type(header["fortran_order"]) is bool, "invalid NPY storage order")
    dtype = _dtype(header["descr"])
    nbytes = budget.reserve(header["shape"], dtype)
    _require(start + header_len + nbytes == len(payload), "NPY data length mismatch/trailing data")
    # Every allocation-relevant field was checked above. Never enable pickle.
    array = np.load(io.BytesIO(payload), allow_pickle=False,
                    max_header_size=limits.max_header_bytes)
    _require(type(array) is np.ndarray and array.dtype == dtype and
             array.shape == header["shape"], "NPY header disagreement")
    return array


def _npz(payload: bytes, budget: _Budget) -> dict[str, np.ndarray]:
    # Bound directory entry count BEFORE ZipFile builds its in-memory directory.
    _require(len(payload) >= 22 and payload[:4] == b"PK\x03\x04", "invalid NPZ magic")
    end = payload.rfind(b"PK\x05\x06", max(0, len(payload) - 65557))
    _require(end >= 0 and end + 22 <= len(payload), "missing ZIP directory")
    disk, cd_disk, disk_count, count, cd_size, cd_offset, comment_len = struct.unpack_from(
        "<4H2IH", payload, end + 4)
    _require(disk == cd_disk == 0 and disk_count == count == 2, "NPZ needs exactly two members")
    _require(comment_len == 0 and end + 22 == len(payload), "ZIP comments/trailing data unsupported")
    _require(cd_offset + cd_size == end, "ZIP64 or invalid directory bounds")
    # A forged EOCD count must not hide many entries in the actual directory.
    cursor = cd_offset
    for _ in range(count):
        _require(cursor + 46 <= end and payload[cursor:cursor + 4] == b"PK\x01\x02",
                 "invalid ZIP central directory")
        name_len, extra_len, entry_comment_len = struct.unpack_from("<3H", payload, cursor + 28)
        cursor += 46 + name_len + extra_len + entry_comment_len
        _require(cursor <= end, "invalid ZIP central entry bounds")
    _require(cursor == end, "ZIP member count disagreement")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        _require(len(infos) == 2 and len(set(names)) == 2 and
                 set(names) == {"data.npy", "coefficients.npy"}, "unexpected/duplicate ZIP member")
        total = 0
        spans = []
        for item in infos:
            _require(item.orig_filename == item.filename and not item.is_dir(), "unsafe ZIP member path")
            _require(item.flag_bits in (0, 0x800) and
                     item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                     "unsupported ZIP flags/compression (including encryption/data descriptors)")
            _require(0 <= item.file_size <= budget.limits.max_payload_bytes, "ZIP member size limit")
            total += item.file_size
            _require(total <= budget.limits.max_payload_bytes, "ZIP total uncompressed size limit")
            _require(item.file_size <= max(1, item.compress_size) * budget.limits.max_zip_ratio,
                     "ZIP compression ratio limit")
            spans.append(_zip_local_span(payload, item, cd_offset))
        # No duplicate/overlapping local records, hidden entries, or unaccounted
        # gaps are accepted. Data-descriptor streams are deliberately unsupported.
        cursor = 0
        for local_start, data_start, data_end in sorted(spans):
            _require(local_start == cursor, "ZIP local records overlap or contain hidden bytes")
            cursor = data_end
        _require(cursor == cd_offset, "ZIP unaccounted bytes before central directory")
        arrays = {}
        for item, (_, data_start, data_end) in zip(infos, spans):
            compressed = memoryview(payload)[data_start:data_end]
            if item.compress_type == zipfile.ZIP_STORED:
                _require(item.compress_size == item.file_size, "stored ZIP member length mismatch")
                raw = bytes(compressed)
            else:
                raw = _bounded_inflate(compressed, item.file_size)
            _require(len(raw) == item.file_size and zlib.crc32(raw) == item.CRC,
                     "ZIP actual length/CRC mismatch")
            arrays[item.filename[:-4]] = _npy(raw, budget)
        return arrays


def _zip_local_span(payload: bytes, item: zipfile.ZipInfo, directory_offset: int) -> tuple[int, int, int]:
    """Validate a local header independently of ZipExtFile's declared-size cap."""
    start = item.header_offset
    _require(0 <= start and start + 30 <= directory_offset, "ZIP local header bounds")
    signature, version, flags, method, _, _, crc, compressed_size, raw_size, name_len, extra_len = (
        struct.unpack_from("<4s5H3I2H", payload, start))
    _require(signature == b"PK\x03\x04" and version == item.extract_version and
             version in (10, 20, 45) and flags == item.flag_bits and
             method == item.compress_type and crc == item.CRC, "ZIP local/central header mismatch")
    name_start = start + 30
    extra_start = name_start + name_len
    data_start = extra_start + extra_len
    _require(data_start <= directory_offset and
             payload[name_start:extra_start] == item.filename.encode("ascii"), "ZIP local name/bounds mismatch")
    cursor = extra_start
    zip64 = None
    while cursor < data_start:
        _require(cursor + 4 <= data_start, "truncated ZIP local extra header")
        tag, size = struct.unpack_from("<HH", payload, cursor)
        cursor += 4
        _require(cursor + size <= data_start, "truncated ZIP local extra field")
        if tag == 1:
            _require(zip64 is None, "duplicate ZIP64 local extra field")
            zip64 = payload[cursor:cursor + size]
        cursor += size
    # np.savez writes local ZIP64 lengths even for small members. Interpret only
    # the two size fields, and compare them with bounded central-directory sizes.
    needs_raw = raw_size == 0xffffffff
    needs_compressed = compressed_size == 0xffffffff
    if needs_raw or needs_compressed:
        _require(version == 45 and zip64 is not None and
                 len(zip64) == 8 * (int(needs_raw) + int(needs_compressed)), "invalid local ZIP64 lengths")
        offset = 0
        if needs_raw:
            raw_size = struct.unpack_from("<Q", zip64, offset)[0]
            offset += 8
        if needs_compressed:
            compressed_size = struct.unpack_from("<Q", zip64, offset)[0]
    else:
        _require(zip64 is None, "unexpected local ZIP64 lengths")
    _require(raw_size == item.file_size and compressed_size == item.compress_size,
             "ZIP local/central size mismatch")
    data_end = data_start + compressed_size
    _require(data_end <= directory_offset, "ZIP compressed data bounds")
    return start, data_start, data_end


def _bounded_inflate(compressed: memoryview, expected_size: int) -> bytes:
    """Raw DEFLATE with bounded output on EVERY call, checking the real EOS.

    ZipExtFile truncates to a declared uncompressed size. That can conceal an
    expanding stream whose real tail exceeds the claim, so it is never used here.
    """
    inflater = zlib.decompressobj(-15)
    output = bytearray()
    position = 0
    pending = b""
    while True:
        if pending:
            chunk = pending
        elif position < len(compressed):
            next_position = min(position + 65536, len(compressed))
            chunk = compressed[position:next_position]
            position = next_position
        else:
            chunk = b""
        # One extra output byte detects dishonest size claims without allowing
        # an unbounded inflate or flush. Larger valid output is drained in chunks.
        block = inflater.decompress(chunk, min(65536, expected_size - len(output) + 1))
        output.extend(block)
        _require(len(output) <= expected_size, "ZIP actual inflated output exceeds declared size")
        pending = inflater.unconsumed_tail
        if inflater.eof:
            _require(not inflater.unused_data and not pending and position == len(compressed),
                     "ZIP compressed stream has trailing/concatenated data")
            _require(len(output) == expected_size, "ZIP inflated length mismatch")
            return bytes(output)
        _require(bool(chunk) or bool(block), "ZIP compressed stream is truncated/missing EOF")
        _require(not pending or len(pending) < len(chunk) or bool(block), "ZIP inflate made no progress")


_FLOAT = re.compile(r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf|nan)", re.I)


def _table(payload: bytes, budget: _Budget) -> dict[str, np.ndarray]:
    # UTF-8 is an explicit profile decision; alternate legacy encodings stay opaque.
    _require(payload.count(b"\n") + payload.count(b"\r") <= budget.limits.max_text_lines,
             "text line count limit")
    text = payload.decode("utf-8-sig")
    _require(not any(c in text for c in ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")),
             "unsupported text line separator")
    lines = text.splitlines()
    _require(all(len(line) <= budget.limits.max_text_line_chars for line in lines), "text line length limit")
    headers = [(i, line) for i, line in enumerate(lines)
               if line.split(None, 1) and line.split(None, 1)[0] in ("Gamma", "ga3ma")]
    _require(len(headers) == 1, "exactly one Gamma/ga3ma header required")
    index, header_line = headers[0]
    columns = header_line.split(None, 2 * budget.limits.max_arrays + 1)
    _require(len(columns) >= 3 and (len(columns) - 1) % 2 == 0, "unpaired table labels")
    labels = []
    for real, imag in zip(columns[1::2], columns[2::2]):
        _require(real.endswith("_real") and imag.endswith("_imag") and
                 real[:-5] == imag[:-5], "real/imag labels must be adjacent matching pairs")
        label = real[:-5]
        if columns[0] == "Gamma":
            _require(label.startswith("(") and label.endswith(")"), "Gamma labels require writer parentheses")
            label = label[1:-1]
        labels.append(_label(label, budget.limits))
    _require(len(labels) == len(set(labels)) <= budget.limits.max_arrays, "duplicate/too many table labels")
    rows = [line for line in lines[index + 1:] if line.strip()]
    _require(len(rows) > 0, "table has no time rows")
    for label in labels:
        budget.reserve((2, len(rows)), np.dtype("f8"))
    matrix = np.empty((len(rows), len(labels) * 2), dtype=np.float64)
    for t, line in enumerate(rows):
        row = line.split(None, len(columns))
        _require(len(row) == len(columns) and row[0] == f"t{t}",
                 "table rows must be complete, ordered and continuous t0..tN")
        _require(all(_FLOAT.fullmatch(value) for value in row[1:]), "invalid table numeric token")
        values = [float(value) for value in row[1:]]
        _require(all(math.isfinite(v) or token.lower().lstrip("+-") in ("nan", "inf")
                     for v, token in zip(values, row[1:])), "table numeric overflow")
        matrix[t] = values
    return {label: matrix[:, 2*i:2*i + 2].T.copy() for i, label in enumerate(labels)}


@dataclass(frozen=True)
class _Symbol:
    name: str


@dataclass(eq=False)
class _Dtype:
    value: np.dtype
    built: bool = False


@dataclass(eq=False)
class _Array:
    value: np.ndarray | None = None


_GLOBALS = {
    ("numpy", "ndarray"): "ndarray",
    ("numpy", "dtype"): "dtype",
    ("numpy.core.multiarray", "_reconstruct"): "reconstruct",
    ("numpy._core.multiarray", "_reconstruct"): "reconstruct",
    ("numpy.core.numeric", "_frombuffer"): "frombuffer",
    ("numpy._core.numeric", "_frombuffer"): "frombuffer",
    ("_codecs", "encode"): "latin1",
    ("__builtin__", "bytes"): "empty_bytes",
    ("builtins", "bytes"): "empty_bytes",
}
_MARK = object()


class _PickleInterpreter:
    """Small inert virtual machine, deliberately NOT an Unpickler.

    GLOBAL is a symbolic label; REDUCE/BUILD match fixed data grammars. There is
    no import, attribute lookup, callable invocation, or Python object state hook.
    """
    def __init__(self, budget: _Budget):
        self.budget = budget
        self.limits = budget.limits
        self.stack = []
        self.memo = {}
        self.raw_bytes = 0

    def pop(self):
        _require(bool(self.stack), "pickle stack underflow")
        item = self.stack.pop()
        _require(item is not _MARK, "unexpected pickle MARK")
        return item

    def marked(self) -> list:
        index = len(self.stack) - 1
        while index >= 0 and self.stack[index] is not _MARK:
            index -= 1
        _require(index >= 0, "missing pickle MARK")
        items = self.stack[index + 1:]
        del self.stack[index:]
        return items

    def symbol(self, module, name):
        _require(type(module) is str and type(name) is str and (module, name) in _GLOBALS,
                 "unknown pickle global")
        return _Symbol(_GLOBALS[(module, name)])

    def array(self, raw, dtype, shape, order):
        _require(type(raw) in (bytes, bytearray), "array requires inline bytes")
        _require(type(dtype) is _Dtype and dtype.built, "array requires completed numeric dtype")
        _require(type(order) is str and order in ("C", "F"), "invalid array order")
        nbytes = self.budget.reserve(shape, dtype.value)
        _require(len(raw) == nbytes, "pickle array byte length mismatch")
        # Only our validated scalar numeric dtype reaches frombuffer. Copying
        # detaches the result from mutable VM state; no serialized function runs.
        return np.frombuffer(raw, dtype=dtype.value).reshape(shape, order=order).copy(order=order)

    def reduce(self, symbol, args):
        _require(type(symbol) is _Symbol and type(args) is tuple, "unsupported pickle REDUCE")
        if symbol.name == "empty_bytes":
            _require(len(args) == 0, "only literal empty bytes reconstruction supported")
            return b""
        if symbol.name == "latin1":
            _require(len(args) == 2 and type(args[0]) is str and
                     type(args[1]) is str and args[1] == "latin1", "unsupported bytes encoding")
            self.raw_bytes += len(args[0])
            _require(self.raw_bytes <= self.limits.max_payload_bytes, "decoded bytes budget")
            return args[0].encode("latin1")
        if symbol.name == "dtype":
            _require(len(args) == 3 and args[1] is False and args[2] is True, "unsupported dtype arguments")
            return _Dtype(_dtype(args[0]))
        if symbol.name == "reconstruct":
            _require(len(args) == 3 and type(args[0]) is _Symbol and args[0].name == "ndarray" and
                     type(args[1]) is tuple and len(args[1]) == 1 and
                     type(args[1][0]) is int and args[1][0] == 0 and
                     type(args[2]) is bytes and args[2] == b"b", "unsupported ndarray reconstruction")
            return _Array()
        if symbol.name == "frombuffer":
            _require(len(args) == 4, "unsupported frombuffer arguments")
            return _Array(self.array(*args))
        raise DecodeError("unsupported pickle reducer")

    def build(self, target, state):
        _require(type(state) is tuple, "unsupported pickle BUILD state")
        if type(target) is _Dtype:
            _require(not target.built and len(state) == 8 and type(state[0]) is int and state[0] == 3,
                     "unsupported dtype BUILD version/rebuild")
            version, endian, subarray, names, fields, size, alignment, flags = state
            _require(type(endian) is str and endian in ("<", ">", "=", "|") and
                     subarray is names is fields is None and
                     type(size) is int and size == -1 and
                     type(alignment) is int and alignment == -1 and
                     type(flags) is int and flags == 0, "unsupported structured/object dtype BUILD")
            _require(endian != "|" or target.value.itemsize == 1, "invalid dtype byte order")
            target.value = target.value.newbyteorder(endian)
            target.built = True
        elif type(target) is _Array:
            _require(target.value is None and len(state) == 5 and
                     type(state[0]) is int and state[0] == 1 and type(state[3]) is bool,
                     "unsupported ndarray BUILD/rebuild")
            _, shape, dtype, fortran, raw = state
            target.value = self.array(raw, dtype, shape, "F" if fortran else "C")
        else:
            raise DecodeError("BUILD target is not a numeric ndarray/dtype token")

    def setitems(self, items):
        _require(self.stack and type(self.stack[-1]) is dict and len(items) % 2 == 0,
                 "invalid pickle dictionary assignment")
        target = self.stack[-1]
        for key, value in zip(items[::2], items[1::2]):
            key = _label(key, self.limits)
            # This immediately excludes nested/cyclic containers and unfinished
            # reconstruction objects; only completed arrays enter dictionaries.
            _require(type(value) is _Array and value.value is not None, "dictionary values must be numeric arrays")
            _require(key not in target and len(target) < self.limits.max_arrays, "duplicate key/dictionary limit")
            target[key] = value

    def run(self, payload: bytes) -> dict[str, np.ndarray]:
        stream = io.BytesIO(payload)
        protocol = None
        frame_end = None
        for number, (opcode, arg, position) in enumerate(pickletools.genops(stream), 1):
            _require(number <= self.limits.max_opcodes, "pickle opcode limit")
            name = opcode.name
            after = stream.tell()
            if frame_end is not None:
                if position == frame_end:
                    frame_end = None
                else:
                    _require(position < frame_end and after <= frame_end and name != "FRAME",
                             "invalid pickle frame boundary")
            if number == 1:
                _require(name == "PROTO" and arg in (2, 3, 4, 5), "only pickle protocols 2..5 supported")
                protocol = arg
                continue
            _require(opcode.proto <= protocol, "opcode exceeds declared pickle protocol")
            if name == "FRAME":
                _require(protocol >= 4 and 0 < arg <= len(payload) - after, "invalid pickle frame length")
                frame_end = after + arg
            elif name == "STOP":
                _require(after == len(payload) and (frame_end is None or after == frame_end),
                         "trailing pickle bytes/incomplete frame")
                _require(len(self.stack) == 1 and type(self.stack[0]) is dict, "pickle root must be a numeric dictionary")
                return {key: value.value for key, value in self.stack[0].items()}
            elif name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4"):
                _require(type(arg) is int and arg.bit_length() <= 64, "pickle integer limit")
                self.stack.append(arg)
            elif name in ("BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"):
                _require(len(arg) <= self.limits.max_payload_bytes, "pickle string limit")
                self.stack.append(arg)
            elif name in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                _require(len(arg) <= self.limits.max_payload_bytes, "pickle bytes limit")
                self.stack.append(arg)
            elif name in ("NONE", "NEWTRUE", "NEWFALSE"):
                self.stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
            elif name == "EMPTY_DICT":
                self.stack.append({})
            elif name == "MARK":
                self.stack.append(_MARK)
            elif name in ("EMPTY_TUPLE", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
                if name == "TUPLE":
                    items = self.marked()
                elif name == "EMPTY_TUPLE":
                    items = []
                else:
                    items = [self.pop() for _ in range(int(name[-1]))][::-1]
                _require(len(items) <= self.limits.max_tuple_items, "pickle tuple limit")
                # No recursive containers are admitted inside tuples. Nested
                # tuples are only shapes/state and are depth-bounded below.
                _require(not any(type(item) is dict for item in items), "dictionary inside tuple unsupported")
                item_tuple = tuple(items)
                self.check_tuple_depth(item_tuple)
                self.stack.append(item_tuple)
            elif name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
                index = len(self.memo) if name == "MEMOIZE" else arg
                _require(type(index) is int and 0 <= index < self.limits.max_memo and
                         index not in self.memo and self.stack and self.stack[-1] is not _MARK,
                         "invalid/duplicate/oversized pickle memo")
                self.memo[index] = self.stack[-1]
            elif name in ("BINGET", "LONG_BINGET"):
                _require(arg in self.memo, "missing pickle memo entry")
                self.stack.append(self.memo[arg])
            elif name == "GLOBAL":
                module, symbol = arg.split(" ")
                self.stack.append(self.symbol(module, symbol))
            elif name == "STACK_GLOBAL":
                symbol, module = self.pop(), self.pop()
                self.stack.append(self.symbol(module, symbol))
            elif name == "REDUCE":
                args, symbol = self.pop(), self.pop()
                self.stack.append(self.reduce(symbol, args))
            elif name == "BUILD":
                state = self.pop()
                _require(bool(self.stack), "BUILD stack underflow")
                self.build(self.stack[-1], state)
            elif name == "SETITEM":
                value, key = self.pop(), self.pop()
                self.setitems([key, value])
            elif name == "SETITEMS":
                self.setitems(self.marked())
            else:
                # Includes all object constructors, persistent IDs, extension
                # codes, external buffers, executable reducers and unknown ops.
                raise DecodeError(f"unsupported pickle opcode: {name}")
            _require(len(self.stack) <= self.limits.max_stack, "pickle stack limit")
        raise DecodeError("pickle missing STOP")

    def check_tuple_depth(self, value):
        # Traversing repeated memo aliases can otherwise expand exponentially.
        # Every shape/state tuple is small; bound the complete visit count too.
        pending = [(value, 0)]
        visits = 0
        while pending:
            item, level = pending.pop()
            visits += 1
            _require(visits <= 256 and level <= 4, "pickle tuple complexity limit")
            if type(item) is tuple:
                pending.extend((child, level + 1) for child in item)


def decode_numeric(payload: bytes, profile: str, *, limits: Limits = DEFAULT_LIMITS) -> DecodeResult:
    """Decode an explicitly selected numeric container, or raise DecodeError.

    Does not guess profiles from filenames or data; does not read/write paths.
    Unknown formats should remain opaque in the caller's preservation store.
    """
    _require(type(payload) is bytes, "payload must be bytes")
    _require(type(profile) is str and profile in PROFILES, "unknown explicit decoder profile")
    _require(0 < len(payload) <= limits.max_payload_bytes, "payload byte limit")
    budget = _Budget(limits)
    try:
        if profile == "savedata.table-text.v1":
            arrays = _table(payload, budget)
            precision = "rounded %.6e export (7 significant decimal digits); parsed as float64"
        elif profile == "numpy.npy.numeric.v1":
            arrays = {"data": _npy(payload, budget)}
            precision = "serialized numeric dtype preserved"
        elif profile == "numpy.npz.numeric.v1":
            arrays = _npz(payload, budget)
            precision = "serialized numeric dtype preserved"
        else:
            arrays = _PickleInterpreter(budget).run(payload)
            precision = "serialized numeric dtype preserved"
        logical_bytes = sum(array.nbytes for array in arrays.values())
        _require(logical_bytes <= limits.max_total_array_bytes, "total returned array byte limit")
        for array in arrays.values():
            array.setflags(write=False)
        return DecodeResult(profile, arrays, {
            "precision": precision,
            "axes": "split RI [real, imag], time" if profile == "savedata.table-text.v1" else "not inferred",
            "transforms": [],
            "array_bytes": logical_bytes,
            "scientific_identity": "not inferred",
        })
    except DecodeError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, UnicodeError,
            SyntaxError, RecursionError, EOFError, OSError, struct.error,
            tokenize.TokenError, zipfile.BadZipFile, zlib.error, NotImplementedError) as exc:
        raise DecodeError(f"invalid or unsupported {profile} payload: {type(exc).__name__}") from exc
