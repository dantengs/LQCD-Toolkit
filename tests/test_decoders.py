"""In-memory synthetic fixtures plus mandatory preserved real-format samples.

Run: python -m unittest discover -s organized/tests -p test_decoders.py -v
No temporary-directory fixture, automatic cleanup, or legacy script execution.
"""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import struct
import sys
import unittest
from unittest import mock
import warnings
import zipfile
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lqcd_analysis.decoders import DEFAULT_LIMITS, DecodeError, decode_numeric


PICKLE = "savedata.pickle.numeric.v1"
NPY = "numpy.npy.numeric.v1"
NPZ = "numpy.npz.numeric.v1"
TEXT = "savedata.table-text.v1"


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array)
    return stream.getvalue()


def zip_bytes(members, compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w", compression=compression) as archive:
            for name, contents in members:
                archive.writestr(name, contents)
    return stream.getvalue()


def raw_npy(header, data=b""):
    header = header.encode("ascii") + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def deflate_bytes(raw):
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(raw) + compressor.flush()


def encoded_zip(first_raw, first_encoded, first_declared_size=None):
    """A two-member ZIP fixture with explicitly supplied DEFLATE bytes."""
    local = bytearray()
    directory = bytearray()
    for name, raw, encoded, declared in (
        (b"data.npy", first_raw, first_encoded,
         len(first_raw) if first_declared_size is None else first_declared_size),
        (b"coefficients.npy", first_raw, deflate_bytes(first_raw), len(first_raw)),
    ):
        start = len(local)
        crc = zlib.crc32(raw)
        local.extend(struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 8, 0, 0,
                                 crc, len(encoded), declared, len(name), 0))
        local.extend(name)
        local.extend(encoded)
        directory.extend(struct.pack("<4s6H3I5H2I", b"PK\x01\x02", 20, 20, 0, 8, 0, 0,
                                     crc, len(encoded), declared, len(name), 0, 0, 0, 0, 0, start))
        directory.extend(name)
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 2, 2, len(directory), len(local), 0)
    return bytes(local + directory + end)


class _MaliciousReducer:
    def __reduce__(self):
        # Merely describing this reducer to pickle.dumps must never execute it.
        return os.system, ("echo DECODER_MUST_NEVER_EXECUTE_THIS",)


class TestNumericDecoders(unittest.TestCase):
    def rejects(self, payload, profile, **kwargs):
        with self.assertRaises(DecodeError):
            decode_numeric(payload, profile, **kwargs)

    def test_explicit_profile_and_payload_limits(self):
        data = npy_bytes(np.arange(3))
        self.rejects(data, ".npy")
        self.rejects(data, PICKLE)
        self.rejects(data, NPY, limits=replace(DEFAULT_LIMITS, max_payload_bytes=8))
        self.rejects(bytearray(data), NPY)
        self.rejects(b"", TEXT)

    def test_pickle_protocols_precisions_endian_order_and_shared_dtype(self):
        fixtures = {
            "float32": np.arange(12, dtype=np.float32).reshape(2, 6),
            "float64": np.arange(12, dtype=np.float64).reshape(2, 6),
            "big_endian": np.arange(12, dtype=">f8").reshape(2, 6),
            "complex": np.asfortranarray((np.arange(12).reshape(3, 4) + 2j).astype("complex128")),
            "boolean": np.array([True, False]),
            "integer": np.arange(7, dtype="int16"),
            "empty": np.empty((2, 0), dtype="float32"),
            "scalar_array": np.array(5, dtype="int64"),
            "noncontiguous": np.arange(20, dtype="float32")[::3],
        }
        for protocol in (2, 3, 4, 5):
            with self.subTest(protocol=protocol):
                result = decode_numeric(pickle.dumps(fixtures, protocol=protocol), PICKLE)
                self.assertEqual(list(result.arrays), list(fixtures))
                for key, original in fixtures.items():
                    actual = result.arrays[key]
                    np.testing.assert_array_equal(actual, original)
                    self.assertEqual(actual.dtype.str, original.dtype.str)
                    self.assertEqual(actual.shape, original.shape)
                    self.assertFalse(actual.flags.writeable)
                self.assertTrue(result.arrays["complex"].flags.f_contiguous)
                self.assertEqual(result.metadata["transforms"], [])

    def test_pickle_old_numpy_module_aliases(self):
        original = {"g": np.arange(8, dtype="float64").reshape(2, 4)}
        payload = pickle.dumps(original, protocol=3).replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
        np.testing.assert_array_equal(decode_numeric(payload, PICKLE).arrays["g"], original["g"])

    def test_pickle_readonly_protocol5_and_aliases(self):
        original = np.arange(8, dtype="float32").reshape(2, 4)
        original.setflags(write=False)
        result = decode_numeric(pickle.dumps({"a": original, "b": original}, protocol=5), PICKLE)
        np.testing.assert_array_equal(result.arrays["a"], original)
        self.assertIs(result.arrays["a"], result.arrays["b"])
        self.rejects(pickle.dumps({"a": original, "b": original}, protocol=5), PICKLE,
                     limits=replace(DEFAULT_LIMITS, max_total_array_bytes=40))

    def test_pickle_malicious_global_never_executes(self):
        payload = pickle.dumps({"evil": _MaliciousReducer()}, protocol=4)
        with mock.patch.object(os, "system") as system:
            self.rejects(payload, PICKLE)
            system.assert_not_called()
        self.rejects(b"\x80\x02cbuiltins\neval\nX\x03\x00\x00\x001+1\x85R.", PICKLE)

    def test_pickle_global_reduce_build_abuse(self):
        # A permitted symbolic global is not itself a permitted reducer.
        self.rejects(b"\x80\x02cnumpy\nndarray\n)R.", PICKLE)
        self.rejects(b"\x80\x02c__builtin__\nbytes\nK\xff\x85R.", PICKLE)
        # BUILD on a dictionary must not reach __setstate__ or dictionary updates.
        self.rejects(b"\x80\x02})b.", PICKLE)
        # A numeric dtype cannot acquire fields/metadata/object state.
        valid = pickle.dumps({"g": np.arange(4, dtype="f8")}, protocol=3)
        self.rejects(valid.replace(b"NNN", b"N}N", 1), PICKLE)
        self.rejects(valid.replace(b"f8", b"O8", 1), PICKLE)
        self.rejects(valid.replace(b"_reconstruct", b"_evil_______", 1), PICKLE)

    def test_pickle_cycles_and_non_array_values(self):
        cyclic = {}
        cyclic["cycle"] = cyclic
        self.rejects(pickle.dumps(cyclic, protocol=4), PICKLE)
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        self.rejects(pickle.dumps({"cycle": cyclic_list}, protocol=4), PICKLE)
        for value in (1, 1.5, [1, 2], {"a": 1}, "data", None):
            with self.subTest(value=value):
                self.rejects(pickle.dumps({"bad": value}, protocol=4), PICKLE)

    def test_pickle_objects_records_buffers_persistent_ids_extensions(self):
        for array in (np.array([object()], dtype=object), np.zeros(2, dtype=[("x", "f8")]),
                      np.array(["abc"]), np.array(["2024-01-01"], dtype="datetime64[D]")):
            self.rejects(pickle.dumps({"bad": array}, protocol=4), PICKLE)
        external = []
        payload = pickle.dumps({"g": np.arange(4)}, protocol=5, buffer_callback=external.append)
        self.assertTrue(external)
        self.rejects(payload, PICKLE)
        for payload in (b"\x80\x02Premote\n.", b"\x80\x02NQ.", b"\x80\x02\x82\x01.",
                        b"\x80\x02N0.", b"\x80\x02\x81."):
            self.rejects(payload, PICKLE)

    def test_pickle_resource_limits(self):
        payload = pickle.dumps({"g": np.arange(8).reshape(2, 4)}, protocol=4)
        for changes in ({"max_elements": 4}, {"max_array_bytes": 8},
                        {"max_total_array_bytes": 8}, {"max_dimension": 2},
                        {"max_ndim": 1}, {"max_opcodes": 2}, {"max_memo": 2},
                        {"max_stack": 2}, {"max_tuple_items": 2}):
            with self.subTest(changes=changes):
                self.rejects(payload, PICKLE, limits=replace(DEFAULT_LIMITS, **changes))
        self.rejects(b"\x80\x02" + b"N" * 6 + b".", PICKLE,
                     limits=replace(DEFAULT_LIMITS, max_stack=4))
        self.rejects(b"\x80\x02}r\xff\xff\xff\xff.", PICKLE)
        self.rejects(b"\x80\x02)" + b"\x85" * 8 + b".", PICKLE)
        self.rejects(pickle.dumps({"a": np.ones(2), "b": np.ones(2)}, protocol=4), PICKLE,
                     limits=replace(DEFAULT_LIMITS, max_arrays=1))

    def test_pickle_malformed_and_frame_bounds(self):
        payload = pickle.dumps({"g": np.arange(4)}, protocol=4)
        for bad in (payload[:-1], payload + b"trailing", payload + payload,
                    payload[:3] + struct.pack("<Q", 2**63) + payload[11:],
                    payload[:3] + struct.pack("<Q", 3) + payload[11:],
                    b"\x80\x02h\xff.", b"\x80\x02}q\x00q\x00."):
            self.rejects(bad, PICKLE)

    def test_npy_preserves_numeric_values_order_and_endian(self):
        for array in (np.arange(12, dtype=">f4").reshape(3, 4),
                      np.asfortranarray(np.arange(12, dtype="complex128").reshape(3, 4)),
                      np.array(True), np.empty((2, 0))):
            result = decode_numeric(npy_bytes(array), NPY)
            actual = result.arrays["data"]
            np.testing.assert_array_equal(actual, array)
            self.assertEqual(actual.dtype.str, array.dtype.str)
            self.assertEqual(actual.flags.f_contiguous, array.flags.f_contiguous)

    def test_npy_header_and_payload_rejections(self):
        for array in (np.array([object()]), np.zeros(2, dtype=[("x", "f8")]), np.array(["abc"])):
            self.rejects(npy_bytes(array), NPY)
        for header in (
            "{'descr': '|O8', 'shape': (1,), 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (9999999999999,), 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (-1,), 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (True,), 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (1,), 'fortran_order': 0}",
            "{'descr': '<f8', 'shape': [1], 'fortran_order': False}",
            "{'descr': '<f8', 'descr': '<f4', 'shape': (1,), 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (1,), 'fortran_order': False, 'extra': 1}",
            "__import__('os').system('echo NEVER_EXECUTE')",
            "{'descr': '<f8', 'shape': (1,), 'fortran_order': False}\x1e",
        ):
            self.rejects(raw_npy(header, b"\x00" * 8), NPY)
        good = npy_bytes(np.arange(4))
        for bad in (good[:-1], good + b"x", good[:6] + b"\x09\x00" + good[8:],
                    b"\x93NUMPY\x01\x00\xff\xff{}"):
            self.rejects(bad, NPY)

    def test_npz_two_numeric_members(self):
        for compressed in (False, True):
            stream = io.BytesIO()
            data = np.arange(24, dtype="f4").reshape(2, 4, 3)
            coefficients = np.array([1., -2., 3.])
            (np.savez_compressed if compressed else np.savez)(stream, data=data, coefficients=coefficients)
            result = decode_numeric(stream.getvalue(), NPZ)
            np.testing.assert_array_equal(result.arrays["data"], data)
            np.testing.assert_array_equal(result.arrays["coefficients"], coefficients)

    def test_npz_paths_counts_duplicates_ratio_and_bounds(self):
        data = npy_bytes(np.arange(4))
        good = [("data.npy", data), ("coefficients.npy", data)]
        for members in ([good[0]], good + [("extra.npy", data)],
                        [good[0], good[0]], [("../data.npy", data), good[1]],
                        [("/data.npy", data), good[1]], [("nested/data.npy", data), good[1]],
                        [("data.npy", npy_bytes(np.array([object()]))), good[1]]):
            self.rejects(zip_bytes(members), NPZ)
        payload = zip_bytes(good)
        self.rejects(payload + b"extra", NPZ)
        self.rejects(payload[:-1], NPZ)
        self.rejects(zip_bytes(good, zipfile.ZIP_BZIP2), NPZ)
        self.rejects(zip_bytes(good, zipfile.ZIP_DEFLATED), NPZ,
                     limits=replace(DEFAULT_LIMITS, max_zip_ratio=1))
        self.rejects(payload, NPZ, limits=replace(DEFAULT_LIMITS, max_total_array_bytes=40))
        # Forge EOCD counts to two while the central directory has three entries.
        payload = bytearray(zip_bytes(good + [("extra.npy", data)]))
        pos = payload.rfind(b"PK\x05\x06")
        struct.pack_into("<HH", payload, pos + 8, 2, 2)
        self.rejects(bytes(payload), NPZ)

    def test_npz_actual_inflate_budget_not_declared_size(self):
        # Regression: ZipExtFile used to truncate a 200136-byte inflate to the
        # forged 136-byte claim and verify only the forged prefix CRC.
        npy = npy_bytes(np.array([1.], dtype="f8"))
        payload = bytearray(zip_bytes([
            ("data.npy", npy + b"\0" * 200000), ("coefficients.npy", npy)], zipfile.ZIP_DEFLATED))
        crc = zlib.crc32(npy)
        struct.pack_into("<I", payload, 14, crc)
        struct.pack_into("<I", payload, 22, len(npy))
        central = payload.find(b"PK\x01\x02")
        struct.pack_into("<I", payload, central + 16, crc)
        struct.pack_into("<I", payload, central + 24, len(npy))
        self.assertLess(len(payload), 4096)
        with self.assertRaisesRegex(DecodeError, "actual inflated output exceeds"):
            decode_numeric(bytes(payload), NPZ, limits=replace(DEFAULT_LIMITS, max_payload_bytes=4096))

    def test_npz_compressed_eof_actual_length_trailing_and_crc(self):
        npy = npy_bytes(np.array([1.], dtype="f8"))
        compressed = deflate_bytes(npy)
        valid = encoded_zip(npy, compressed)
        np.testing.assert_array_equal(decode_numeric(valid, NPZ).arrays["data"], [1.])
        for encoded, declared in (
            (compressed + b"hidden tail", len(npy)),
            (compressed + deflate_bytes(b"concatenated stream"), len(npy)),
            (compressed[:-1], len(npy)),
            (compressed, len(npy) + 1),
            (deflate_bytes(npy + b"hidden uncompressed tail"), len(npy)),
            (deflate_bytes(npy[:-1] + b"x"), len(npy)),
        ):
            with self.subTest(declared=declared, compressed_length=len(encoded)):
                self.rejects(encoded_zip(npy, encoded, declared), NPZ)
        payload = bytearray(valid)
        struct.pack_into("<H", payload, 6, 8)  # Local data-descriptor flag conflicts with central.
        self.rejects(bytes(payload), NPZ)
        payload = bytearray(valid)
        struct.pack_into("<I", payload, 22, len(npy) + 1)  # Local-only size lie.
        self.rejects(bytes(payload), NPZ)

    def test_npz_inflate_chunk_boundaries(self):
        # Compressible data exercises repeated bounded-output calls; random data
        # also exercises multiple compressed-input chunks (>64 KiB).
        rng = np.random.default_rng(121113)
        for data in (np.tile(np.arange(1024, dtype="f8"), 128), rng.normal(size=20000)):
            stream = io.BytesIO()
            np.savez_compressed(stream, data=data, coefficients=np.ones(2))
            result = decode_numeric(stream.getvalue(), NPZ)
            np.testing.assert_array_equal(result.arrays["data"], data)

    def test_bounded_deterministic_mutations(self):
        # Some mutations validly change numeric data/labels; acceptance is fine.
        # All other cases must reject through the public DecodeError boundary.
        original = {"g": np.arange(8, dtype="f8").reshape(2, 4)}
        samples = [(pickle.dumps(original, protocol=p), PICKLE) for p in (2, 3, 4, 5)]
        samples.append((npy_bytes(original["g"]), NPY))
        stream = io.BytesIO()
        np.savez_compressed(stream, data=original["g"], coefficients=np.ones(2))
        samples.append((stream.getvalue(), NPZ))
        rng = random.Random(121113)
        for source, profile in samples:
            for _ in range(500):
                changed = bytearray(source)
                for _ in range(rng.randrange(1, 4)):
                    changed[rng.randrange(len(changed))] = rng.randrange(256)
                try:
                    decode_numeric(bytes(changed), profile)
                except DecodeError:
                    pass

    def test_text_header_location_labels_precision_and_nan(self):
        text = b"propagator diagnostic\n[1 2\n 3 4]\n\nGamma\t(g5_g5)_real (g5_g5)_imag\r\nt0 1.234567e-02 -2.0\r\nt1 nan inf\n"
        result = decode_numeric(text, TEXT)
        self.assertEqual(list(result.arrays), ["g5_g5"])
        self.assertEqual(result.arrays["g5_g5"].shape, (2, 2))
        self.assertEqual(result.arrays["g5_g5"][0, 0], 0.01234567)
        self.assertTrue(np.isnan(result.arrays["g5_g5"][0, 1]))
        self.assertIn("rounded %.6e", result.metadata["precision"])
        result = decode_numeric(b"ga3ma g1_real g1_imag\nt0 1 2", TEXT)
        np.testing.assert_array_equal(result.arrays["g1"], [[1], [2]])

    def test_text_rejects_incomplete_ambiguous_and_overbudget(self):
        for text in (
            "ga3ma g_real g_imag\nt1 1 2", "ga3ma g_real g_imag\nt0 1 2\nt2 3 4",
            "ga3ma g_real g_imag\nt0 1 2\nt0 3 4", "ga3ma g_real g_imag\nt0 1",
            "ga3ma g_real h_imag\nt0 1 2", "ga3ma g_imag g_real\nt0 1 2",
            "ga3ma g_real g_imag g_real g_imag\nt0 1 2 3 4",
            "ga3ma g_real g_imag\nt0 1 2\nGamma (g)_real (g)_imag\nt0 1 2",
            "Gamma g_real g_imag\nt0 1 2", "ga3ma g_real g_imag",
            "ga3ma g_real g_imag\nt0 1e9999 2", "ga3ma g_real g_imag\nt0 __import__('os') 2",
        ):
            self.rejects(text.encode(), TEXT)
        payload = b"ga3ma g_real g_imag\nt0 1 2"
        self.rejects(payload, TEXT, limits=replace(DEFAULT_LIMITS, max_array_bytes=8))
        self.rejects(payload, TEXT, limits=replace(DEFAULT_LIMITS, max_text_line_chars=8))
        self.rejects(b"\xff" + payload, TEXT)


class TestFormatFixtures(unittest.TestCase):
    def test_preserved_hashes_formats_and_exact_text_rounding(self):
        root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "correlators"
        # Missing fixtures are an error, never silently skipped.
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        records = {record["local_name"]: record for record in provenance["files"]}
        for stem, expected_count in (("two_point", 70), ("three_point", 16)):
            decoded = {}
            for extension, profile in (("bin", PICKLE), ("txt", TEXT)):
                name = stem + "." + extension
                payload = (root / name).read_bytes()
                record = records[name]
                self.assertEqual(provenance["origin"], "synthetic-format-fixtures")
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
                decoded[extension] = decode_numeric(payload, profile).arrays
            self.assertEqual(len(decoded["bin"]), expected_count)
            self.assertEqual(set(decoded["bin"]), set(decoded["txt"]))
            maximum_error = 0.0
            for key, binary in decoded["bin"].items():
                text = decoded["txt"][key]
                self.assertEqual(binary.shape, (2, 64))
                self.assertEqual(binary.dtype, np.dtype("float64"))
                rounded = np.array([float(f"{value:.6e}") for value in binary.flat]).reshape(binary.shape)
                np.testing.assert_array_equal(text, rounded)
                maximum_error = max(maximum_error, float(np.max(np.abs(text - binary))))
            print(f"FORMAT_FIXTURE {stem}: {expected_count} labels, shape=(2,64), dtype=float64, "
                  f"exact %.6e correspondence, max_abs_rounding_error={maximum_error:.17g}")


if __name__ == "__main__":
    unittest.main()
