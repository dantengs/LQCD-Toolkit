"""Synthetic legacy NumPy pickle flags; no private correlators are embedded."""
import pickle
import pickletools
import unittest

import numpy as np

from lqcd_analysis.decoders import DecodeError, decode_numeric


PROFILE = "savedata.pickle.numeric.v1"


def integer_flag_payload(array, invalid_index=None):
    """Encode only the three NumPy flag booleans as protocol-2 BININT1."""
    payload = pickle.dumps({"signal": array}, protocol=2)
    operations = list(pickletools.genops(payload))
    chunks, flag_index = [], 0
    for i, (op, arg, start) in enumerate(operations):
        end = operations[i + 1][2] if i + 1 < len(operations) else len(payload)
        if op.name in ("NEWFALSE", "NEWTRUE"):
            value = 2 if flag_index == invalid_index else int(op.name == "NEWTRUE")
            chunks.append(b"K" + bytes([value]))
            flag_index += 1
        else:
            chunks.append(payload[start:end])
    if flag_index != 3:
        raise AssertionError("Synthetic fixture must encode two dtype flags and one order flag")
    return b"".join(chunks)


class LegacyIntegerFlagTests(unittest.TestCase):
    def test_float_precision_endian_and_memory_order(self):
        for dtype in ("<f4", "<f8", ">f8"):
            for order in ("C", "F"):
                original = np.array(np.arange(18).reshape(3, 6) / 7, dtype=dtype, order=order)
                with self.subTest(dtype=dtype, order=order):
                    result = decode_numeric(integer_flag_payload(original), PROFILE).arrays["signal"]
                    np.testing.assert_array_equal(result, original)
                    self.assertEqual(result.dtype.str, original.dtype.str)
                    self.assertEqual(result.flags.f_contiguous, original.flags.f_contiguous)
                    self.assertFalse(result.flags.writeable)

    def test_invalid_integer_rejected_at_each_flag(self):
        original = np.arange(12, dtype="f8").reshape(2, 6)
        for index in range(3):
            with self.subTest(flag=index), self.assertRaises(DecodeError):
                decode_numeric(integer_flag_payload(original, index), PROFILE)

    def test_object_dtype_still_rejected(self):
        with self.assertRaises(DecodeError):
            decode_numeric(pickle.dumps({"signal": np.array([object()], dtype=object)}, protocol=2), PROFILE)


if __name__ == "__main__":
    unittest.main()
