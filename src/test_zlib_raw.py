import unittest

import numpy as np

from zlib_raw import decode, encode


class ZlibRawTest(unittest.TestCase):
    def test_float32_roundtrip_is_exact(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(size=2048).astype(np.float32)

        recovered = decode(encode(samples))

        self.assertEqual(recovered.dtype, np.dtype("<f4"))
        self.assertTrue(np.array_equal(samples, recovered))

    def test_bad_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode(b"not a payload")


if __name__ == "__main__":
    unittest.main()
