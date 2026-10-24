import unittest

import numpy as np

import ar4
import harmonic_rice
import mlp_predictor
import zlib_int
from fixed_point import G_PER_COUNT, to_g, to_int16


class BaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        t = np.arange(2048)
        signal = (
            900 * np.sin(2 * np.pi * 0.035 * t)
            + 250 * np.sin(2 * np.pi * 0.11 * t)
            + rng.normal(0, 30, len(t))
        )
        cls.samples = np.rint(signal).astype(np.int16)

    def assert_roundtrip(self, encode, decode) -> None:
        recovered = decode(encode(self.samples))
        self.assertTrue(np.array_equal(self.samples, recovered))

    def test_fixed_point_error_is_bounded(self) -> None:
        values = np.array([-1.2344, 0.0, 2.3456])
        recovered = to_g(to_int16(values))
        self.assertLessEqual(float(np.max(np.abs(values - recovered))), G_PER_COUNT / 2)

    def test_zlib_raw_int16_is_exact(self) -> None:
        self.assert_roundtrip(zlib_int.encode_raw, zlib_int.decode_raw)

    def test_zlib_delta_is_exact(self) -> None:
        self.assert_roundtrip(zlib_int.encode_delta, zlib_int.decode_delta)

    def test_ar4_is_exact(self) -> None:
        self.assert_roundtrip(ar4.encode, ar4.decode)

    def test_harmonic_rice_is_exact(self) -> None:
        self.assert_roundtrip(harmonic_rice.encode, harmonic_rice.decode)

    def test_mlp_residual_is_exact(self) -> None:
        recordings = np.vstack([
            self.samples,
            np.roll(self.samples, 3),
            np.roll(self.samples, 7),
        ])
        model = mlp_predictor.train(
            recordings[:2],
            recordings[2:],
            epochs=3,
            batch_size=512,
        )
        payload = mlp_predictor.encode(self.samples, model)
        recovered = mlp_predictor.decode(payload, model)
        self.assertTrue(np.array_equal(self.samples, recovered))


if __name__ == "__main__":
    unittest.main()
