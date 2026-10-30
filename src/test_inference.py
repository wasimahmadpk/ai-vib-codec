import unittest

import numpy as np

from autoencoder import BLOCK
from run_inference import measure, prepare


class InferenceTest(unittest.TestCase):
    def test_prepare_keeps_a_whole_recording(self) -> None:
        samples, rpm = prepare(np.zeros(BLOCK * 2), 1150.0)
        self.assertEqual(samples.shape, (BLOCK * 2,))
        self.assertEqual(rpm, 1150.0)

    def test_prepare_rejects_a_partial_block(self) -> None:
        with self.assertRaises(ValueError):
            prepare(np.zeros(BLOCK - 1), 1150.0)

    def test_measure_counts_a_smaller_packet(self) -> None:
        original = np.ones(BLOCK, dtype=np.float32)
        restored = original.copy()
        stats = measure(original, restored, b"x" * 100, 1150.0)
        self.assertEqual(stats["raw_bytes"], BLOCK * 2)
        self.assertEqual(stats["packet_bytes"], 100)
        self.assertGreater(stats["removed"], 0)
        self.assertAlmostEqual(stats["energy"], 100.0)
        self.assertAlmostEqual(stats["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
