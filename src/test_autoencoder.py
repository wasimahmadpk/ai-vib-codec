import unittest

import numpy as np
import torch

from autoencoder import BLOCK, CODE_TIME, HEADER, Autoencoder, Codec
from signals import split_by_time


class AutoencoderTest(unittest.TestCase):
    def test_code_length_matches_the_block(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=8)
        samples = torch.zeros(2, BLOCK)
        speed = torch.zeros(2)
        code = model.encoder(samples, speed)
        self.assertEqual(tuple(code.shape), (2, 8, CODE_TIME))
        restored = model.decoder(code, speed)
        self.assertEqual(tuple(restored.shape), (2, BLOCK))

    def test_packet_is_int8_and_decodes(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=8)
        model.scale.fill_(0.02)
        codec = Codec(model, mean=0.0, std=1.0, rpm_mean=800.0, rpm_std=100.0)
        block = np.linspace(-1.0, 1.0, BLOCK, dtype=np.float32)
        payload = codec.encode(block, 900.0)
        self.assertEqual(len(payload), HEADER.size + 8 * CODE_TIME)
        restored = codec.decode(payload, 900.0)
        self.assertEqual(restored.shape, (BLOCK,))
        self.assertEqual(restored.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(restored)))
        self.assertFalse(np.allclose(restored, block))

    def test_bad_payload_is_rejected(self) -> None:
        codec = Codec(Autoencoder(8), 0.0, 1.0, 800.0, 100.0)
        with self.assertRaises(ValueError):
            codec.decode(b"nope", 900.0)

    def test_split_keeps_recordings_in_time_order(self) -> None:
        years = np.array([3.0, 0.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        train, validation, test = split_by_time(years)
        self.assertEqual(len(train) + len(validation) + len(test), len(years))
        self.assertLessEqual(years[train].max(), years[validation].min())
        self.assertLessEqual(years[validation].max(), years[test].min())


if __name__ == "__main__":
    unittest.main()
