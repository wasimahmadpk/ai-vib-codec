import unittest

import numpy as np
import torch

from autoencoder import (
    BLOCK,
    CODE_TIME,
    GEAR_ORDER,
    HEADER,
    SIDE,
    Autoencoder,
    Codec,
    restore_block,
    strip_block,
)
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

    def test_finer_code_keeps_the_block_length(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=16, code_time=32)
        samples = torch.zeros(2, BLOCK)
        speed = torch.zeros(2)
        code = model.encoder(samples, speed)
        self.assertEqual(tuple(code.shape), (2, 16, 32))
        restored = model.decoder(code, speed)
        self.assertEqual(tuple(restored.shape), (2, BLOCK))
        codec = Codec(model, 0.0, 1.0, 800.0, 100.0)
        payload = codec.encode(np.zeros(BLOCK, dtype=np.float32), 900.0)
        self.assertEqual(len(payload), HEADER.size + 16 * 32)
        self.assertEqual(codec.decode(payload, 900.0).shape, (BLOCK,))

    def test_one_millisecond_code_keeps_the_block_length(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=16, code_time=128)
        samples = torch.zeros(2, BLOCK)
        speed = torch.zeros(2)
        code = model.encoder(samples, speed)
        self.assertEqual(tuple(code.shape), (2, 16, 128))
        restored = model.decoder(code, speed)
        self.assertEqual(tuple(restored.shape), (2, BLOCK))
        codec = Codec(model, 0.0, 1.0, 800.0, 100.0)
        payload = codec.encode(np.zeros(BLOCK, dtype=np.float32), 900.0)
        self.assertEqual(len(payload), HEADER.size + 16 * 128)

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

    def test_side_info_restores_the_tone_and_the_peak(self) -> None:
        rpm = 900.0
        freq = GEAR_ORDER * rpm / 60.0
        time = np.arange(BLOCK) / 12_800
        block = (0.4 * np.cos(2 * np.pi * freq * time + 0.3)).astype(np.float32)
        block[100] += np.float32(1.5)
        residual, (index, kept, coeff_cos, coeff_sin) = strip_block(block, rpm)
        self.assertEqual(index, 100)
        self.assertEqual(len(kept), 1)
        restored = restore_block(residual, rpm, index, kept, coeff_cos, coeff_sin)
        self.assertAlmostEqual(float(restored[100]), float(np.float16(block[100])), places=3)
        others = np.delete(restored - block, 100)
        self.assertLess(float(np.max(np.abs(others))), 1e-4)
        self.assertLess(float(np.max(np.abs(np.delete(residual, 100)))), 0.002)

    def test_side_packet_keeps_the_peak_sample(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=16, code_time=128)
        codec = Codec(model, 0.0, 1.0, 800.0, 100.0, side_info=True)
        rpm = 900.0
        freq = GEAR_ORDER * rpm / 60.0
        time = np.arange(BLOCK) / 12_800
        block = (0.2 * np.sin(2 * np.pi * freq * time)).astype(np.float32)
        block[400] = np.float32(1.7)
        payload = codec.encode(block, rpm)
        self.assertEqual(len(payload), HEADER.size + 16 * 128 + SIDE.size)
        self.assertEqual(payload[:4], b"AE02")
        restored = codec.decode(payload, rpm)
        self.assertAlmostEqual(float(restored[400]), float(np.float16(block[400])), places=3)

    def test_packed_window_keeps_the_shock_and_shrinks_the_code(self) -> None:
        torch.manual_seed(0)
        model = Autoencoder(code_channels=16, code_time=128)
        plain = Codec(model, 0.0, 1.0, 800.0, 100.0, side_info=True, window=16)
        packed = Codec(model, 0.0, 1.0, 800.0, 100.0, side_info=True, window=16, compress_code=True)
        rpm = 900.0
        freq = GEAR_ORDER * rpm / 60.0
        time = np.arange(BLOCK) / 12_800
        block = (0.2 * np.sin(2 * np.pi * freq * time)).astype(np.float32)
        block[400:408] += np.float32(1.4)
        _residual, (start, kept, _cos, _sin) = strip_block(block, rpm, window=16)
        payload = packed.encode(block, rpm)
        self.assertEqual(payload[:4], b"AE03")
        restored = packed.decode(payload, rpm)
        self.assertLess(float(np.max(np.abs(restored[start:start + 16] - kept))), 1e-5)
        plain_payload = plain.encode(block, rpm)
        self.assertEqual(plain_payload[:4], b"AE02")

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
