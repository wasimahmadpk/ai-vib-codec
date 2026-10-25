# ai-vib-codec

Lossy compression of one-second wind-turbine vibration, with exact codecs as the comparison.

`DATASET02` (LTU, DOI 10.5878/bcmv-wq08): 2,982 recordings, 16,384 samples at 12.8 kHz, acceleration in g plus shaft speed. Split by time, oldest first: 2,087 train, 447 validation, 448 test. A recording is never cut in half. The integer codecs round each sample to 0.001 g. That rounding is 0.000289 g RMSE, and they then restore every count.

Code is in `src/`. The notebook `notebooks/data_analysis.ipynb` is a small look at the file. `data/` and `results/` stay off git.

## Chosen codec

The pipeline keeps one autoencoder: 2,048 int8 values (16 channels by 128 steps), zlib-packed, plus 8 bytes per block for the order-35 tone and the single largest sample. The encoder is a small 1D conv, 18 KiB, and is the piece that runs at the edge. Shaft speed and the weights are shared, so they are not in the packet. An old packet needs this same model.

On the 448 test recordings it removes **59.0%** of the int16 file (6.56 bits/sample) and returns **99.7%** of the energy. RMSE is 0.033 g. The loudest tone is exact. Sharp peaks are still about 0.7 g off.

```bash
.venv/bin/python src/run_autoencoder.py --code-channels 16 --code-time 128 --side-info --pack-code
```

## Why not the others

Same 448 recordings. Removed is versus the int16 file. Energy recovered is the share of the recording that comes back. 0% is silence.

| Method | Removed | Energy back |
|---|---:|---:|
| Rice, best exact codec | 28.3% | 100% |
| Autoencoder, 512 values | 87.3% | 82% |
| Autoencoder, 256 values | 93.6% | 69% |
| Autoencoder, 64 values | 98.2% | 46% |
| Autoencoder, 16 values | 99.4% | none |
| **Autoencoder, 2048 packed** | **59.0%** | **99.7%** |

The other exact codecs (zlib, delta, AR4, a tiny MLP) remove 15–19% and return every count. Shorter autoencoder codes are smaller files and worse recordings. Rice is the comparison when the counts have to come back exactly.

```bash
.venv/bin/python src/run_baseline.py --data data/dataset02_f32.npz
```
