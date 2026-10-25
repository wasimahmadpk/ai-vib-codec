# ai-vib-codec

Lossy compression of vibration recordings, with exact codecs as the comparison.

Wind turbine, `DATASET02` (LTU, DOI 10.5878/bcmv-wq08): 2,982 recordings, 16,384 samples at 12.8 kHz, acceleration in g plus shaft speed. Split by time, oldest first: 2,087 train, 447 validation, 448 test. A recording is never cut in half. The integer codecs round each sample to 0.001 g. That rounding is 0.000289 g RMSE, and they then restore every count.

Lab-motor bearing, CWRU slice: 18 files, 4 healthy and 14 faulty, inner race, ball, and outer race. Fault files are 12 kHz. Healthy files are 48 kHz and are resampled. Load is 0 to 3 HP. The shaft runs at about 1,720 to 1,800 RPM. Each file stays whole: 10 train, 3 validation, 5 test.

Code is in `src/`. `notebooks/turbine_eda.ipynb` looks at the Luleå wind-turbine file. `notebooks/cwru_eda.ipynb` looks at the lab-motor bearing set. `data/` and `results/` stay off git.

## Chosen codec

The pipeline keeps one autoencoder: 2,048 int8 values (16 channels by 128 steps), zlib-packed, plus 8 bytes per block for the order-35 tone and the single largest sample. The encoder is a small 1D conv, 18 KiB, and is the piece that runs at the edge. Shaft speed and the weights are shared, so they are not in the packet. An old packet needs this same model.

On the 448 test recordings it removes **59.0%** of the int16 file (6.56 bits/sample) and returns **99.7%** of the energy. RMSE is 0.033 g. The loudest tone is exact. Sharp peaks are still about 0.7 g off.

The CWRU 2048-value model is a separate training run, so a quiet block counts as much as a loud one. It does not replace this codec.

```bash
.venv/bin/python src/run_autoencoder.py --code-channels 16 --code-time 128 --side-info --pack-code
```

## Comparison

Same methods on both sets. Removed is versus the int16 file. Energy recovered is the share of the recording that comes back. 0% is silence. Fault place is how many of the five held-out CWRU files are still named. The 4/5 miss is the healthy file.

| Dataset | Method | Removed | Energy back | Fault place |
|---|---|---:|---:|---:|
| Wind turbine | Rice, best exact codec | 28.3% | 100% | — |
| Wind turbine | Autoencoder, 512 values | 87.3% | 82% | — |
| Wind turbine | Autoencoder, 256 values | 93.6% | 69% | — |
| Wind turbine | Autoencoder, 64 values | 98.2% | 46% | — |
| Wind turbine | Autoencoder, 16 values | 99.4% | none | — |
| Wind turbine | **Autoencoder, 2048 packed** | **59.0%** | **99.7%** | — |
| CWRU | Rice, best exact codec | 35.7% | 100% | 5/5 |
| CWRU | Autoencoder, 512 values | 87.3% | 68% | 4/5 |
| CWRU | Autoencoder, 256 values | 93.6% | none | 1/5 |
| CWRU | Autoencoder, 64 values | 98.2% | none | 1/5 |
| CWRU | Autoencoder, 16 values | 99.4% | none | 1/5 |
| CWRU | **Autoencoder, 2048 packed** | **52.4%** | **89%** | **5/5** |

The other exact codecs (zlib, delta, AR4, a tiny MLP) remove less than Rice and return every count. Shorter autoencoder codes are smaller files and worse recordings. On CWRU, codes of 256 values and shorter call every fault healthy. Rice is the comparison when the counts have to come back exactly.

```bash
.venv/bin/python src/run_baseline.py --data data/dataset02_f32.npz
.venv/bin/python src/run_cwru_benchmark.py
```
