# ai-vib-codec

AI compression of vibration signals.

Start with `data_analysis.ipynb`. The Python code lives in `src/`. Loading and the segment measurements are in `src/signals.py`. The notebook calls those and plots a small subset of the LTU wind-turbine set.

## Classical baselines

The dataset contains acceleration in decimal g rather than original ADC integers. The integer baselines use one fixed representation: one int16 count is `0.001 g`. This adds at most `0.0005 g` rounding error and requires no scale in each packet. Every codec then recovers those int16 values exactly.

Run the lossless baselines with:

```bash
.venv/bin/python src/run_baseline.py --data data/dataset02_f32.npz
```

`DATASET02.csv` has 2,982 recordings. The script keeps them in time order: 2,087 train, 447 validation, 448 test. The shared MLP is trained on the first part. Every method is measured on the same later recordings. Packets use 2,048-sample blocks, and their headers count toward the size.

On that test split, percent of int16 bytes removed:

- `zlib(raw)`: `15.2 ± 5.4%` (13.56 bits/sample)
- `zlib(delta)`: `18.0 ± 5.2%` (13.12 bits/sample)
- `zlib(AR4)`: `18.5 ± 4.8%` (13.05 bits/sample)
- `zlib(MLP)`: `18.3 ± 4.4%` (13.08 bits/sample)
- `Rice(harmonic)`: `28.3 ± 6.3%` (11.47 bits/sample)

These codecs recover the int16 counts exactly. Rounding to millig costs `0.000289 g` RMSE on the whole file. `zlib(raw float32)` is a separate reference: it removes `22.7 ± 8.4%` of the float32 bytes (24.75 bits/sample) and keeps those float32 values exact.

## Lossy autoencoder

`src/autoencoder.py` is the proposed compressor. Each block of 2,048 samples, plus the shaft speed, goes through a small 1D convolutional encoder. The packet is an int8 code. A decoder on the other side rebuilds the block. The shaft speed and the weights are shared, so they are not stored again in each packet. Training uses waveform error plus spectrum error. The spectrum term is not part of the packet.

```bash
.venv/bin/python src/run_autoencoder.py --code-channels 32
```

The same 2,087 / 447 / 448 split is used. Three code sizes, measured on the 448 held-out recordings:

- 16 int8 values: `99.41%` smaller than int16, `0.94 g` RMSE, `-0.3 dB` SNR
- 64 int8 values: `98.24%` smaller, `0.58 g` RMSE, `3.1 dB` SNR
- 256 int8 values: `93.55%` smaller, `0.42 g` RMSE, `5.7 dB` SNR

Recovery below is the share of the recording's energy that comes back. `100%` means the waveform returns. `0%` means it does not. The 16-value code returns `-8.6%`, which is worse than silence, so that counts as no recovery.

- Rice: `28.3%` smaller, `100.0%` recovered (`0.0001%` lost to millig rounding)
- 256 int8 values: `93.6%` smaller, `69.5%` recovered
- 64 int8 values: `98.2%` smaller, `45.9%` recovered
- 16 int8 values: `99.4%` smaller, `0%` recovered

Encode and decode of one recording take about `1 ms`.
