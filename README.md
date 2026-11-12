# ai-vib-codec

Lossy compression of vibration recordings. Exact codecs are the baseline. The chosen model is a small convolutional autoencoder that stores an int8 code.

## Data

Wind turbine, `DATASET02` (LTU, DOI 10.5878/bcmv-wq08): 2,982 recordings from one sensor, 16,384 samples at 12.8 kHz, acceleration in g plus shaft speed. The release has six files from that same sensor. This project uses `DATASET02` only. Split by time, oldest first: 2,087 train, 447 validation, 448 test (rows 2534–2981). A recording is never cut in half.

The size baseline rounds each sample to the nearest 0.001 g and stores that count as int16. One count is one millig. That rounding is 0.000289 g RMSE, and the exact codecs then restore every count. The autoencoder is trained on the g values, not on those counts.

Lab-motor bearing, CWRU slice: 18 files, 4 healthy and 14 faulty, inner race, ball, and outer race. Pits are 0.007 in and 0.021 in. Load is 0 to 3 HP. The shaft runs at about 1,720 to 1,800 RPM. Fault files are 12 kHz. Healthy files are 48 kHz and are resampled to 12 kHz once, at load, and left there. Each file stays whole: 10 train, 3 validation, 5 test. The CWRU files do not name the acceleration unit.

Code is in `src/`. `notebooks/turbine_eda.ipynb` looks at the turbine file. `notebooks/cwru_eda.ipynb` looks at the lab-motor set. `data/` and `results/` stay off git.

## Setup

```bash
git clone https://github.com/wasimahmadpk/ai-vib-codec.git
cd ai-vib-codec
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

That installs numpy, scipy, matplotlib, jupyter, ipykernel, and torch. The commands below use `.venv/bin/python` from this folder.

A first run downloads recordings when they are missing. A turbine command fetches [DATASET02.csv](https://api.researchdata.se/dataset/2024-248/1/file/data/DATASET02.csv) (about 850 MB) and stores `data/dataset02_f32.npz`. A lab-motor command fetches the 18 Case Western `.mat` files into `data/cwru/`. A file that is already there is left alone.

The trained weights are not in this repository. Inference and the benchmark stop if the `.pt` files below are missing. Training writes a new checkpoint into `data/`.

## Run

Compress one finished turbine recording. The default codec is `data/ae_2048_zlib.pt`. With no `--index`, a terminal asks for a test row from 2534 to 2981. Enter replays the first of those rows.

```bash
.venv/bin/python src/run_inference.py
.venv/bin/python src/run_inference.py --index 2600
.venv/bin/python src/run_inference.py --signal recording.npy --rpm 1150
```

Score the saved models again and rewrite `results/benchmark.md`. This does not train the autoencoders. It does fit the small MLP baseline and the lab-motor fault classifier again. Both are tiny and are not saved. `--turbine-codec` and `--cwru-codec` only choose the packed 2048 checkpoint for that table row. The other sizes are still loaded from the names below.

```bash
.venv/bin/python src/run_benchmark.py
.venv/bin/python src/run_benchmark.py --turbine-codec data/ae_2048_zlib.pt --cwru-codec data/cwru_ae_2048_zlib.pt
```

Turbine checkpoints the benchmark expects: `data/ae_16.pt`, `ae_64.pt`, `ae_256.pt`, `ae_512.pt`, `ae_1024.pt`, `ae_2048.pt`, and `ae_2048_zlib.pt`.

Lab-motor checkpoints: `data/cwru_ae_16.pt`, `cwru_ae_64.pt`, `cwru_ae_256.pt`, `cwru_ae_512.pt`, `cwru_ae_1024.pt`, `cwru_ae_2048.pt`, and `cwru_ae_2048_zlib.pt`.

```bash
.venv/bin/python src/run_baseline.py --data data/dataset02_f32.npz
.venv/bin/python src/run_cwru_benchmark.py
```

`run_baseline.py` scores the exact codecs on the turbine test rows. `run_cwru_benchmark.py` trains the lab-motor autoencoders. Running it with no flag overwrites `data/cwru_ae_16.pt` through `data/cwru_ae_2048.pt`.

## Chosen codec

One autoencoder: 2,048 int8 values (16 channels by 128 steps), zlib-packed, plus 8 bytes per block. Those 8 bytes are the order-35 gear tone, as two float16 coefficients, and the single largest sample, as its position and a float16 value. The encoder is a small 1D conv, 18 KiB. Shaft speed and the weights stay out of the packet.

On the 448 test recordings it removes **59.0%** of the int16 file (6.56 bits/sample) and returns **99.7%** of the energy. RMSE is 0.033 g. The loudest tone is exact. Sharp peaks are still about 0.7 g off.

Train that turbine codec with:

```bash
.venv/bin/python src/run_autoencoder.py --code-channels 16 --code-time 128 --side-info --pack-code
```

That writes `data/ae_2048_pack.pt`, not `data/ae_2048_zlib.pt`. The published row used `data/ae_2048_zlib.pt`, which is the same packed setup. Point later commands at the file you just wrote:

```bash
.venv/bin/python src/run_inference.py --codec data/ae_2048_pack.pt
.venv/bin/python src/run_benchmark.py --turbine-codec data/ae_2048_pack.pt
```

CWRU has its own 2048 run. Each block is weighted by its own energy, otherwise the quiet healthy file barely shows up in the loss. That checkpoint does not replace the turbine codec.

## Comparison

Same methods on both sets. Removed is versus the int16 file. Energy back is how much of the recording returns. 0% would be silence. Fault place is how many of the five held-out CWRU files are still named. The 4/5 miss is the healthy file. Encode and decode are milliseconds for one recording: 1.28 s on the turbine, about 10 s on CWRU. The last CWRU row sends six features a second and does not rebuild the recording, so energy and decode are blank.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th align="right">Removed</th>
      <th align="right">Energy back</th>
      <th align="right">Encode (ms)</th>
      <th align="right">Decode (ms)</th>
      <th align="right">Fault place</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Wind turbine</td>
      <td>Rice, best exact codec</td>
      <td align="right">28.3%</td>
      <td align="right">100%</td>
      <td align="right">31.2</td>
      <td align="right">24.0</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td>AE, 16 values</td>
      <td align="right">99.4%</td>
      <td align="right">none</td>
      <td align="right">1.09</td>
      <td align="right">0.67</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td>AE, 64 values</td>
      <td align="right">98.2%</td>
      <td align="right">46%</td>
      <td align="right">0.99</td>
      <td align="right">0.64</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td>AE, 256 values</td>
      <td align="right">93.6%</td>
      <td align="right">69%</td>
      <td align="right">1.02</td>
      <td align="right">0.64</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td>AE, 512 values</td>
      <td align="right">87.3%</td>
      <td align="right">82%</td>
      <td align="right">0.83</td>
      <td align="right">0.55</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td>AE, 1024 values</td>
      <td align="right">74.8%</td>
      <td align="right">93%</td>
      <td align="right">0.77</td>
      <td align="right">0.55</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td><strong>AE, 2048 packed</strong></td>
      <td align="right"><strong>59.0%</strong></td>
      <td align="right"><strong>99.7%</strong></td>
      <td align="right"><strong>1.18</strong></td>
      <td align="right"><strong>0.71</strong></td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td rowspan="8">CWRU</td>
      <td>Rice, best exact codec</td>
      <td align="right">35.7%</td>
      <td align="right">100%</td>
      <td align="right">214</td>
      <td align="right">157</td>
      <td align="right">5/5</td>
    </tr>
    <tr>
      <td>AE, 16 values</td>
      <td align="right">99.4%</td>
      <td align="right">none</td>
      <td align="right">7.5</td>
      <td align="right">5.0</td>
      <td align="right">1/5</td>
    </tr>
    <tr>
      <td>AE, 64 values</td>
      <td align="right">98.2%</td>
      <td align="right">none</td>
      <td align="right">7.5</td>
      <td align="right">5.0</td>
      <td align="right">1/5</td>
    </tr>
    <tr>
      <td>AE, 256 values</td>
      <td align="right">93.6%</td>
      <td align="right">none</td>
      <td align="right">7.6</td>
      <td align="right">5.0</td>
      <td align="right">1/5</td>
    </tr>
    <tr>
      <td>AE, 512 values</td>
      <td align="right">87.3%</td>
      <td align="right">68%</td>
      <td align="right">6.1</td>
      <td align="right">4.3</td>
      <td align="right">4/5</td>
    </tr>
    <tr>
      <td>AE, 1024 values</td>
      <td align="right">74.8%</td>
      <td align="right">77%</td>
      <td align="right">6.4</td>
      <td align="right">4.4</td>
      <td align="right">4/5</td>
    </tr>
    <tr>
      <td><strong>AE, 2048 packed</strong></td>
      <td align="right"><strong>52.4%</strong></td>
      <td align="right"><strong>89%</strong></td>
      <td align="right"><strong>7.5</strong></td>
      <td align="right"><strong>4.2</strong></td>
      <td align="right"><strong>5/5</strong></td>
    </tr>
    <tr>
      <td>Six features, exact</td>
      <td align="right">99.8%</td>
      <td align="right">—</td>
      <td align="right">3.6</td>
      <td align="right">—</td>
      <td align="right">5/5</td>
    </tr>
  </tbody>
</table>

zlib, delta, AR(4), and a small MLP remove less than Rice and still return every count. Rice finds the eight strongest tones in a block, subtracts them, and codes the leftover integers. AR(4) and the MLP instead predict each sample from the few samples before it. On CWRU, 256 values and shorter call every fault file healthy. The six features are RMS, kurtosis, crest factor, and the inner, outer, and ball envelope lines, sent as plain numbers.
