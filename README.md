# ai-vib-codec

Lossy compression of vibration recordings. Exact codecs are the baseline.

Wind turbine, `DATASET02` (LTU, DOI 10.5878/bcmv-wq08): 2,982 recordings, 16,384 samples at 12.8 kHz, acceleration in g plus shaft speed. Split by time, oldest first: 2,087 train, 447 validation, 448 test. A recording is never cut in half. The integer codecs round each sample to 0.001 g. That rounding is 0.000289 g RMSE, and they then restore every count.

Lab-motor bearing, CWRU slice: 18 files, 4 healthy and 14 faulty, inner race, ball, and outer race. Fault files are 12 kHz. Healthy files are 48 kHz and are resampled. Load is 0 to 3 HP. The shaft runs at about 1,720 to 1,800 RPM. Each file stays whole: 10 train, 3 validation, 5 test.

Code is in `src/`. `notebooks/turbine_eda.ipynb` looks at the Luleå wind-turbine file. `notebooks/cwru_eda.ipynb` looks at the lab-motor bearing set. `data/` and `results/` stay off git.

A first run fills `data/` when the files are missing. A turbine command downloads [DATASET02.csv](https://api.researchdata.se/dataset/2024-248/1/file/data/DATASET02.csv) (about 850 MB) and stores `data/dataset02_f32.npz`. A lab-motor command downloads the 18 Case Western `.mat` files into `data/cwru/`. A file that is already there is left alone.

## Chosen codec

One autoencoder: 2,048 int8 values (16 channels by 128 steps), zlib-packed, plus 8 bytes per block for the order-35 tone and the single largest sample. The encoder is a small 1D conv, 18 KiB. Shaft speed and the weights are shared, so they stay out of the packet. An old packet only decodes with this model.

On the 448 test recordings it removes **59.0%** of the int16 file (6.56 bits/sample) and returns **99.7%** of the energy. RMSE is 0.033 g. The loudest tone is exact. Sharp peaks are still about 0.7 g off.

CWRU has its own 2048 run. Each block is weighted by its own energy, otherwise the quiet healthy file barely shows up in the loss. That checkpoint is not the turbine codec above.

```bash
cd /Users/wasim/ai-vib-codec
.venv/bin/python src/run_autoencoder.py --code-channels 16 --code-time 128 --side-info --pack-code
```

## Comparison

Same methods on both sets. Removed is versus the int16 file. Energy back is how much of the recording returns. 0% means silence. Fault place is the five held-out CWRU files. The 4/5 miss is the healthy file. Encode and decode are milliseconds for one recording: 1.28 s on the turbine, about 10 s on CWRU. The last CWRU row sends six features a second and does not rebuild the recording, so energy and decode are blank.

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

zlib, delta, AR4, and a small MLP remove less than Rice and still return every count. On CWRU, 256 values and shorter call every fault file healthy. The six features are rms, kurtosis, crest factor, and the inner, outer, and ball lines, sent as plain numbers.

`src/run_benchmark.py` loads the saved autoencoders, scores both test sets again, and writes `results/benchmark.md`. It does not train them. Point `--turbine-codec` or `--cwru-codec` at a new checkpoint and run the same command.

`src/run_inference.py` is a separate command. It compresses one recording with the saved turbine codec and prints the packet size and the error. The default is a replay of the first held-out test recording. A new file is a float `.npy` plus its shaft speed. The recording is already finished, so the whole file is compressed at once.

```bash
cd /Users/wasim/ai-vib-codec
.venv/bin/python src/run_benchmark.py
.venv/bin/python src/run_inference.py
.venv/bin/python src/run_inference.py --signal recording.npy --rpm 1150
.venv/bin/python src/run_baseline.py --data data/dataset02_f32.npz
.venv/bin/python src/run_cwru_benchmark.py
```
