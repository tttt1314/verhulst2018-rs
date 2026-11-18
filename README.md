# verhulst2018-rs

Verhulst et al. (2018) 傳輸線耳蝸模型 cochlear core 的 Rust 重寫版，提供 Python 綁定（PyO3 + maturin + rust-numpy）。

> © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License.
>
> 本 repository 是 Verhulstetal2018Model v1.2（upstream commit `ed8d576`）中 `cochlear_model2018.py` 與 `cochlea_utils.c` 的衍生作品，**只能用於非商業學術研究**。授權全文見 [LICENSE](LICENSE)，修改內容見 [NOTICE](NOTICE)。

## 重點

- **預設的忠實模式在 Apple arm64 上與原版逐位元相同。** 不只是「誤差很小」：在 82 個已存的原版模擬上（click：rtol 1e-2 到 1e-8；4 相位掃頻：rtol 1e-2、1e-4），輸出的 `ear_canal_pressure_pa` 完全相等（max |diff| = 0），RHS 呼叫次數也一樣。
- 只存 `oto_emission` 和指定的探針通道。延遲線改用 per-channel 環狀緩衝。整個 process 峰值約 130 MB；原版存全部通道時，掃頻約 2.7–3.3 GB。
- `rtol`、`atol` 可以設定，預設 1e-2、1e-13（原版寫死的值）。
- 提供可選的「純 RHS」變體 `pole_update="accepted_step"`，明確標示為不同的模型，**不是**預設。
- 可直接取代 worker：`python -m verhulst2018_rs.worker` 的命令列和輸出 `.npz` 都和 `carfac_ephys/tl_reference_worker.py` 相同。
- **速度：單次模擬快 4.8 倍（click 0.70 s 對 3.33 s；掃頻 39.6 s 對 189.9 s），沒有達到 10 倍的目標。** 原因見下方〈速度〉。

## 安裝

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install maturin numpy scipy pytest
maturin develop --release
```

Google Drive 內建置時，建議把 Rust 的 build 目錄放到雲端同步範圍外：`export CARGO_TARGET_DIR=$HOME/.cache/verhulst2018-rs-target`。

## 使用

```python
import numpy as np, verhulst2018_rs as vr
poles = np.loadtxt(".../StartingPoles.dat")
r = vr.simulate_pressure(raw_pa_100k, poles)            # worker 慣例：補 2 個 0，輸出長度 = len(raw)
r.oto_emission                                           # 等於原版 model.oto_emission[:len(raw)]
r = vr.simulate(stim, 100_000, 1000, poles, probe_freq=[1000.0], rtol=1e-4,
                Zweig_irregularities=0)                  # init_model(...) + solve() 的對應
```

`simulate` 的參數名稱和原版 `init_model` 相同（`compression_slope`、`Zweig_irregularities`、`non_linearity_type`、`KneeVar`、`low_freq_irregularities`、`subject`、`IrrPct`），另外有 `rtol`、`atol`、`nsteps`、`pole_update`、`probe_channels`、`compact_delay`。Puria 中耳濾波和 600–4000 Hz 的 emission 濾波仍用 SciPy，呼叫方式和原版相同。`Rth` 由 NumPy 的 `RandomState(subject)` 產生（和原版的 `np.random.seed` 是同一串亂數），但不會改動全域亂數產生器的狀態。

Worker：

```bash
python -m verhulst2018_rs.worker --source-dir <Verhulstetal2018Model> --input stim.npy \
  --output out.npz --metadata out.json --poles poles.dat [--rtol 1e-4] [--smooth] \
  [--pole-update accepted_step] [--probe-channels 500]
```

`--source-dir` 只用來計算來源檔的雜湊值（記錄 provenance）和找預設的 poles 檔，不會載入原版程式碼。metadata 會記錄實際使用的 `rtol`、套件版本、git commit 和 `_core` 函式庫的雜湊值。`tone-probes` 模式需要 `carfac_ephys`（用來讀取 protocol）。

## 忠實度：做法

原版的數值結果取決於三段程式碼各自的編譯方式，這個 port 逐一對應：

| 原版元件 | 這裡的處理 |
|---|---|
| NumPy 運算（`TLsolver`、`SheraParameters`、`ZweigImpedance`、init） | 照原式的運算順序寫，不用 FMA（NumPy ufunc 不會合併乘加）；`exp`、`pow`、`atan`、`cos` 等函數呼叫同一個 libm。 |
| `tridiag.so`（`gcc -O3 -ffast-math`，Apple clang arm64） | 從 `otool` 反組譯確認實際的指令序列：`cubic_interpolate` 的參數是 **float32**；`a0` 被重排成 `(y1-(y0+y2))+y3`；三次式被合併成 3 個 FMA；Thomas 演算法中的 `b - a*c` 用 FMA。 |
| SciPy 1.18.1 `dopri5`（`scipy/integrate/src/dop.c` 的 **C 翻譯版**） | 照它的控制流程實作：safety 0.9、fac1 0.2、fac2 10、beta 0.04、nmax 500、HINIT、剛性偵測。clang 的 FP contraction 規則（先檢查左運算元）也從反組譯確認。注意：C 包裝層寫回 `work[6]=h` 時用的是**輸入的** h（0），所以**每個輸出樣本都會重新執行 HINIT**，這就是每個樣本約 9 次 RHS 的來源。 |
| RHS 的副作用 | 每次評估都套用 1% 極點規則（包含 HINIT 的那次和試算 stage）；`Dev += frac`／`-= frac` 的浮點來回誤差會留在 `Dev`；每個輸出樣本後更新延遲線和 `ZweigImpedance`；`self.V1[0]=self.Vtmp[0]` 會透過 view 寫回 `r.y`，這裡也照做。 |

另外有幾個不影響結果的優化，都經過逐位元驗證：

- 快取三對角矩陣的分解（矩陣是常數）；
- 把 `delay_line`、`calculate_g`、`calculate_right` 和 forward sweep 合併成一個迴圈；
- 延遲線改存 float32（原版也是先轉成 float32 才計算）；
- per-channel 環狀緩衝，並在執行時檢查需要的延遲長度，超過就直接報錯；
- 每個樣本只收集一次 5 點視窗並預先計算三次式係數；
- 1% 規則先用乘法做篩選，結果接近門檻時才回到原版的除法判斷。

篩選的誤差約 1e-13，門檻保留了 1e-7 的相對邊界。

## 忠實度：驗證結果

### 單元測試（`tests/test_against_original.py`，26 個測試全部通過）

和原版直接比較（`cubic_interpolate`、`delay_line`、`solve_tridiagonal` 透過 ctypes 呼叫原版的 `tridiag.so`），全部要求**逐位元相等**：

- 三次內插（5000 組隨機值）、延遲線（兩個分支都涵蓋）、三對角求解；
- `init_model` 之後的全部 34 個向量和 22 個純量，涵蓋 6 種設定（smooth、`low_freq_irregularities=0`、disp、linear、其他 subject／IrrPct／KneeVar／slope）；
- 隨機極點下的 `SheraParameters`、`ZweigImpedance`（包含 `Zrp*` 的 wrap-around）；
- 單次 `TLsolver` 評估：輸出的導數，以及之後的 `SheraP`、`Dev`、`YZweig`、`g`、`Qsol`、`Zrp1`，分別測試不更新、會更新、飽和三種強度；
- 端到端短模擬：線性（另外確認相對誤差 ≤ 1e-6）、三種強度的非線性 click、smooth、disp、`low_freq_irregularities=0`、rtol 1e-4 與 1e-6、逐漸加強的純音（會觸發極點更新），並比較 RHS 呼叫次數和探針通道的 V／Y。

Rust 端另有 `cargo test`：Thomas 演算法、cubic、DOPRI5（諧振子）、remainder 的等價性。

### 和已存原版模擬的比較（`validation/run_validation.py`，結果在 `validation/results/parity.json`）

| 資料 | 執行數 | 忠實模式和原版的差異 |
|---|---|---|
| click，2 條件 × L40–L60 × irregular/smooth，rtol 1e-2、1e-4（L45 加 1e-6；post-hoc 1e-6、1e-8） | 54 | **54/54 逐位元相同** |
| 4 相位 L60 掃頻（208,732 樣本），2 條件 × irregular/smooth × 4 相位，rtol 1e-2 與 1e-4 | 32 | **32/32 逐位元相同** |

（rtol 1e-6 的掃頻沒有重跑，因為太耗時。）

### 非線性驗收標準（`validation/analyze_rs.py`，用 r_stability 自己的計分流程，結果在 `validation/results/acceptance.json`）

click 的 r−s 用已存的 48 kHz 轉換、0.8–5 kHz 濾波、onset 後 2–60 ms 計分。掃頻用 r−s 的 4 相位 rwCR crop。表中數字是相對 L2 差異，比較對象是 rtol 1e-2 的原版。標準是「差異 ≤ 原版在 rtol 1e-2／1e-4／1e-6 之間的最大變動」，門檻沒有為了結果調整。

| 案例 | 原版的 rtol 變動 | 忠實模式 | accepted_step 變體 |
|---|---|---|---|
| NH click L40 | 0.010 | **0** | 0.000 ✓ |
| NH click L45 | 0.120 | **0** | 0.244 ✗ |
| NH click L50 | 0.050 | **0** | 0.058 ✗ |
| NH click L55 | 0.018 | **0** | 0.033 ✗ |
| NH click L60 | 0.012 | **0** | 0.035 ✗ |
| Slope20 click L40 | 0.003 | **0** | 0.000 ✓ |
| Slope20 click L45 | 0.004 | **0** | 0.000 ✓ |
| Slope20 click L50 | 0.396 | **0** | 0.406 ✗ |
| Slope20 click L55 | 0.246 | **0** | 0.028 ✓ |
| Slope20 click L60 | 0.049 | **0** | 0.117 ✗ |
| NH 掃頻 r−s rwCR | 0.006 | **0** | 0.117 ✗（corr 0.993） |
| Slope20 掃頻 r−s rwCR | 0.012 | **0** | 0.084 ✗（corr 0.996） |

結論：

- **忠實模式通過所有案例**，而且差異是 0。
- **accepted_step 變體確實是不同的模型。** 它的掃頻 rwCR 和原版差 8–12%，大約是原版數值雜訊底的 10 倍。所以它**不能**當作原版的替代品，只適合用來研究「極點更新時機」本身的影響。L40 的差異是 0，是因為那個強度下沒有發生極點更新。

## 速度

以下是 Apple M1 Pro（10 核心，32 GB）、Python 3.12、NumPy 2.5.3、SciPy 1.18.1 的結果，原版和 Rust 版在同一台機器上依序執行（`benchmarks/bench.py`，原始數據在 `benchmarks/results.json`）。量測期間背景有其他程式在跑，所以絕對時間會有約 ±10% 的波動。

| 刺激 | 實作 | 時間（中位數） | 相對原版 | 峰值記憶體（整個 process） | 和已存原版輸出逐位元相同 |
|---|---|---|---|---|---|
| click 8,200 樣本 | 原版，`probe_freq="all"`（worker 的設定） | 3.33 s | 1× | 381 MB | ✓ |
| | 原版，單一探針 | 3.36 s | 1× | 318 MB | ✓ |
| | **Rust 忠實模式** | **0.70 s** | **4.8×** | 127 MB | ✓ |
| | Rust accepted_step（不同模型） | 0.60 s | 5.6× | 127 MB | ✗ |
| L60 掃頻 208,732 樣本 | 原版，`probe_freq="all"` | 189.9 s | 1× | **2,677 MB** | ✓ |
| | 原版，單一探針 | 187.1 s | 1× | 325 MB | ✓ |
| | **Rust 忠實模式** | **39.6 s** | **4.8×** | 136 MB | ✓ |
| | Rust accepted_step（不同模型） | 26.3 s | 7.2× | 136 MB | ✗ |

click 重複 3 次，掃頻各 1 次。時間不含 import。Rust 的峰值記憶體大部分來自 Python、NumPy 和 SciPy 本身，模型狀態約 10 MB。

**沒有達到 10 倍的原因：**

在逐位元相同的前提下，每次 RHS 都必須依序執行 Thomas 演算法約 1000 步互相依賴的運算：forward 每步是一個 FMA 加一個乘法，backward 每步是一個 FMA。這個依賴鏈在 M1 上約 12k cycles，也就是約 4 µs。再加上 DOPRI5 的誤差範數必須用固定順序逐項相加，HINIT 每個樣本也有約 6000 次依序的加法。算下來，click 的下限約 0.39 s。和這次量到的 3.33 s 相比，最多只能快約 8.5 倍，而且前提是其他工作完全不花時間。目前約 11 µs／RHS（含積分器），其中約一半是這些無法重排的依序計算。

我也試過不要求逐位元相同的「fast」算術版本（SPIKE 分塊三對角求解、多項式 exp、多累加器範數），只換掉 Thomas 演算法時並沒有更快，原因是：

1. 合併後的迴圈原本就把其他工作藏在 Thomas 依賴鏈的延遲裡；
2. 在 rtol = 1e-2 下，任何捨入差異都會改變步長決策，線性情況的差異會變成 3e-5，不符合 ≤ 1e-6 的要求。

所以我把這個版本移除了。

實際使用時，記憶體可能比單次速度更重要：每次模擬約 130 MB（原版約 2.7–3.3 GB），10 核心可以全部平行執行而不會用到 swap。先前 48 次掃頻模擬在原版上因為 swap 花了 39 分鐘，以同樣 6–8 個平行工作估算，現在約 5–8 分鐘。

## 已知差異和限制

- 逐位元相同只在 Apple arm64 上驗證過，而且只針對這台機器上的 `tridiag.so` 和 SciPy 1.18.1 wheel。其他平台上 `fma` 會改成一般的乘法加加法，但原版在那些平台上的編譯結果（fast-math 的重排方式）也可能不同，所以只能預期是「同一個演算法」，數值會有極小的差異。
- 原版的 `init_model` 會呼叫 `np.random.seed(subject)`，改動全域亂數產生器的狀態，這裡不會。
- 不提供原版那種存全部通道的 `Vsolution`／`Ysolution`。需要的通道請用 `probe_channels` 或 `probe_freq`（支援 `"all"`，但會占用相應的記憶體）。
- `compact_delay=True`（預設）時，如果極點需要的延遲超過該通道的緩衝長度，模擬會直接報錯（實際上不會發生：SheraMu ≤ 1.9 < SheraMuMax = 3）。改用 `compact_delay=False` 就會使用原版的緩衝布局。
- 原版 `probe_freq` 用頻率清單的路徑使用 `np.float`，在 NumPy 2 已經不存在；這裡改用 `argmin`，語意相同。

## 授權

依 UGent Academic License 第 4 節：本衍生作品只能為非商業學術研究目的散布，散布時必須附上同一授權（[LICENSE](LICENSE)），每個修改過的檔案都必須標註「已修改」（每個衍生檔案的開頭都有這段聲明），並保留上方引用的 copyright 聲明。注意第 2 節：**原始 Work 本身**未經 UGent 書面同意不得再散布。本 repository 不含原版程式碼；測試會在執行時從使用者本機的原版目錄（`VERHULST_SOURCE_DIR`）載入。
