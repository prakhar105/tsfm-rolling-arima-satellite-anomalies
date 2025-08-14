# Satellite Anomaly Detection – TSFM & Rolling Auto ARIMA

This repository contains two complementary approaches for detecting anomalies in **satellite orbital parameters** using **Fengyun‑2F** telemetry data:

1. **IBM Ignite Time Series Foundation Model (TSFM)** with TinyTimeMixer – deep learning–based forecasting and anomaly scoring.
2. **Rolling Auto ARIMA** – statistical forecasting with residual–based anomaly detection.

---

## 📌 Project Overview

Both approaches aim to model and predict **eccentricity** (and related orbital parameters) from Fengyun‑2F satellite data, flagging anomalies when predictions deviate significantly from actual observations.

- **TSFM** excels at capturing complex temporal dependencies in multivariate series.
- **Rolling Auto ARIMA** provides a transparent, statistical baseline with interpretable residuals.

---Rolling_Auto_Arima

## 📂 Structure

```
├── IBM.ipynb                     # TSFM workflow
├── Rolling_Auto_Arima.ipynb    # Rolling Auto ARIMA workflow
├── data/
│   └── Fengyun_2F.csv             # Satellite dataset
├── results/                       # Output plots and anomaly flags
└── README.md                      # This file
```

---

## 1️⃣ IBM Ignite TSFM – Anomaly Detection

# IBM Ignite TSFM – Anomaly Detection on Satellite Data

This project demonstrates **anomaly detection** in satellite orbital parameters using the **IBM Ignite Time Series Foundation Model (TSFM)**, specifically the **TinyTimeMixer** architecture.

The model is fine-tuned on the **Fengyun-2F** satellite's orbital data to predict **eccentricity** and detect anomalies based on prediction errors.

---

## 📌 Overview

- **Model**: TinyTimeMixer (TSFM)
- **Data**: Fengyun-2F satellite telemetry
- **Target**: Eccentricity
- **Observables**: Argument of perigee, inclination, mean anomaly, Brouwer mean motion, right ascension
- **Forecasting Parameters**:
  - Context Length: 512
  - Forecast Length: 96
  - Few-shot Fraction: 0.02
- **Output**:
  - Fine-tuned TSFM model
  - Anomaly scores
  - Forecast visualizations

---

## 📂 Project Structure

```
├── IBM.ipynb                # Main Jupyter Notebook
├── Fengyun_2F.csv           # Satellite dataset
├── results/                 # Anomaly plots and outputs
└── README.md                # Documentation
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/tsfm-anomaly-detection.git
cd tsfm-anomaly-detection
```

2. Install dependencies:
```bash
pip install numpy==1.26.4 scikit-learn torch torchvision torchaudio
```

3. Clone and install IBM TSFM:
```bash
git clone --depth 1 --branch v0.2.9 https://github.com/IBM/tsfm.git
cd tsfm
pip install ".[notebooks]"
cd ..
```

---

## 🚀 Usage

1. Place your dataset in the root folder (CSV format).
2. Open the notebook:
```bash
jupyter notebook IBM.ipynb
```
3. Adjust:
   - `context_length` and `forecast_length`
   - `target_columns` and `observable_columns`
   - Split ratios in `split_config`
4. Run all cells to:
   - Load and preprocess data
   - Fine-tune TinyTimeMixer
   - Compute anomaly scores
   - Visualize anomalies

---

## 📊 Example Output

Example anomaly detection plot:

![Example Anomaly Plot](results/example_plot.png)

---

## 🛠 Configuration

- **`context_length`**: Number of past timesteps used for forecasting.
- **`forecast_length`**: Number of future timesteps predicted.
- **`fewshot_fraction`**: Fraction of training data for few-shot fine-tuning.
- **`scaler_type`**: Data scaling method (`minmax`, `standard`).

---

## 📚 References

- [IBM TSFM GitHub](https://github.com/IBM/tsfm)
- [TinyTimeMixer Paper](https://arxiv.org/abs/2305.XXXXX)
- [IBM Ignite](https://www.ibm.com/ignite)

---

## ✨ Acknowledgements

- IBM Research for TSFM
- Fengyun-2F dataset providers
- Open-source contributors to anomaly detection research


---

## 2️⃣ Rolling Auto ARIMA – Anomaly Detection

# Rolling Auto ARIMA – Satellite Anomaly Detection (Fengyun‑2F Eccentricity)

This repository/notebook implements a **Rolling Auto ARIMA** pipeline to detect anomalies in **satellite orbital elements**, with a focus on **Fengyun‑2F eccentricity**. The method uses **pmdarima’s `auto_arima`** to select optimal ARIMA orders and a **rolling/expanding forecast** loop to generate errors; anomalies are flagged when the absolute residual exceeds a **percentile-based threshold** (default: 95th).

> If you are combining this with a TSFM notebook, keep this README alongside a separate `README.md` for TSFM, or make a top-level README that links to both methods.

---

## ✨ What this notebook does

- Loads Fengyun‑2F orbital data (eccentricity + other elements).  
- Cleans and aligns timestamps; forward-fills small gaps (naïve completion).  
- Explores stationarity (ADF test), rolling mean/std, and differencing/log transforms as needed.  
- Splits a single time series into **train / validation / test** by index ranges.  
- Fits **`auto_arima`** on a training window and **rolls forward**: predict → compute residual → (optionally) update/refit.  
- Computes an **anomaly threshold** from the validation residual distribution (e.g., **95th percentile**).  
- Flags test points as anomalous if `|residual| >= threshold`.  
- Produces plots for the raw series, rolling statistics, train/val/test spans, predictions, residuals, and anomaly markers.

---

## 🧱 Data & Target

- **Dataset**: Fengyun‑2F orbital elements (CSV/Parquet).  
- **Target variable**: `eccentricity`.  
- **Other columns often explored**: `argument_of_perigee`, `inclination`, `mean_anomaly`, `Brouwer_mean_motion`, `right_ascension`, etc.  
- **Datetime handling**: the timestamp column is renamed to `Datetime` and set as a **pandas DateTimeIndex**.

> Tip: Keep the sampling frequency consistent (e.g., hourly/daily). The notebook includes basic forward fill for small gaps.

---

## ⚙️ Method Overview

1. **Preprocessing**
   - Rename timestamp → `Datetime`, set as index.
   - Optional: log transform, differencing, and stationarity checks (ADF).
   - Plot rolling mean & std to visualize drift/variance.

2. **Train/Validation/Test Split**
   - Choose indices for `train`, `val`, `test` (e.g., via `iloc` spans).  
   - `train` for initial fit, `val` for threshold calibration, `test` for final anomaly detection.

3. **Rolling Forecast with `auto_arima`**
   - Fit **`auto_arima`** on the training slice.
   - **Roll** over the validation/test window(s): one-step forecast → compute residual (`y - ŷ`).
   - Strategy:
     - **Expanding window** (append new point and refit/update)
     - or **Fixed window** (slide window of constant length)

4. **Anomaly Threshold**
   - Compute residuals on the **validation** period.
   - Set threshold = **percentile** (default: 95th) of `|residual|`.
   - Flag anomalies where `|residual_test| >= threshold`.

---

## 📦 Dependencies

- `pandas`, `numpy`
- `matplotlib`
- `pmdarima` (for `auto_arima`)
- `statsmodels` (ADF, optional diagnostics)
- `scikit-learn` (optional utilities)

Install (example):
```bash
pip install pandas numpy matplotlib pmdarima statsmodels scikit-learn
```

---

## 🗂️ Project Structure (suggested)

```
├── Rolling_Auto_Arima.ipynb   # This notebook
├── data/
│   └── fengyun_2f.csv            # Your time series (timestamp + columns)
├── results/
│   ├── predictions.csv           # Optional: saved forecasts/residuals
│   ├── anomalies.csv             # Optional: saved anomaly indices
│   └── plots/                    # Saved figures
└── README.md  # This file
```

---

## 🚀 How to Run

1. Put your dataset under `data/` and update the load path in the notebook cell that reads the CSV/Parquet.
2. Open the notebook:
   ```bash
   jupyter notebook Rolling_Auto_Arima.ipynb
   ```
3. In the **parameters** cells, review and adjust:
   - **Split indices**: `train_start`, `train_end`, `val_start`, `val_end`, `test_start`, `test_end`
   - **Rolling strategy**: expanding vs fixed window
   - **`auto_arima` search space**: `start_p/q`, `max_p/q`, seasonal flags if needed
   - **Threshold percentile**: default `95` (use `90–99` depending on sensitivity)
   - **Transforms**: log/diff if ADF suggests non-stationarity
4. Run all cells to:
   - Fit ARIMA, forecast, compute residuals
   - Calibrate threshold
   - Produce anomaly flags and plots

---

## 📊 Outputs

- **Forecast vs Actual plots** with train/val/test shading
- **Residual series** and **percentile threshold** line
- **Anomaly markers** on the test set
- (Optional) CSV exports for `predictions`, `residuals`, and `anomalies`

---

## 🔧 Key Parameters (common defaults)

- `threshold_percentile = 95`
- `rolling_window = expanding` (update with each new point)
- `auto_arima` options (tune per dataset):
  - `start_p=0, start_q=0, max_p=3, max_q=3`
  - `seasonal=False` (set `True` and specify `m` if seasonal pattern exists)
  - `information_criterion="aic"`
  - `stepwise=True` (faster search)

> If the series shows strong seasonality, consider **SARIMA** via `seasonal=True, m=period`.

---

## ✅ Evaluation Ideas

- **MAE/RMSE** on validation/test predictions
- **Precision/Recall** of anomaly labels (if you have ground truth)
- **Residual distribution** sanity checks (Q–Q plots, normality tests)
- **Backtesting** with multiple folds to validate threshold robustness

---

## 📝 Notes & Tips

- **Differencing** can stabilize the mean; log transform stabilizes variance.
- Use **ADF** (Augmented Dickey-Fuller) to test stationarity before ARIMA.
- **Percentile thresholding** is simple and robust; for adaptive detection, try **rolling MAD** or **EWMA** of residuals.
- Refit/update cadence is a trade‑off: frequent refits can improve accuracy but increase compute time.

---

## 📚 References

- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*
- `pmdarima` documentation: https://alkaline-ml.com/pmdarima/
- `statsmodels` time series: https://www.statsmodels.org/

---

## 📝 License

MIT (or update to your preferred license).


---

## 🔗 References

- IBM TSFM GitHub: https://github.com/IBM/tsfm
- pmdarima docs: https://alkaline-ml.com/pmdarima/
- Hyndman & Athanasopoulos – *Forecasting: Principles and Practice*
- statsmodels time series: https://www.statsmodels.org/

---

## 📜 License

MIT (or update to your preferred license).
