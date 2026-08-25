# 🛰️ Pakistan Air Intelligence

**AI-Powered Air Quality Monitoring & PM2.5 Prediction Platform**

An interactive Streamlit dashboard for exploring 90 days of hourly air-quality and weather data across 10 major Pakistani cities, with a trained Gradient Boosting model that forecasts next-hour PM2.5 concentrations.

---

## ✨ Features

- **Overview Dashboard** — dataset-wide KPIs (records, cities, avg/max PM2.5, avg PM10, date range) and AQI category distribution
- **AI PM2.5 Prediction** — pick a city and hour, optionally tweak readings, and get a next-hour PM2.5 forecast from the trained model, alongside the real historical value for that hour as a sanity check
- **City Intelligence** — per-city stats and a pollutant-by-pollutant pollution ranking across all 10 cities
- **Air Quality Analytics** — PM2.5/PM10 trends, multi-pollutant comparison, and weather-vs-pollution relationships (scatter + correlation heatmap)
- **Trends & Patterns** — hourly, day-of-week, monthly, and seasonal pollution patterns, plus a City × Hour heatmap
- **Model Intelligence** — Gradient Boosting vs. GRU vs. LSTM comparison and feature importance, both read live from the project's result files
- **Data Explorer** — filter by city, AQI category, season, date, and hour, then export the filtered data as CSV
- **About** — dataset background and the project's data-quality correction report

All values are computed dynamically from the dataset and model artifacts — nothing is hardcoded or fabricated.

---

## 🧠 Model

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Tuned Gradient Boosting** ⭐ | 4.17 | 6.59 | **0.9795** |
| GRU | 9.92 | 14.00 | 0.9083 |
| LSTM | 20.77 | 32.74 | 0.4983 |

The Gradient Boosting model predicts **next-hour PM2.5** using the current hour's pollutant and weather readings plus engineered lag, rolling-window, and cyclical time features (see the training notebook for full details). The app never retrains the model — it loads the existing `.joblib` artifacts and only performs inference.

---

## 📁 Project Structure

```text
Pakistan Air Intellignece/
│
├── app.py                                  # Streamlit application (entry point)
├── requirements.txt                        # Python dependencies
│
└── assets/
    ├── dataset/
    │   ├── DATA_QUALITY_REPORT.md           # v2 data-quality correction report
    │   └── pakistan_air_quality_final_clean_v2.csv
    │
    ├── models/
    │   ├── feature_importance.csv
    │   ├── final_model_comparison.csv
    │   ├── pm25_preprocessor.joblib         # fitted ColumnTransformer
    │   └── tuned_gradient_boosting_pm25.joblib
    │
    ├── pakistan-air-intelligence.ipynb      # full EDA + feature engineering + training notebook
    └── Pakistan_Air_Intelligence_SRS.pdf    # software requirements specification
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone <your-repo-url>
cd "Pakistan Air Intellignece"
pip install -r requirements.txt
```

> ⚠️ **Important:** the `.joblib` model files were saved with `scikit-learn==1.6.1`. Installing a newer scikit-learn version can raise an `AttributeError` when unpickling them — `requirements.txt` pins the correct version.

### Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Dataset

- **Coverage:** 10 cities — Faisalabad, Islamabad, Karachi, Lahore, Multan, Peshawar, Quetta, Rahim Yar Khan, Rawalpindi, Sialkot
- **Period:** Nov 6, 2025 – Feb 4, 2026 (hourly resolution)
- **Size:** 21,840 rows × 26 columns
- **Fields:** PM2.5, PM10, CO, NO₂, SO₂, O₃, dust, temperature, humidity, precipitation, wind speed/direction, pressure, plus derived time and AQI-category fields

Version `v2` includes a documented correction: weather columns were flat/repeated for a 21-day window at the start of collection across all cities; this was identified, root-caused, and re-fetched from Open-Meteo's historical weather API. Pollutant data was unaffected throughout. Full details are in `assets/dataset/DATA_QUALITY_REPORT.md` and surfaced in the app's **About** page.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, custom CSS
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **ML:** scikit-learn (Gradient Boosting), joblib

---

## ⚠️ Disclaimer

Predictions shown in this app are AI-generated estimates based on historical environmental patterns and the supplied input conditions. They are not certified air-quality readings and should not be used for medical or regulatory decisions.

---

## 👤 Author

**Abdul Rehman**
