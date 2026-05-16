# 🌫️ Air Quality Forecasting with LLM-Based Decision Support

An end-to-end Environmental AI system that predicts **PM2.5 air pollution levels** using deep learning models and generates grounded, explainable health recommendations using **Large Language Models (LLMs)**.

---

## 📌 Overview

This project combines:

- **Time-series forecasting**
- **AQI-based environmental risk analysis**
- **LLM-powered explanation generation**
- **Hallucination auditing and validation**

to create a trustworthy air-quality decision-support pipeline.

The system forecasts hourly PM2.5 concentration levels, converts them into AQI categories, and generates actionable recommendations for users such as runners, cyclists, schools, and sensitive populations.

---

## 🚀 Features

### 🔹 Forecasting Models
- LSTM-based PM2.5 prediction
- ISSA-LSTM hyperparameter optimization
- AR(24) statistical baseline
- Persistence forecasting baseline

### 🔹 Explainability & Decision Support
- AQI category conversion using EPA standards
- LLM-generated health recommendations
- User-facing environmental guidance
- Interactive QA-based recommendation system

### 🔹 Reliability & Safety
- Hallucination detection framework
- AQI consistency validation
- Numerical grounding checks
- Physical-range verification
- Structured JSON-based output auditing

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| ML Libraries | Scikit-learn, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| LLM | Qwen2.5-7B-Instruct |
| Research Tools | Jupyter Notebook |

---

## 📂 Dataset

### Beijing PM2.5 Dataset
Source: UCI Machine Learning Repository

Features used:
- PM2.5
- Temperature
- Dew Point
- Pressure
- Wind Speed

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Missing-value imputation
- MinMax normalization
- Chronological train/validation/test split
- 60-hour lookback sequence generation

### 2️⃣ Feature Selection
- mRMR (Minimum Redundancy Maximum Relevance)
- Random Forest feature importance analysis

### 3️⃣ Forecasting
- Baseline LSTM
- ISSA-LSTM optimization
- AR(24)
- Persistence model

### 4️⃣ AQI Mapping
- Deterministic AQI conversion using EPA standards

### 5️⃣ LLM Explanation Layer
The LLM receives:
```json
{
  "predicted_pm25_ugm3": "...",
  "predicted_aqi": "...",
  "aqi_category": "...",
  "top_features": {...}
}
```

and generates:
- Explanation
- Primary pollution driver
- Health recommendation
- Confidence note

### 6️⃣ Hallucination Audit
The system validates:
- PM2.5 consistency
- AQI consistency
- AQI category correctness
- Feature grounding
- Physical value ranges
- Confidence thresholds

---

## 📊 Results

| Model | RMSE |
|---|---|
| AR(24) | 21.67 |
| Persistence | 22.02 |
| Baseline LSTM | 24.53 |
| ISSA-LSTM | 28.78 |

### LLM Evaluation
- ✅ 0.0% hard hallucination rate
- ✅ 100% AQI consistency
- ✅ 100% category consistency
- ✅ Structured explanation generation

---

## 🔍 Key Contributions

- Built a reproducible PM2.5 forecasting pipeline
- Integrated LLM-based environmental explanation generation
- Designed a deterministic hallucination-audit framework
- Compared deep learning models against strong statistical baselines
- Conducted qualitative QA analysis for real-world decision-support scenarios
- Authored a full IEEE-style research paper for the project

---

## 📈 Future Improvements

- Transformer-based time-series forecasting
- Retrieval-Augmented Generation (RAG)
- Uncertainty-aware AQI prediction
- Real-time deployment pipeline
- Event-weighted loss functions
- Multi-city forecasting support

---

## 📖 Research Focus Areas

- Environmental AI
- Time-Series Forecasting
- Explainable AI (XAI)
- Large Language Models
- AI Safety & Hallucination Detection
- Decision Support Systems

---

## 📜 License

This project is intended for academic and research purposes.
