Air Quality Forecasting with LLM-Based Decision Support

An end-to-end environmental AI system that predicts PM2.5 air pollution levels using deep learning and generates grounded health recommendations using Large Language Models (LLMs).

Overview

This project combines time-series forecasting, AQI conversion, and LLM-based explanation generation to create a trustworthy air-quality decision-support pipeline.

The system:

Forecasts hourly PM2.5 concentration using LSTM-based models
Converts predictions into AQI categories
Generates natural-language health recommendations using Qwen2.5-7B-Instruct
Audits LLM responses using deterministic hallucination checks
Features
PM2.5 forecasting using:
LSTM
ISSA-LSTM optimization
AR(24) baseline
Persistence baseline
Feature selection using mRMR and Random Forest importance
AQI category mapping based on EPA guidelines
LLM-generated explanations and recommendations
Hallucination detection and grounding validation
Interactive QA-based decision support analysis
Tech Stack
Python
PyTorch
Scikit-learn
Pandas / NumPy
Qwen2.5-7B-Instruct
Matplotlib / Seaborn
Dataset
Beijing PM2.5 Dataset (UCI Machine Learning Repository)
Key Contributions
Built a reproducible environmental forecasting pipeline
Integrated LLM-based explanation generation with structured outputs
Designed a hallucination-audit framework for grounded AI responses
Evaluated forecasting performance against strong statistical baselines
Conducted qualitative analysis for real-world AQI decision support
Results
AR(24) achieved the best RMSE performance
LLM explanation layer achieved:
0.0% hard hallucination rate
100% AQI/category consistency under implemented checks
Future Improvements
Transformer-based forecasting models
Retrieval-augmented AQI recommendations
Uncertainty-aware forecasting
Event-weighted loss functions
Real-time deployment and monitoring
