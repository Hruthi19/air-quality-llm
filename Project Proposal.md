**Enhancing Air Quality Prediction with**

**LLM-Based Explanation and Decision Support**

*Hruthi Muggalla, Akanksha Karra, Atharva Nilkanth, Sai Krishna Ghanta*

# **1\. Task Description**

This project aims to enhance an existing air quality prediction model by incorporating a Large Language Model (LLM) as a post-processing interpretability layer. The base system employs a hybrid machine learning architecture that combines minimum Redundancy Maximum Relevance (mRMR) and Random Forest (RF) for feature selection with an Improved Sparrow Search Algorithm (ISSA)-optimized Long Short-Term Memory (LSTM) network for Air Quality Index (AQI) prediction \[1\]. While the existing model demonstrates strong numerical prediction accuracy, it lacks human-interpretable outputs suitable for non-technical stakeholders. This project proposes augmenting the pipeline with an LLM layer that receives the predicted AQI value alongside key environmental and meteorological features, and generates natural language explanations, actionable health recommendations, and an interactive question-answering interface. The ultimate goal is to bridge the gap between raw predictive output and real-world decision support for users, policymakers, and health-conscious individuals.

# **2\. Motivation and Main Challenges**

Air pollution is a leading environmental risk factor globally, contributing to millions of premature deaths annually. The ability to accurately forecast the AQI enables timely interventions for public health protection. However, high-performing deep learning models such as LSTM are inherently black-box in nature, producing only a numerical output without contextual reasoning. This creates a critical gap between prediction and practical action, particularly for stakeholders who lack a data science background. The motivation behind this project is to make AI-driven AQI predictions usable, transparent, and actionable. By integrating an LLM layer, the system can translate complex model outputs into plain-language insights that inform everyday decisions, such as whether to exercise outdoors, adjust commuting plans, or issue public health advisories. The main technical challenges include:

• Lack of interpretability: The base LSTM model does not communicate which pollutants or meteorological conditions drive a given AQI prediction, limiting user trust and understanding.

• Integration of structured and unstructured reasoning: Combining precise numerical model outputs with the flexible, open-ended generation capabilities of an LLM requires careful prompt engineering to ensure factual grounding.

• Reliability and hallucination prevention: LLM outputs must be constrained to reflect the actual data values passed in, avoiding fabricated statistics or unsupported health claims. 

**3\. Dataset**

The project will use the UCI Air Quality Dataset \[7\], a widely used benchmark for air pollution time-series modeling. This dataset contains hourly averaged responses from an array of metal oxide chemical sensors deployed in a polluted area of an Italian city, alongside reference analyzer measurements. It covers the following feature categories:

 	•  Air pollutant concentrations: PM2.5, PM10, NO2, SO2, CO, O3, NOx

•  Meteorological variables: Temperature, relative humidity, wind speed, visibility

 Preprocessing will include normalization using min-max scaling, missing value imputation via forward-fill and linear interpolation, and temporal resampling where necessary. Feature selection will be applied using mRMR to reduce redundancy among correlated pollutant variables, followed by RF-based importance ranking to identify the top contributing predictors passed into the LLM prompt.

# **4\. Related Work**

The foundation of this project is the work by Wu et al. (2023) \[1\], which introduces a hybrid air quality prediction model combining mRMR-based feature selection with an ISSA-optimized LSTM network. The ISSA improves on the standard Sparrow Search Algorithm by incorporating adaptive position updates and Levy flight strategies, enabling more effective hyperparameter optimization for the LSTM. This model achieves state-of-the-art accuracy on standard AQI benchmark datasets and serves as the core prediction engine in our pipeline. 

Long Short-Term Memory networks, originally proposed by Hochreiter and Schmidhuber (1997) \[2\], have become the standard architecture for sequential and time-series forecasting tasks. Their gated memory cell mechanism allows them to capture long-range temporal dependencies in environmental data, making them particularly well-suited to AQI prediction where pollution levels exhibit both diurnal cycles and multi-day weather-driven trends. Numerous subsequent works have applied LSTM variants to air quality forecasting, consistently demonstrating superior performance over traditional statistical methods such as ARIMA. Random Forests, introduced by Breiman (2001) \[3\], are ensemble learning methods that aggregate predictions from multiple decision trees to improve generalization. In the context of this project, RF is used not only for feature importance ranking via mean decrease in impurity, but also as a complementary predictor to validate the LSTM output. The integration of Large Language Models into machine learning workflows has gained significant traction following the introduction of GPT-3 by Brown et al. (2020) \[4\], which demonstrated that large-scale language models can perform complex reasoning tasks with minimal task-specific fine-tuning. More recent work on Retrieval-Augmented Generation (RAG) has shown that grounding LLM outputs with external knowledge bases substantially reduces hallucination rates and improves factual accuracy, which is especially important in health-critical applications such as air quality advisories. These advances make LLMs a viable and powerful choice for post-hoc explanation in structured prediction pipelines.

**5\. LLM Integration Plan**

## **5.1 Inputs to the LLM**

•  Predicted AQI value and corresponding health category (Good / Moderate / Unhealthy / Hazardous)

•  Top contributing pollutant parameters with their measured and normalized values (e.g., PM2.5, NO2)

•  Meteorological context values (temperature, humidity, wind speed)

•  RF-derived feature importance scores for the top-k features

•  Optional: User's natural language query for the interactive QA mode

## **5.2 Outputs from the LLM**

Example explanation: "The predicted AQI is Unhealthy (142) primarily due to elevated PM2.5 concentrations (58 ug/m3), which account for 47% of the model's feature importance score. Low wind speeds (1.2 m/s) are preventing effective dispersion of particulate matter."

Example recommendation: "It is advisable to limit prolonged outdoor exertion, particularly for children, the elderly, and individuals with respiratory conditions. Consider wearing an N95 mask if outdoor activity is necessary."

Example interactive QA: User query: "Is it safe to go for a run today?" — LLM response contextualizes the AQI prediction against activity-specific health thresholds.

# **6\. Four-Week Project Plan**

| Week | Focus | Key Tasks |
| :---- | :---- | :---- |
| **Week 1** | **Data & Base Model Setup** | •   Obtain and preprocess UCI Air Quality dataset •   Handle missing values and normalize features •   Implement mRMR feature selection pipeline |
| **Week 2** | **LLM Integration Design** | •   Design prompt engineering strategy with AQI  •   Map RF feature importances to language explanations • Define structured JSON output schema |
| **Week 3** | **System Build & QA Module** | •   Build full end-to-end pipeline (LSTM → LLM → output) •   Implement interactive QA system for user queries |
| **Week 4** | **Evaluation & Final Report** | • Evaluate LLM output quality (relevance, accuracy, hallucination rate) |

**7\. Expected Contributions**

• A fully integrated pipeline combining ISSA-LSTM time-series prediction with LLM-based natural language explanation, applicable beyond air quality to other environmental monitoring domains.

•  A prompt engineering framework for grounding LLM outputs in structured ML model outputs, with explicit hallucination prevention via hard-coded domain thresholds (EPA/WHO AQI standards).

•  An interactive QA module that enables non-expert users to query the system in plain language, lowering the barrier to using AI-powered environmental tools.

• A reproducible evaluation protocol for assessing LLM explanation quality in the context of scientific prediction models, including factual consistency and actionability metrics.

**8\. References**

**\[1\]** Wu, H., Yang, T., Li, H., & Zhou, Z. (2023). Air quality prediction model based on mRMR-RF feature selection and ISSA-LSTM. Scientific Reports, 13, 1-15. https://doi.org/10.1038/s41598-023-xxxxx

**\[2\]** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

**\[3\]** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

**\[4\]** Brown, T., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems (NeurIPS), 33, 1877-1901.

**\[5\]** Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 33, 9459-9474.

**\[6\]** Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS, 30\.

**\[7\]** De Vito, S., et al. (2008). UCI Air Quality Dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/air+quality

**\[8\]** OpenAQ Platform. Open-source air quality data. https://openaq.org/

