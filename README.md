# 💳 AI Credit Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning application for estimating serious credit delinquency risk from customer financial and credit-history information.

## 🚀 Live Demo

### [Open AI Credit Risk Predictor](https://ai-credit-risk-predictor0.streamlit.app/)

---

## 📌 Overview

**AI Credit Risk Predictor** is a machine learning web application that predicts the probability of serious credit delinquency.

The project covers:

```text
Data
 ↓
Preprocessing
 ↓
Model Training
 ↓
Hyperparameter Tuning
 ↓
Threshold Analysis
 ↓
Evaluation
 ↓
Model Persistence
 ↓
Streamlit Deployment

The deployed application uses a saved ML pipeline and does not retrain the model.

---

## 🚀 Features

* 📊 Interactive model performance dashboard
* 🧮 Individual customer risk assessment
* 📁 Batch CSV prediction
* 📈 ROC-AUC, PR-AUC, Precision, Recall & F1 metrics
* 🔎 Global feature importance
* 📥 Download batch prediction results
* ⚙️ Configurable prediction threshold
* 💾 Saved Scikit-learn ML pipeline

---

## 📊 Model Performance

Final out-of-sample test results:

| Metric             |      Score |
| ------------------ | ---------: |
| ROC-AUC            | **0.8644** |
| PR-AUC             | **0.3859** |
| F1 Score           | **0.3416** |
| Precision          | **22.03%** |
| Recall             | **76.01%** |
| Decision Threshold |   **0.50** |

The dataset is highly imbalanced:

* **93.32%** — No serious delinquency
* **6.68%** — Serious delinquency

Because of this imbalance, the project focuses on ROC-AUC, PR-AUC, precision, recall, and F1 rather than accuracy alone.

---

## 🔎 Top Risk Drivers

| Rank | Feature               | Importance |
| ---: | --------------------- | ---------: |
| 🥇 1 | Revolving Utilization | **0.3709** |
| 🥈 2 | 90+ Days Late         | **0.2106** |
| 🥉 3 | 30–59 Days Past Due   | **0.1895** |

> Feature importance represents the model's global behavior and should not be interpreted as causation or as an individual customer's explanation.

---

## 🧠 Machine Learning

### Models Evaluated

* Logistic Regression
* Random Forest

### Selected Model

**Tuned Random Forest Classifier**

### Preprocessing

* Missing-value handling using median imputation
* Train / validation / test split
* Cross-validation
* Randomized hyperparameter search
* Decision-threshold analysis

### Model Persistence

The trained pipeline is saved using Joblib:

models/
├── credit_risk_model.pkl
└── credit_risk_config.pkl

---

## 🗂️ Dataset

### Give Me Some Credit

| Property          |                 Value |
| ----------------- | --------------------: |
| Records           |               150,000 |
| Raw columns       |                    12 |
| Modeling features |                    10 |
| Task              | Binary Classification |
| Target            |    `SeriousDlqin2yrs` |

### Features

RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberOfTimes90DaysLate
NumberRealEstateLoansOrLines
NumberOfTime60-89DaysPastDueNotWorse
NumberOfDependents

---

## 🖥️ Application

The Streamlit application contains five sections:

📊 Dashboard
🧮 Assessment
📁 Batch Prediction
📈 Performance
ℹ️ About

---

## 📁 Project Structure

```text
AI-Credit-Risk-Predictor/
│
├── data/
│   └── cs-training.csv
│
├── models/
│   ├── credit_risk_model.pkl
│   └── credit_risk_config.pkl
│
├── notebooks/
│   └── credit_risk_model.ipynb
│
├── screenshots/
│   ├── dashboard.png
│   ├── assessment.png
│   ├── batch_prediction.png
│   └── performance.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## ⚙️ Installation

### Clone the repository

git clone https://github.com/<your-username>/AI-Credit-Risk-Predictor.git
cd AI-Credit-Risk-Predictor

### Create virtual environment

**Windows**

python -m venv .venv
.venv\Scripts\activate

**macOS / Linux**

python3 -m venv .venv
source .venv/bin/activate

### Install dependencies

pip install -r requirements.txt

### Run the application

streamlit run app.py

---

## ⚠️ Limitations

This project is intended for **educational and portfolio purposes**.

* Dataset may not represent modern lending environments.
* Predictions are statistical estimates, not causal conclusions.
* Global feature importance is not an individual explanation.
* Real-world lending systems require fairness, calibration, monitoring, security, governance, and domain validation.

**This application should not be used as the sole basis for actual financial or lending decisions.**

---

## 🔮 Future Improvements

* SHAP-based individual explanations
* Probability calibration
* Fairness & bias evaluation
* Model drift monitoring
* Data-quality monitoring
* FastAPI inference API
* Docker deployment
* CI/CD pipeline
* Cloud deployment
* Authentication & audit logging
* Automated retraining

---

## 👨‍💻 Author

**Manthan Pawar**

AI/ML Student | Python | Machine Learning | AI Engineering

* GitHub: `https://github.com/ManthanPawar04`
* LinkedIn: `www.linkedin.com/in/manthanpawar4`

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgement

Dataset: **Give Me Some Credit**

Built as an end-to-end machine learning portfolio project covering data preparation, model training, evaluation, model persistence, and Streamlit deployment.