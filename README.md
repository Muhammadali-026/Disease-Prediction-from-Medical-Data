# 🫀 Disease Prediction from Medical Data

> **CodeAlpha Machine Learning Internship — Task 4**  
> Developer: **Muhammad Ali** | Platform: **Google Colab**

A machine learning web app that predicts whether a patient has **heart disease** based on 13 clinical measurements. Four ML models are trained, compared, and the best one is deployed in an interactive **Gradio** interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [ML Models](#ml-models)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [UI Tabs](#ui-tabs)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

This project takes patient medical data (age, cholesterol, blood pressure, etc.) and predicts whether the patient has heart disease or not. It trains **4 different ML models**, evaluates them using multiple metrics, picks the best performer by ROC-AUC score, and deploys it in a live **Gradio** web interface — all inside Google Colab.

---

## 🎥 Demo

> Run the notebook on Google Colab and a public shareable link will be generated automatically via `app.launch(share=True)`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AXHQVYPYViNJDRSvryWa8CpPvgufypiM)

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | Cleveland Heart Disease (UCI ML Repository) |
| Source | [GitHub Mirror](https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv) |
| Patients | 303 |
| Features | 13 (+ 1 target) |
| Target | Binary — `1` = Heart Disease, `0` = No Heart Disease |
| Missing Values | None |
| Train / Test Split | 80% / 20% (stratified) |

---

## 🔬 Features Used

| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `sex` | 1 = Male, 0 = Female |
| `chest_pain` | Chest pain type (0–3) |
| `blood_pressure` | Resting blood pressure (mm Hg) |
| `cholesterol` | Serum cholesterol (mg/dl) |
| `blood_sugar` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `ecg` | Resting ECG result (0–2) |
| `max_heart_rate` | Maximum heart rate achieved (bpm) |
| `exercise_angina` | Exercise-induced angina (1 = Yes) |
| `st_depression` | ST depression induced by exercise vs rest |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `vessels` | Number of major vessels coloured (0–3) |
| `thal` | 0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect |

---

## 🤖 ML Models

Four models are trained and compared:

| Model | Key Parameters | Notes |
|-------|----------------|-------|
| **Logistic Regression** | `max_iter=1000, C=1.0` | Fast, interpretable baseline; uses scaled data |
| **Random Forest** | `n_estimators=200, max_depth=8` | Ensemble of 200 decision trees |
| **XGBoost** | `n_estimators=200, max_depth=4, lr=0.1` | Gradient boosting; often highest accuracy |
| **SVM** | `kernel=rbf, C=1.0` | Finds optimal decision boundary; uses scaled data |

The model with the **highest ROC-AUC score** on the test set is automatically selected as the winner. Expected best model accuracy: **~85–90%**, ROC-AUC: **~0.90–0.95**.

---

## ⚙️ How It Works

```
Patient Data (13 features)
        ↓
  Preprocessing
  • Fill missing values (median)
  • StandardScaler normalization
        ↓
  Train / Test Split (80/20, stratified)
        ↓
  Train 4 Models simultaneously
        ↓
  Evaluate on Test Set
  • Accuracy · Precision · Recall · F1 · ROC-AUC
        ↓
  Best Model Selected (by AUC)
        ↓
  Gradio Web App deployed
        ↓
  User inputs patient data → Prediction + Probability
```

**Prediction output example:**
```python
proba = model.predict_proba(patient_data)
# → [0.23, 0.77]  means 77% chance of Heart Disease
```

---

## 📁 Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── task_4___disease_prediction_from_medical_data.py   # Main Colab script
├── README.md
│
└── /tmp/  (auto-generated during runtime)
    ├── eda_charts.png            # Exploratory Data Analysis charts
    ├── model_comparison.png      # All 4 models — metrics bar chart
    ├── roc_curves.png            # ROC-AUC curves for all models
    ├── feature_importance.png    # RF vs XGBoost feature importance
    ├── best_model_evaluation.png # Confusion matrix of best model
    └── best_disease_model.pkl    # Saved best model
```

---

## 🚀 Installation & Usage

### Run on Google Colab (Recommended)

1. Open the notebook link above in Google Colab.
2. Run all cells (`Runtime → Run All`).
3. A public shareable Gradio link will appear at the end — click it to use the app.

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

# Install dependencies
pip install xgboost scikit-learn matplotlib seaborn pandas gradio

# Run the script
python task_4___disease_prediction_from_medical_data.py
```

Then open `http://localhost:7860` in your browser.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | ~85% | ~84% | ~86% | ~85% | ~0.91 |
| Random Forest | ~87% | ~86% | ~88% | ~87% | ~0.93 |
| XGBoost | ~88% | ~87% | ~89% | ~88% | ~0.94 |
| SVM | ~84% | ~83% | ~85% | ~84% | ~0.91 |

> *Actual values may vary slightly by run. Cross-validation accuracy: ~82–88%.*

---

## 🖥️ UI Tabs

The Gradio app has **5 tabs**:

| Tab | What it shows |
|-----|---------------|
| 🔍 **Predict** | Enter patient data via sliders → get prediction + probability |
| 📊 **Model Comparison** | Bar charts and metrics table for all 4 models |
| 🔑 **Feature Importance** | RF vs XGBoost feature ranking + confusion matrix |
| 📈 **Data Analysis** | EDA charts — class distribution, age, sex, cholesterol, etc. |
| ⚙️ **How It Works** | Step-by-step explanation of the ML pipeline |

---

## ⚠️ Disclaimer

> This tool is built **for educational purposes only** as part of the CodeAlpha AI/ML Internship.  
> It is **not** a medical device and should **not** be used for clinical diagnosis.  
> A real clinical tool requires regulatory approval, larger validated datasets, and review by qualified medical professionals.

---

## 🙌 Acknowledgements

- [CodeAlpha](https://www.codealpha.tech) — Internship program
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — Cleveland Heart Disease Dataset
- [Scikit-learn](https://scikit-learn.org) · [XGBoost](https://xgboost.readthedocs.io) · [Gradio](https://gradio.app)

---

## 📬 Contact — CodeAlpha

- 🌐 Website: [www.codealpha.tech](https://www.codealpha.tech)
- 📧 Email: services@codealpha.tech
- 💬 WhatsApp: +91 9336576683
