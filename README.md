# ✈️ Airline Passenger Referral Prediction

> **Binary Classification · NLP · FastAPI · Streamlit Deployment**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Joblib](https://img.shields.io/badge/Model-Joblib%20%7C%20Pickle-4CAF50?style=flat-square)

---

## 📌 Project Overview

In the fiercely competitive aviation industry, **word-of-mouth referrals** are one of the most powerful drivers of growth. This project builds a machine learning classification system that **predicts whether a passenger will recommend an airline** — enabling airlines to prioritize customer engagement, tailor loyalty programmes, and convert satisfied travellers into brand advocates.

**Key business applications:**
- Enhance customer service by targeting potential advocates
- Tailor marketing and loyalty programs to high-referral passengers
- Make data-driven decisions to improve service quality
- Drive revenue growth through increased positive referrals

---

## 📊 Model Performance

| Model | Accuracy | Recall | F1 Score | AUC-ROC |
|-------|----------|--------|----------|---------|
| Logistic Regression | 91% | 88% | 89% | 0.91 |
| Random Forest | 93% | 91% | 92% | 0.94 |
| **SVM (RBF) ⭐ Deployed** | **95%** | **94%** | **94%** | **0.96** |

> **Priority metric: Recall** — minimizing false negatives is critical to avoid missing potential brand advocates.

---

## 🗂️ Dataset

**File:** `data_airline_reviews.xlsx`

| Column | Description |
|--------|-------------|
| `airline` | Airline name |
| `overall` | Overall rating (1–10) |
| `customer_review` | Open-text review (NLP feature) |
| `traveller_type` | Business / Solo / Couple / Family |
| `cabin` | Economy / Business / First |
| `seat_comfort` | Rating (1–5) |
| `cabin_service` | Rating (1–5) |
| `food_bev` | Rating (1–5) |
| `entertainment` | Rating (1–5) |
| `ground_service` | Rating (1–5) |
| `value_for_money` | Rating (1–5) |
| `recommended` | **Target** — yes / no |

---

## 🔑 Top Predictive Features (ELI5 Permutation Importance)

```
ground_service   ████████████████████  0.0805
cabin_service    ███████████████████   0.0756
seat_comfort     ████████████          0.0497
food_bev         ███████               0.0299
entertainment    █                     0.0016
traveller_type   █                     0.0011
```

---

## ⚙️ ML Pipeline

```
Data Loading → Cleaning & Preprocessing → EDA & Visualization
     ↓
Feature Engineering (TF-IDF on reviews, VIF, Label Encoding)
     ↓
Model Training (Logistic Regression | Random Forest | SVM)
     ↓
Hyperparameter Tuning (GridSearchCV) + Cross-Validation
     ↓
Model Evaluation (Accuracy, Recall, F1, AUC-ROC, ELI5)
     ↓
Model Serialization → FastAPI Backend → Streamlit Frontend
```

---

## 🚀 Deployment Architecture

```
best_model_svc.joblib
        │
        ▼
  FastAPI (/predict endpoint)
        │
        ▼
  Streamlit App (Live UI)
        │
        ▼
  Streamlit Cloud (Deployed)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-learn (LR, SVM, RF), XGBoost |
| **NLP** | TF-IDF, cosine similarity |
| **Explainability** | ELI5 Permutation Importance |
| **Model Saving** | Joblib, Pickle |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Tuning** | GridSearchCV |
| **Stats** | Statsmodels (VIF) |

---

## 📦 Quickstart — Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Ajaya210/Airline-Passenger-Referral-Prediction
cd Airline-Passenger-Referral-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Backend
```bash
uvicorn app.main:app --reload
```
API docs available at: `http://localhost:8000/docs`

### 4. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
Airline-Passenger-Referral-Prediction/
│
├── Airline_Passenger_Referral_Prediction.ipynb   # Main notebook
├── best_model_svc.joblib                          # Saved SVM model
├── streamlit_app.py                               # Streamlit frontend
├── app/
│   └── main.py                                    # FastAPI backend
├── data/
│   └── data_airline_reviews.xlsx                  # Dataset
├── requirements.txt
└── README.md
```

---

## 🔍 Key Insights

- **Ground service** and **cabin service** are the two strongest predictors of whether a passenger will recommend the airline
- **SVM with RBF kernel** outperformed both Logistic Regression and Random Forest on all metrics
- **TF-IDF on customer reviews** added meaningful signal beyond structured rating features
- **Recall was prioritised** over precision to ensure genuine advocates are never missed

---

## 🧪 Conclusion

This project demonstrates a complete, production-ready ML pipeline — from raw airline reviews to a deployed prediction app. The SVM classifier achieved 95% accuracy and 0.96 AUC-ROC, offering reliable, interpretable predictions that can meaningfully support airline CRM and marketing decisions.

---

## 👤 Author

**Ajaya Kumar Pradhan**  
Data Analyst · Power BI Developer · ML Engineer  
📍 Bhubaneswar, Odisha, India

[![GitHub](https://img.shields.io/badge/GitHub-ajayaconnect-181717?style=flat-square&logo=github)](https://github.com/ajayaconnect)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/)

---

*Built as part of AlmaBetter Full Stack Data Science certification capstone project.*
