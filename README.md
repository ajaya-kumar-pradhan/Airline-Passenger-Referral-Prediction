# ✈️ Airline Passenger Referral Prediction

> **Classification · Random Forest · Streamlit · Hugging Face Deployment**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face-orange?style=for-the-badge)](https://huggingface.co/spaces/ajayapradhanconnect/Airline-Passenger-Referral-Prediction)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 🚀 Live Demo
**Click the link below to try the passenger sentiment predictor instantly (Free & Always On):**

### 👉 [**Launch Airline Referral Predictor**](https://huggingface.co/spaces/ajayapradhanconnect/Airline-Passenger-Referral-Prediction)

---

## 📌 Project Overview
In the highly competitive aviation industry, passenger referrals are the ultimate growth engine. This project uses machine learning to predict whether a passenger will become a **Promoter** (refer the airline to others) or a **Detractor** based on their flight experience.

### 🌟 Key Features
- **Modern Flight Dashboard**: Elegant UI with real-time prediction results.
- **Multi-Factor Input**: Analyzes Seat Comfort, Crew Service, Food, Entertainment, and more.
- **Instant Sentiment Analysis**: Provides an "Advocacy Score" or "Churn Probability" based on model confidence.
- **Responsive Design**: Works perfectly on mobile and desktop.

---

## 🤖 The Model (Random Forest Classifier)
The core engine is a **Random Forest Classifier** trained on a massive dataset of airline reviews. It evaluates the relative importance of different service factors to determine the passenger's likelihood of referral.

- **Accuracy**: Optimized for high precision and recall.
- **Preprocessors**: Uses custom Scalers and Label Encoders for robust data handling.
- **Portability**: Serialized using `joblib` for rapid deployment.

---

## 🛠️ Tech Stack
- **Backend**: Python, Scikit-learn, Pandas, NumPy, Joblib.
- **Frontend**: Streamlit (with Custom Inter UI & CSS).
- **Deployment**: Hugging Face Spaces.
- **Data Engine**: Openpyxl (for dataset processing).

---

## 📁 Project Structure
```text
Airline-Passenger-Referral-Prediction/
├── app.py                      # Main Passenger Referral Predictor
├── requirements.txt            # Production Dependencies
├── airline_svc_model.joblib   # Trained Classifier Model (96MB)
├── airline_encoder.joblib     # Airline Name Encoder
├── features_list.joblib       # Feature Mapping
├── scaler.joblib              # Feature Scaler
└── README.md                  # You are here!
```

---

## 📦 Run Locally
1. **Clone the repo**
   ```bash
   git clone https://github.com/ajaya-kumar-pradhan/Airline-Passenger-Referral-Prediction.git
   cd Airline-Passenger-Referral-Prediction
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the app**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author
**Ajaya Kumar Pradhan**  
Data Analyst · Power BI Developer · ML Engineer  
📍 Bhubaneswar, Odisha, India

[![GitHub](https://img.shields.io/badge/GitHub-ajayaconnect-181717?style=flat-square&logo=github)](https://github.com/ajaya-kumar-pradhan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/)

---
*Built as part of Data Science Professional Certification.*
