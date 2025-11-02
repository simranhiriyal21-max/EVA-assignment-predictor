# 🎯 EVA Assignment Predictor  
_A LightGBM / XGBoost–based AI Ticket Classifier with Streamlit Deployment and ServiceNow Integration_

---

## 📘 Project Overview

This project implements an **AI-based Automatic IT Ticket Assignment System** using **Natural Language Processing (NLP)** and **Machine Learning (LightGBM, XGBoost)**.

It predicts the **assigned group / category** for a ticket based on its textual description.

The app includes:
- Data preprocessing and training in **Google Colab**
- Model evaluation with **ROC–AUC curves**
- Deployment using **Streamlit Cloud**
- Integration with **ServiceNow PDI** through a REST API call

---

## 🧠 Architecture Overview


---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python 3 |
| Model Algorithms | LightGBM, XGBoost |
| Feature Engineering | TF-IDF Vectorizer |
| App Framework | Streamlit |
| Deployment | Streamlit Community Cloud |
| Data Source | Synthetic Ticket Dataset (`tickets_synthetic.csv`) |
| Integration | REST API from ServiceNow PDI |

---

## 🧾 Folder Structure


---

## 🚀 How to Run the Project

### **1️⃣ Train & Save Models in Google Colab**

1. Open your Colab notebook.  
2. Load `tickets_synthetic.csv` and preprocess using TF-IDF or embeddings.  
3. Train **LightGBM** and/or **XGBoost** models.  
4. Plot and save ROC/AUC charts.  
5. Save artifacts:
   ```python
   joblib.dump(model_lgb, 'model_lgb.joblib')
   joblib.dump(tfv, 'tfidf_vectorizer.joblib')
   joblib.dump(le, 'label_encoder.joblib')
