# 📉 Customer Churn Prediction — Flask Deployment

A **production-style Machine Learning application** that predicts customer churn and helps businesses take **proactive retention decisions**.

This project goes beyond a Jupyter notebook — it includes:

- ✅ Data Cleaning & Preprocessing Pipeline
- ✅ Model Training & Feature Interpretation
- ✅ **XGBoost vs Logistic Regression** comparison (with business reasoning)
- ✅ Flask Web App for real-time predictions
- ✅ Deployment-ready project structure

---

## 🚀 Project Overview

Customer churn is a key revenue leak for subscription-based businesses.  
This system predicts **whether a customer is likely to churn**, and **why**, based on behavioral & service usage patterns.

### ✨ Why XGBoost?
We compared **Logistic Regression** vs **XGBoost**:

| Model | Strengths | Weaknesses |
|------|-----------|------------|
| Logistic Regression | Higher recall (catches more churners) | Too many false alarms → higher retention cost |
| **XGBoost** ✅ | Better balance: strong recall **+** fewer false positives | Slightly more complex |

**Churn is not linear.**  
It depends on *combinations* of behaviors, like:


XGBoost **learns these interactions**—Logistic Regression does not.

---

## 🧠 Tech Stack

| Layer | Tools Used |
|------|------------|
| Language | Python |
| ML Model | XGBoost |
| Data Pipeline | Scikit-Learn ColumnTransformer |
| Web Framework | Flask |
| Visualization | Matplotlib / Plotly |
| Deployment | Local / Cloud-ready |

---

## 🗂️ Project Structure


---

## 🎮 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/churn-prediction-app.git
cd churn-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
💡 Key Insights Learned

Model performance ≠ Real-world performance — false positives have cost.

Business problems are solved with interpretability, not just accuracy.

Deployment is what separates ML practitioners from ML engineers.

✍️ Author

Ibrahim Hajuri
Machine Learning Engineer | Builder | Curious Mind
Made with ❤️ and way too much coffee.
