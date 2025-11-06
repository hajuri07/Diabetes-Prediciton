# 📉 Customer Churn Prediction — Flask Deployment

> **⚠️ Important Note:**  
> If you want to run the Flask app locally, **make sure to use `clean_churn_data.csv`** (the cleaned dataset).  
> The raw dataset may contain formatting issues (like string `" "` in numeric fields), which will cause errors during preprocessing.  
> So simply place `clean_churn_data.csv` in the project directory and the app will run smoothly ✅

---


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
---

## 🗄️ Why SQL in This Project?

In real business environments, churn prediction isn't just a one-time calculation —  
**models run continuously**, and results are stored for:

- Performance tracking
- Customer outreach workflows
- Business decision dashboards

So in this project, I used **SQL** to:

1. **Store the cleaned dataset**
2. **Save model predictions with churn probability**
3. **Enable filtering & querying** (e.g., “Show customers above 75% churn risk”)

Example queries used:

```sql
-- Get top 20 highest-risk customers
SELECT customerID, Churn_Probability
FROM churn_predictions
ORDER BY Churn_Probability DESC
LIMIT 20;

-- Count churn vs non-churn distribution
SELECT Prediction, COUNT(*)
FROM churn_predictions
GROUP BY Prediction;

-- Find customers with high charges and high churn probability
SELECT customerID, MonthlyCharges, Churn_Probability
FROM churn_predictions
WHERE MonthlyCharges > 80
  AND Churn_Probability > 0.6;


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
