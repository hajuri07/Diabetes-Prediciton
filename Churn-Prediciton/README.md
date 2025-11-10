<h1 align="center">Customer-Churn  prediciton 🚀</h1>

<p align="center">
  <em>This project predicts whether a customer will churn (leave the service).  
Businesses can use these predictions to take retention actions before the customer leaves.</em>
</p>

---
**Demo:** **  


---

## 🔍 Project Summary

This is a complete **end-to-end Machine Learning pipeline** that includes:

| Step | Description |
|------|-------------|
| **Data Cleaning** | Removed missing values, handled duplicates, standardized column formats |
| **EDA** | Visualized churn patterns & correlations to understand feature impact |
| **Feature Engineering** | Label Encoding, One-Hot Encoding, Scaling & Feature Selection |
| **Model Selection** | Compared Logistic Regression, Random Forest, and XGBoost |
| **Final Model** | **XGBoost** selected due to stronger recall on imbalanced data |
| **Evaluation** | Used **Precision, Recall & Confusion Matrix** to measure real performance |
| **Deployment** | Built an interactive **Streamlit** app for real-time predictions |

---

## ⚠️ **Important Note Before Running**

This project uses a **cleaned dataset** to avoid encoding and scaling mismatches.

Use: clean_churn.csv
Do NOT use the original raw dataset.



Using the raw dataset will cause:
- Shape mismatch errors
- Unknown-category errors
- Wrong predictions

So always work with:
clean_churn.csv ✅


---

## 🧠 Why XGBoost Was Selected

The dataset is **imbalanced** → fewer churn cases than non-churn.

- Logistic Regression performed poorly due to linear boundaries.
- Random Forest gave good accuracy but **low recall**.
- **XGBoost balanced data better and achieved higher recall**, which is critical because:

Missing a churn customer is worse than giving a false alarm.


So **we optimized for Recall**, not Accuracy.

---

## 🧪 Model Evaluation

| Metric | Meaning |
|-------|--------|---------|
| **Precision** | Of predicted churners, how many were real churners |
| **Recall** |  How many actual churners we successfully detected |
| **Confusion Matrix** | Used to verify true/false positives/negatives |

**Recall was prioritized** because the business goal = **catch as many leaving customers as possible.**

---

## 🛠️ Tech Stack

Python • Pandas • NumPy • Scikit-Learn • XGBoost • Streamlit • Matplotlib / Seaborn


---

## 📦 Dataset

Contains customer behavioral and service-related features.  
Target column:
Churn (1 = Leaving, 0 = Staying)
---
**Made with ❤️**
