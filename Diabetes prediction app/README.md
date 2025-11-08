<h1 align="center">🤖 Diabetes Prediction using Machine Learning</h1>

<p align="center">
  <em>A Binary Classification Project built on the <b>Pima Indians Diabetes Dataset</b></em><br>
  <strong>Predict diabetes risk instantly using Machine Learning ⚡</strong>
</p>

---

## 🌟 Overview

This project is a **binary classification** problem based on the popular *Pima Indians Diabetes Dataset*.  
It uses **Supervised Machine Learning algorithms** — *Logistic Regression, Random Forest, and XGBoost* —  
to predict whether a person is diabetic or not.

After evaluating all models, the **best-performing model** was deployed as an  
**interactive Streamlit web app** 🧠 allowing users to input health data and get predictions instantly.

---

## 🔥 Features

- ✅ End-to-end ML workflow (EDA → Model Training → Evaluation → Deployment)  
- ✅ Multiple ML models trained and compared (*Logistic Regression, Random Forest, XGBoost*)  
- ✅ Accuracy, Precision, Recall, F1-Score comparison table  
- ✅ Feature scaling using **StandardScaler**  
- ✅ Model bundling using **joblib** for reusability  
- ✅ Interactive **Streamlit** web app with a clean UI  
- ✅ Instant diabetes risk prediction based on user input  

---

## 🧩 Tech Stack

| **Category** | **Tools Used** |
|---------------|----------------|
| 🐍 Language | Python |
| 🤖 ML Libraries | scikit-learn, XGBoost, NumPy, Pandas |
| 📊 Visualization | Matplotlib, Seaborn |
| 💾 Model Saving | joblib |
| 🌐 Web App Framework | Streamlit |
| 🧠 Environment | Kaggle Notebook + VS Code (Anaconda) |

---

## 📊 Model Development

<details>
<summary><b>1️⃣ Data Understanding & Cleaning</b></summary>

- Used Pima Indians Diabetes Dataset from Kaggle  
- Checked null values, outliers, and feature distributions  
- Scaled features using `StandardScaler`
</details>

<details>
<summary><b>2️⃣ Exploratory Data Analysis (EDA)</b></summary>

- Pairplots to visualize feature relationships  
- Observed that higher glucose and BMI correlated strongly with diabetes
</details>

<details>
<summary><b>3️⃣ Model Training</b></summary>

- Trained and compared three algorithms:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
</details>

---


## 💾 Model Saving

```python
obj = {"scaler": scaler, "model": model}
joblib.dump(obj, "scaler_model_bundle.joblib")
