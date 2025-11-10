import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION AND ASSET LOADING (Using Caching) ---

# Set up page configuration first
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    # Load your saved model, preprocessor, and threshold
    with st.spinner('Loading Model Assets...'):
        preprocess = joblib.load("preprocess.joblib")
        model = joblib.load("modelxgb.joblib")
        confi = joblib.load("confi.joblib")
        
        # Load necessary data for EDA and unique values
        df_raw = pd.read_csv('Customer-Churn-data.csv')
        churn_rate_df = pd.read_csv('Churn_Rate.csv')
        churn_rate_by_co_df = pd.read_csv("Churn_Rate_by_contract_type.csv")
        
        # Load SVD data if available (required for the EDA page)
        try:
            # NOTE: Ensure you save your 'plot_df_2d' from your analysis to a CSV
            plot_df_2d = pd.read_csv("svd_2d_plot_data.csv") 
        except FileNotFoundError:
            plot_df_2d = pd.DataFrame({'PC1': [], 'PC2': [], 'Churn': []})
            
    return preprocess, model, confi, df_raw, churn_rate_df, churn_rate_by_co_df, plot_df_2d

preprocess, model, confi, df_raw, churn_rate_df, churn_rate_by_co_df, plot_df_2d = load_assets()
CHURN_THRESHOLD = confi['threshold'] # Should be 0.35

# --- 2. HELPER FUNCTIONS ---

def get_unique_values(df, col_name):
    # Get unique values, ensuring 'No internet service' etc. are present for selection
    return df[col_name].unique().tolist()

def prepare_input(input_data, df_raw):
    """Preprocesses a dictionary of user input into the format expected by the model."""
    df_input = pd.DataFrame([input_data])
    
    # 1. Apply Binary Mapping (as done in your original code)
    yes_no_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", 
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                   "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    
    for col in yes_no_cols:
        if col in df_input.columns:
            # The .astype(int) is crucial as the model expects integers
            df_input[col] = df_input[col].map(yes_no_map).fillna(0).astype(int) 

    # 2. Handle TotalCharges and SeniorCitizen
    df_input['TotalCharges'] = pd.to_numeric(df_input["TotalCharges"], errors='coerce').fillna(df_raw["TotalCharges"].median())
    df_input['SeniorCitizen'] = df_input['SeniorCitizen'].astype(int)

    # 3. Select only the columns used in the ColumnTransformer (multicat + numeric)
    multicat_cols=['Contract', 'InternetService', 'PaymentMethod']
    numeric_cols=['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # The preprocessor ONLY processes the columns it was trained on!
    X_processed = preprocess.transform(df_input[multicat_cols + numeric_cols])
    return X_processed

# --- 3. PAGE DEFINITIONS ---

def page_dashboard():
    st.title("📊 Churn Dashboard & Overview")
    st.markdown("### Key Metrics and Initial Data View")
    st.markdown("---")

    # Metrics and Overall Churn Plot
    total = churn_rate_df['n'].iloc[0]
    churners = churn_rate_df['churners'].iloc[0]
    churn_percent = (churners / total) * 100
    non_churners = total - churners
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Overall Churn Rate", value=f"{churn_percent:.2f}%")
    
    with col2:
        # F1-Score for the Churn class (1) @ Thr 0.35
        st.metric(label="Model F1-Score (Churn)", value=f"0.660", help="XGBoost model F1-Score for the Churn class.")

    with col3:
        plot_churn_df = pd.DataFrame({'Status':['Not Churned','Churned'], 'Count':[non_churners, churners]})
        fig_pie = px.pie(plot_churn_df, values='Count', names='Status', 
                         title='Churners vs Non-Churners', hole=.3,
                         color_discrete_sequence=['#4c78a8', '#e45756'])
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # EDA Plots (Your Plots 2 & 3)
    st.subheader("Data Views")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("##### Churn Rate by Contract Type")
        fig_contract = px.bar(churn_rate_by_co_df, x='contract', y='churn_rate', 
                              labels={'churn_rate': 'Churn Rate (%)', 'contract': 'Contract Type'},
                              color='contract',
                              color_discrete_sequence=['#4c78a8', '#f58518', '#e45756'])
        st.plotly_chart(fig_contract, use_container_width=True)

    with colB:
        st.markdown("##### Average Monthly Charges: Churn vs Non-Churn")
        avg_charges = df_raw.groupby('Churn')['MonthlyCharges'].mean().reset_index()
        avg_charges['Churn_Status'] = avg_charges['Churn'].map({0: 'Non-Churn', 1: 'Churn'})
        fig_charges = px.bar(avg_charges, x='Churn_Status', y='MonthlyCharges',
                             color='Churn_Status',
                             color_discrete_sequence=['#54a24b', '#7293cb'])
        fig_charges.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        st.plotly_chart(fig_charges, use_container_width=True)


def page_predict():
    st.title("🔮 Real-Time Churn Prediction")
    st.markdown("### Input Customer Data to Get a Prediction")
    st.markdown("---")
    
    # --- Input Form ---
    with st.form("churn_prediction_form"):
        st.subheader("Customer Details")
        
        # Layout inputs logically across columns
        col1, col2, col3, col4 = st.columns(4)
        gender = col1.selectbox("Gender", get_unique_values(df_raw, 'gender'))
        # SeniorCitizen is 0/1 in your data after transformation
        senior_citizen = col2.selectbox("Senior Citizen (0/1)", [0, 1]) 
        partner = col3.selectbox("Partner", get_unique_values(df_raw, 'Partner'))
        dependents = col4.selectbox("Dependents", get_unique_values(df_raw, 'Dependents'))

        col5, col6, col7 = st.columns(3)
        tenure = col5.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        MonthlyCharges = col6.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=50.0, step=0.1)
        TotalCharges = col7.number_input("Total Charges ($)", min_value=0.0, value=600.0)

        st.subheader("Service Details")
        colA, colB, colC, colD, colE = st.columns(5)
        phone_service = colA.selectbox("Phone Service", get_unique_values(df_raw, 'PhoneService'))
        multiple_lines = colB.selectbox("Multiple Lines", get_unique_values(df_raw, 'MultipleLines'))
        internet_service = colC.selectbox("Internet Service", get_unique_values(df_raw, 'InternetService'))
        online_security = colD.selectbox("Online Security", get_unique_values(df_raw, 'OnlineSecurity'))
        online_backup = colE.selectbox("Online Backup", get_unique_values(df_raw, 'OnlineBackup'))

        colF, colG, colH, colI, colJ = st.columns(5)
        device_protection = colF.selectbox("Device Protection", get_unique_values(df_raw, 'DeviceProtection'))
        tech_support = colG.selectbox("Tech Support", get_unique_values(df_raw, 'TechSupport'))
        streaming_tv = colH.selectbox("Streaming TV", get_unique_values(df_raw, 'StreamingTV'))
        streaming_movies = colI.selectbox("Streaming Movies", get_unique_values(df_raw, 'StreamingMovies'))

        st.subheader("Contract and Billing")
        colK, colL, colM = st.columns(3)
        contract = colK.selectbox("Contract Type", get_unique_values(df_raw, 'Contract'))
        payment_method = colL.selectbox("Payment Method", get_unique_values(df_raw, 'PaymentMethod'))
        paperless_billing = colM.selectbox("Paperless Billing", get_unique_values(df_raw, 'PaperlessBilling'))

        submitted = st.form_submit_button("Predict Churn Status")
    
    if submitted:
        # Consolidate all inputs into a dictionary
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service, 
            'MultipleLines': multiple_lines, 'InternetService': internet_service, 
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 
            'Contract': contract, 'PaperlessBilling': paperless_billing, 
            'PaymentMethod': payment_method, 'MonthlyCharges': MonthlyCharges, 
            'TotalCharges': TotalCharges,
            # Note: The raw data contains all columns, we only select the processed ones later.
        }
        
        with st.spinner('Calculating Prediction...'):
            # Prepare the input using the robust helper function
            X_processed = prepare_input(input_data, df_raw)
            
            # Predict probability
            y_proba = model.predict_proba(X_processed)[:, 1][0]
            prediction = 1 if y_proba >= CHURN_THRESHOLD else 0
        
        st.markdown("---")
        st.subheader("Prediction Outcome")
        
        prob_percent = y_proba * 100
        
        # Display results clearly
        if prediction == 1:
            st.error(f"**HIGH CHURN RISK**")
            st.markdown(f"**Probability of Churn:** **`{prob_percent:.2f}%`** (above threshold of {CHURN_THRESHOLD})")
            st.info("💡 **Action:** Immediate, high-value retention offer is recommended.")
        else:
            st.success(f"**LOW CHURN RISK**")
            st.markdown(f"**Probability of Churn:** **`{prob_percent:.2f}%`** (below threshold of {CHURN_THRESHOLD})")
            st.info("💡 **Action:** Monitor engagement; consider a small, targeted upgrade offer.")

        st.markdown("##### Model Confidence")
        st.progress(y_proba)


def page_insights():
    st.title("🧠 Model Insights and Feature Importance")
    st.markdown("### Understanding What Drives the Prediction")
    st.markdown("---")

    # --- Feature Importance Plot ---
    st.subheader("Top 10 Feature Importance (XGBoost)")
    
    # Reconstruct feature importance data for plotting (as per your code)
    multicat_cols=['Contract', 'InternetService', 'PaymentMethod']
    numeric_cols=['tenure', 'MonthlyCharges', 'TotalCharges']
    ohe_features = preprocess.named_transformers_['one_hot'].get_feature_names_out(multicat_cols)
    all_features = np.r_[ohe_features, numeric_cols]
    importances = model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig_feat = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                      title="Top 10 Features Impacting Churn Prediction",
                      color_discrete_sequence=px.colors.qualitative.D3)
    fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("---")
    
    # --- Model Performance Metrics ---
    st.subheader("Model Performance (Threshold: 0.35)")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("##### Classification Report (Focus: Churn=1)")
        # Display the classification report clearly
        st.code("""
                  precision    recall  f1-score   support

           0        0.893     0.923     0.907      1869
           1        0.697     0.627     0.660       645

    accuracy                            0.844      2514
        """)
    
    with colB:
        st.markdown("##### Confusion Matrix Heatmap")
        # Display the confusion matrix (mocked based on your report)
        cm_mock = np.array([[1869 * 0.923, 1869 * 0.077], [645 * 0.373, 645 * 0.627]]).astype(int)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_mock, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Predicted Non-Churn', 'Predicted Churn'],
                    yticklabels=['Actual Non-Churn', 'Actual Churn'],
                    cbar=False, linewidths=.5, linecolor='black')
        ax_cm.set_title("XGBoost Confusion Matrix")
        st.pyplot(fig_cm)

# --- 4. MAIN APP LOGIC ---

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Churn", "Model Insights"])

# Page routing
if page == "Dashboard":
    page_dashboard()
elif page == "Predict Churn":
    page_predict()
elif page == "Model Insights":
    page_insights()
