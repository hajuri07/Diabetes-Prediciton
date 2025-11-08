import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model, Preprocessor, and Config
@st.cache_resource
def load_assets():
    # Load your saved assets
    preprocess = joblib.load("preprocess.joblib")
    model = joblib.load("modelxgb.joblib")
    confi = joblib.load("confi.joblib")
    # Load your EDA data (you need to save these dataframes first)
    churn_rate_df = pd.read_csv('Churn_Rate.csv')
    churn_rate_by_co_df = pd.read_csv("Churn_Rate_by_contract_type.csv")
    # A dummy clean df for getting unique categorical values
    clean_df = pd.read_csv("clean_churn_data.csv") 
    return preprocess, model, confi, churn_rate_df, churn_rate_by_co_df, clean_df

preprocess, model, confi, churn_rate_df, churn_rate_by_co_df, clean_df = load_assets()
CHURN_THRESHOLD = confi['threshold']

# --- Page Setup ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Churn", "Model Insights", "Exploratory Data Analysis"])

if page == "Dashboard":
    # --- Dashboard Page Content ---
    st.title("📉 Customer Churn Prediction Dashboard")
    st.markdown("### Model-Driven Insights for Retention")
    
    # 1. Overall Churn Rate Metric & Plot
    total = churn_rate_df['n'].iloc[0]
    churners = churn_rate_df['churners'].iloc[0]
    churn_percent = (churners / total) * 100
    non_churners = total - churners
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Overall Churn Rate", value=f"{churn_percent:.2f}%", delta_color="inverse")
        if churn_percent > 30:
            st.warning("High Churn Alert! Immediate action is recommended.")
        else:
            st.success("Churn Rate is within acceptable limits.")

    with col2:
        plot_churn_df = pd.DataFrame({'Status':['Not Churned','Churned'], 'Count':[non_churners, churners]})
        fig_pie = px.pie(plot_churn_df, values='Count', names='Status', 
                         title='Churners vs Non-Churners', hole=.3,
                         color_discrete_sequence=['#1f77b4', '#d62728'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Optional: Initial Data Loading Animation
    with st.spinner('Preparing visual components...'):
        st.success('Dashboard Ready!')
        
elif page == "Predict Churn":
    # --- Prediction Page Content ---
    st.title("🎯 Real-Time Churn Prediction")
    st.markdown("Use the controls below to input customer data and predict their churn status.")

    # Input Form (Using st.columns for better organization)
    with st.form("churn_prediction_form"):
        st.subheader("Customer Profile")
        colA, colB, colC = st.columns(3)
        
        # Numeric Inputs
        tenure = colA.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        MonthlyCharges = colB.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=50.0)
        # TotalCharges requires more complex handling, simplifying for mockup
        TotalCharges = colC.number_input("Total Charges ($)", min_value=0.0, value=600.0)

        st.subheader("Service and Contract Details")
        colD, colE, colF = st.columns(3)
        # Categorical Inputs (using unique values from your data)
        contract = colD.selectbox("Contract Type", clean_df['Contract'].unique())
        internet_service = colE.selectbox("Internet Service", clean_df['InternetService'].unique())
        payment_method = colF.selectbox("Payment Method", clean_df['PaymentMethod'].unique())

        # Yes/No Binary (simplified)
        st.markdown("##### Other Services")
        colG, colH, colI = st.columns(3)
        partner = colG.selectbox("Partner", ["No", "Yes"])
        dependents = colH.selectbox("Dependents", ["No", "Yes"])
        paperless_billing = colI.selectbox("Paperless Billing", ["No", "Yes"])
        
        # The code needs all features, but we will use the processed ones only in the final prediction

        submitted = st.form_submit_button("Get Prediction")
    
    if submitted:
        # Create a DataFrame for prediction based on *all* required columns
        # NOTE: This mockup skips the full feature set for brevity. 
        # You need to reconstruct a full 1D dataframe with all features (Gender, Seniors, etc.)
        # and map 'Yes'/'No'/'No internet service' to their integer/binary values BEFORE 
        # feeding into the preprocessor.
        
        # Example of how to structure the data for the preprocessor:
        # NOTE: You must ensure the order and column names match your x_train exactly.
        
        raw_data = {
            'Contract': contract, 'InternetService': internet_service, 
            'PaymentMethod': payment_method, 'tenure': tenure, 
            'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
            # Add all other features here...
        }
        
        # ***CRITICAL: Mocking a complete input dataframe for preprocessor***
        # In a real app, you must create a df with all 20 columns from your code's 'x'
        # and then map the 'Yes'/'No' columns to 1/0 as you did in your code.
        
        # Creating a simplified, incomplete input for this mockup:
        # In your final app, use ALL features and apply the binary mapping first.
        
        # --- Simplified Prediction Process for Mockup ---
        input_df = pd.DataFrame([{
            'Contract': contract, 'InternetService': internet_service, 'PaymentMethod': payment_method, 
            'tenure': tenure, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges,
            # Add placeholders for all other features here (e.g., 'Partner': 'No', etc.)
        }])
        
        # Apply the binary encoding map to the input_df *before* using the preprocessor
        # e.g., input_df["Partner"] = input_df["Partner"].map({"Yes": 1, "No": 0}) 
        
        # Only select the columns used in your ColumnTransformer
        cols_for_preprocess = ['Contract', 'InternetService', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Since the preprocessor was trained only on a subset of features, we must simplify.
        # In a real deployment, load the full x_train column names and ensure the input_df matches.
        
        with st.spinner('Running XGBoost Model...'):
            # This line will only work if input_df has ALL the categorical and numerical columns 
            # and the binary columns are already converted to 0/1.
            # input_processed = preprocess.transform(input_df[cols_for_preprocess]) 
            
            # MOCKING the prediction for the sake of the blueprint:
            # You will replace this with the actual prediction call:
            # y_proba = model.predict_proba(input_processed)[:, 1][0] 
            y_proba = 0.65  # MOCK PROBABILITY
            
            prediction = 1 if y_proba >= CHURN_THRESHOLD else 0
        
        st.subheader("Prediction Result")
        
        # --- Animated Output ---
        if prediction == 1:
            st.error(f"⚠️ **HIGH CHURN RISK**")
            st.markdown(f"The model predicts this customer is **likely to Churn** with **{y_proba:.2f} probability**.")
            st.info("💡 **Retention Strategy:** Recommend a personalized, high-value two-year contract offer or a loyalty discount program.")
        else:
            st.success(f"✅ **LOW CHURN RISK**")
            st.markdown(f"The model predicts this customer is **unlikely to Churn** with **{y_proba:.2f} probability**.")
            st.balloons() # Small animation for good news!
            st.info("💡 **Retention Strategy:** Continue with existing service, consider a small upgrade offer to increase satisfaction.")

elif page == "Model Insights":
    # --- Model Insights Page Content ---
    st.title("🧠 XGBoost Model Insights")

    # 1. Feature Importance Plot
    st.subheader("Feature Importance (XGBoost)")
    
    # Reconstruct feature importance data for plotting
    ohe_features = preprocess.named_transformers_['one_hot'].get_feature_names_out(['Contract', 'InternetService', 'PaymentMethod'])
    all_features = np.r_[ohe_features, ['tenure', 'MonthlyCharges', 'TotalCharges']]
    importances = model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig_feat = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                      title="Top 10 Most Important Features",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Confusion Matrix Heatmap
    st.subheader("Confusion Matrix and Classification Report (XGBoost @ Thr 0.35)")
    # MOCKING the data for visualization, you should pre-calculate the final metrics
    # and load them, as the preprocessor/model are not fully loaded with the full X_test here.
    cm_mock = np.array([[1200, 100], [150, 450]])
    
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_mock, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Predicted Non-Churn (0)', 'Predicted Churn (1)'],
                yticklabels=['Actual Non-Churn (0)', 'Actual Churn (1)'])
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_ylabel("Actual Label")
    ax_cm.set_xlabel("Predicted Label")
    st.pyplot(fig_cm)
    
    # 3. Classification Report
    st.code("""
    # Classification Report (XGBoost, Threshold: 0.35)
    # This report favors higher Recall (finding actual churners)
    
                  precision    recall  f1-score   support

           0        0.893     0.923     0.907      1869
           1        0.697     0.627     0.660       645

    accuracy                            0.844      2514
   macro avg        0.795     0.775     0.783      2514
weighted avg        0.843     0.844     0.843      2514
    """)
    
elif page == "Exploratory Data Analysis":
    # --- EDA Page Content (Using Plotly for interactivity) ---
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visualizing key drivers of customer churn directly from the data.")
    
    # 1. Churn by Contract Type (Your Plot 2)
    st.subheader("Churn Rate by Contract Type")
    fig_contract = px.bar(churn_rate_by_co_df, x='contract', y='churn_rate', 
                          title='Churn Rate (%) by Contract Type',
                          labels={'churn_rate': 'Churn Rate (%)', 'contract': 'Contract Type'},
                          color='contract',
                          color_discrete_sequence=['#4c78a8', '#f58518', '#e45756'])
    st.plotly_chart(fig_contract, use_container_width=True)

    # 2. Average Monthly Charges (Your Plot 3)
    avg_charges = clean_df.groupby('Churn')['MonthlyCharges'].mean().reset_index()
    avg_charges['Churn_Status'] = avg_charges['Churn'].map({0: 'Non-Churn', 1: 'Churn'})
    
    st.subheader("Average Monthly Charges: Churn vs Non-Churn")
    fig_charges = px.bar(avg_charges, x='Churn_Status', y='MonthlyCharges',
                         title='Average Monthly Charges by Churn Status',
                         color='Churn_Status',
                         color_discrete_sequence=['#54a24b', '#7293cb'])
    fig_charges.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig_charges.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_charges, use_container_width=True)
    
    # 3. Interactive SVD/PCA Plot (Your fig_2d)
    # NOTE: You need to save the SVD results (Xtr_2D and y_train) to a CSV/joblib 
    # and load it here to make this work, as the training is done outside the app.
    st.subheader("Customer Data Projection (SVD)")
    # MOCKING data loading for SVD plot
    # plot_df_2d = pd.read_csv("svd_2d_plot_data.csv") 
    
    # Since you showed the 2D plot, we'll plot that interactively.
    # Replace the following with your loaded 2D SVD DataFrame:
    st.info("To make this chart interactive, save your `plot_df_2d` to a file (e.g., CSV) and load it here.")
    # fig_2d_plot = px.scatter(
    #     plot_df_2d, x="PC1", y="PC2",
    #     color="Churn",
    #     opacity=0.6,
    #     title="Customer Churn — 2D SVD Visualization"
    # )
    # st.plotly_chart(fig_2d_plot, use_container_width=True)
