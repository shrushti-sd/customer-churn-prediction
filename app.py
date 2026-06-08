# ============================================================
# CUSTOMER CHURN PREDICTION — Step 4: Streamlit Deployment
# ============================================================
# Run: streamlit run app.py
# Make sure churn_model.pkl and scaler.pkl are in same folder
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ── UI ────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their profile.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
st.subheader("🧾 Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents?", ["Yes", "No"])
    phone_service   = st.selectbox("Phone Service?", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    tenure            = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges   = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges     = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                         float(tenure * monthly_charges), step=10.0)
    contract          = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method    = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])

st.subheader("🌐 Online Services")
col3, col4 = st.columns(2)
with col3:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
with col4:
    tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Build raw input dict
    binary_map = {'Yes': 1, 'No': 0,
                  'No phone service': 0, 'No internet service': 0, 'Male': 1, 'Female': 0}

    input_dict = {
        'gender'            : binary_map[gender],
        'SeniorCitizen'     : 1 if senior_citizen == 'Yes' else 0,
        'Partner'           : binary_map[partner],
        'Dependents'        : binary_map[dependents],
        'tenure'            : tenure,
        'PhoneService'      : binary_map[phone_service],
        'MultipleLines'     : binary_map[multiple_lines],
        'OnlineSecurity'    : binary_map[online_security],
        'OnlineBackup'      : binary_map[online_backup],
        'DeviceProtection'  : binary_map[device_protect],
        'TechSupport'       : binary_map[tech_support],
        'StreamingTV'       : binary_map[streaming_tv],
        'StreamingMovies'   : binary_map[streaming_movies],
        'PaperlessBilling'  : binary_map[paperless_billing],
        'MonthlyCharges'    : monthly_charges,
        'TotalCharges'      : total_charges,
        # Internet service dummies (DSL is base/dropped)
        'InternetService_Fiber optic' : 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No'          : 1 if internet_service == 'No' else 0,
        # Contract dummies (Month-to-month is base/dropped)
        'Contract_One year'  : 1 if contract == 'One year' else 0,
        'Contract_Two year'  : 1 if contract == 'Two year' else 0,
        # Payment method dummies (Bank transfer is base/dropped)
        'PaymentMethod_Credit card (automatic)' : 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check'        : 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check'            : 1 if payment_method == 'Mailed check' else 0,
    }

    # Build DataFrame aligned to training feature order
    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ **This customer is LIKELY TO CHURN**")
        st.metric("Churn Probability", f"{proba*100:.1f}%", delta="High Risk")
        st.markdown("""
        **💡 Retention Suggestions:**
        - Offer a discounted long-term contract upgrade
        - Provide complimentary tech support or services
        - Send a personalized retention offer via email
        - Assign a dedicated account manager
        """)
    else:
        st.success(f"✅ **This customer is LIKELY TO STAY**")
        st.metric("Churn Probability", f"{proba*100:.1f}%", delta="Low Risk")
        st.markdown("""
        **💡 Engagement Suggestions:**
        - Upsell premium services (streaming, backup)
        - Reward with loyalty points or discounts
        - Introduce referral program
        """)

    # Probability gauge
    st.progress(int(proba * 100))
    st.caption(f"Model confidence: {proba*100:.1f}% probability of churn")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit | Dataset: Telco Customer Churn (Kaggle)")
