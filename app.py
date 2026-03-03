import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Loan Prediction System", page_icon="🏦", layout="centered")

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# -----------------------------------------------------------------------------
# 2. NAVIGATION
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
with c2:
    if st.button("Home"): st.session_state['page'] = 'Home'
with c3:
    if st.button("Predict"): st.session_state['page'] = 'Predict'
with c4:
    if st.button("About"): st.session_state['page'] = 'About'
st.markdown("---")

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_emi(principal, rate, tenure_years):
    r = rate / (12 * 100) # Monthly interest rate
    n = tenure_years * 12 # Total months
    if n == 0: return 0
    emi = (principal * r * pow(1 + r, n)) / (pow(1 + r, n) - 1)
    return emi

# -----------------------------------------------------------------------------
# 4. DATA & MODEL TRAINING
# -----------------------------------------------------------------------------
DATA_PATH = 'train.csv'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def train_model(df):
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    # Feature Engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
    df['Balance_Income'] = df['Total_Income'] - df['EMI']
    
    le_target = LabelEncoder()
    df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    
    encoders = {}
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Save column order
    feature_columns = X.columns.tolist()
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    return rf, encoders, le_target, feature_columns

df = load_data()
if df is not None:
    model, encoders, target_encoder, feature_columns = train_model(df)

# -----------------------------------------------------------------------------
# 5. PAGE LOGIC
# -----------------------------------------------------------------------------

if st.session_state['page'] == "Home":
    # --- CLEAN HOME PAGE ---
    st.title("Loan Approval Prediction System")
    st.subheader("Using Machine Learning")
    
    st.markdown("---")

    st.markdown("### About This Application")
    st.info("""
    Welcome to the next generation of banking technology. This application utilizes advanced 
    **Machine Learning** algorithms to automate the loan eligibility assessment process. 
    By analyzing key financial indicators—such as **Income, Credit History, and Loan Term**—our 
    system provides an instant, objective, and data-driven prediction.
    """)

    st.write("") # Spacer

    st.markdown("### Why Use This System?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("⚡ **Real-Time Analysis**\n\nGet an 'Approved' or 'Rejected' status in milliseconds without waiting for manual verification.")
        st.success("🛡️ **Financial Guardrails**\n\nAutomatically detects and filters out mathematically insolvent applications (e.g., high debt ratio).")

    with col2:
        st.success("🎯 **High Accuracy**\n\nBuilt on the robust **Random Forest Classifier**, ensuring reliable and objective risk assessment.")
        st.success("💡 **Advisory Intelligence**\n\nWe don't just say 'No.' If your application is rejected, our system provides actionable tips to improve.")

    st.markdown("---")
    
    st.markdown("### How It Works")
    st.markdown("""
    1. Click on **Predict** in the navigation menu above.
    2. Enter your details (Income, Education, Loan Amount, etc.).
    3. Click the **"Predict Status"** button.
    4. Receive your instant eligibility report with a confidence score.
    """)


elif st.session_state['page'] == "Predict":
    st.title("📋 Loan Application Form")
    
    if df is None:
        st.error("Error: 'train.csv' not found.")
    else:
        with st.form("prediction_form"):
            st.subheader("Applicant Details")
            c1, c2 = st.columns(2)
            
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                
            with c2:
                applicant_income = st.number_input("Applicant Income (Monthly ₹)", value=5000, min_value=0)
                coapplicant_income = st.number_input("Co-Applicant Income (Monthly ₹)", value=0, min_value=0)
                loan_amount = st.number_input("Loan Amount (Total ₹)", value=100000, min_value=1000)
                loan_term_years = st.number_input("Loan Term (Years)", value=15, min_value=1)
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                
                cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Predict Status")

        if submit_btn:
            # --- 1. DATA PREPARATION ---
            loan_amt_k = loan_amount / 1000
            loan_term_m = loan_term_years * 12
            total_income = applicant_income + coapplicant_income
            
            real_emi = calculate_emi(loan_amount, 10.0, loan_term_years)
            model_emi = (loan_amt_k * 1000) / loan_term_m
            balance_income = total_income - model_emi
            
            credit_history = 1.0 if cibil_score >= 600 else 0.0

            # --- 2. PRE-PREDICTION VALIDATION (Constraints) ---
            
            rejection_reasons = []
            tips = []
            
            # Constraint 1: Minimum Income
            if total_income < 10000:
                rejection_reasons.append(f"**Income Threshold:** Household income (₹{total_income}) is below the minimum dataset requirement of ₹10,000.")
                tips.append("💡 **Tip:** Add a Co-Applicant with a stable income.")

            # Constraint 2: High Leverage Ratio
            elif real_emi > (total_income * 0.75):
                rejection_reasons.append(f"**High Risk:** EMI (₹{real_emi:,.0f}) exceeds 75% of income, classified as unsafe leverage.")
                tips.append("💡 **Tip:** Increase the loan tenure to reduce the monthly EMI.")

            # Constraint 3: Insolvency
            elif real_emi > total_income:
                rejection_reasons.append(f"**Critical Insolvency:** EMI (₹{real_emi:,.0f}) exceeds Total Income.")
                tips.append("💡 **Tip:** Reduce the Loan Amount significantly.")
            
            # Constraint 4: Excessive Multiplier
            elif loan_amount > (total_income * 150):
                rejection_reasons.append(f"**Outlier Detected:** Loan is {loan_amount/total_income:.0f}x income (Model Limit: 150x).")
                tips.append("💡 **Tip:** Reduce the Loan Amount.")

            # Constraint 5: Subsistence Check
            elif (total_income - real_emi) < 3000:
                 rejection_reasons.append(f"**Affordability:** Remaining balance after EMI is below subsistence level (₹3,000).")
                 tips.append("💡 **Tip:** Add a Co-Applicant.")
            
            # Constraint 6: CIBIL Cutoff
            elif cibil_score < 600:
                 rejection_reasons.append(f"**Credit Score:** CIBIL {cibil_score} is below the qualifying threshold.")
                 tips.append("💡 **Tip:** Clear debts to improve score.")

            
            # --- 3. MODEL PREDICTION ---
            if rejection_reasons:
                st.error("❌ **Rejected (Pre-Screening)**")
                for reason in rejection_reasons:
                    st.warning(f"⚠️ {reason}")
                
                if tips:
                    st.markdown("#### 💡 Suggestions:")
                    for tip in tips:
                        st.info(tip)
            
            else:
                # Valid Data -> Send to ML Model
                se_map = "Yes" if "Yes" in self_employed else "No"
                
                input_df = pd.DataFrame({
                    'Gender': [gender], 
                    'Married': [married], 
                    'Dependents': [dependents],
                    'Education': [education], 
                    'Self_Employed': [se_map],
                    'ApplicantIncome': [applicant_income], 
                    'CoapplicantIncome': [coapplicant_income],
                    'LoanAmount': [loan_amt_k], 
                    'Loan_Amount_Term': [loan_term_m],
                    'Credit_History': [credit_history],
                    'Property_Area': [property_area],
                    'Total_Income': [total_income],
                    'EMI': [model_emi],
                    'Balance_Income': [balance_income]
                })
                
                input_df = input_df[feature_columns]
                
                try:
                    for col, le in encoders.items():
                        input_df[col] = le.transform(input_df[col].astype(str))
                    
                    pred = model.predict(input_df)
                    prob = model.predict_proba(input_df)
                    confidence = np.max(prob) * 100
                    result = target_encoder.inverse_transform(pred)[0]
                    
                    st.markdown("### Results")
                    st.info(f"**Financial Analysis:** EMI ₹{real_emi:,.0f} | Income ₹{total_income:,.0f}")
                    
                    if result == 'Y':
                        st.success(f"✅ **Approved** (Model Confidence: {confidence:.2f}%)")
                    else:
                        st.error(f"❌ **Rejected** (Model Confidence: {confidence:.2f}%)")
                        st.write("Reason: The Machine Learning model identified high-risk patterns in the applicant profile.")
                        
                        st.markdown("#### 💡 Improvement Tips:")
                        ai_tips = []
                        if total_income < 40000:
                            ai_tips.append("• **Income:** Your income is on the lower side for this loan amount.")
                        if loan_term_years < 10:
                            ai_tips.append("• **Tenure:** Try increasing the loan term.")
                        if not ai_tips:
                            ai_tips.append("• **Co-Applicant:** Adding a co-applicant can improve approval chances.")
                            
                        for t in ai_tips:
                            st.write(t)

                except Exception as e:
                    st.error(f"System Error: {e}")

elif st.session_state['page'] == "About":
    st.title("About the Project")

    # 1. THE PROBLEM (Native clean red alert)
    st.error("""
    **The Problem: Manual Underwriting**
    
    Traditionally, banks relied on manual verification processes which had major disadvantages:
    * **High Turnaround Time:** It took days or weeks to process a single application.
    * **Human Bias:** Decisions often varied from officer to officer.
    * **Static Rules:** Simple rules failed to see the bigger picture.
    """)

    st.write("") 

    # 2. THE SOLUTION (Native clean green alert)
    st.success("""
    **The Solution: Intelligent Automation**
    
    This project replaces the manual process with a **Hybrid Machine Learning Architecture**. 
    It combines strict financial logic (Guardrails) with AI pattern recognition (Random Forest) to make safer, faster decisions.
    """)

    st.write("")

    # 3. WORKFLOW
    st.markdown("### 🔄 Project Workflow")
    st.info("""
    This system was built in **4 key stages**:
    
    1. **Data Analysis (Jupyter Notebook):** Analyzed raw data, performed **Data Cleaning** (handling missing values), and removed outliers.
    2. **Model Training:** Selected **Random Forest Classifier** for its superior accuracy (81%) and saved the model.
    3. **Backend Logic:** Implemented "Guardrails" to reject insolvent applications (e.g., EMI > Income) instantly.
    4. **Frontend Development:** Built this Streamlit interface for real-time user interaction.
    """)

    st.divider()

    # 4. TECH SPECS
    st.subheader("🛠️ Technical Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Machine Learning:**")
        st.caption("- Algorithm: Random Forest Classifier\n- Trees: 200 Estimators\n- Accuracy: ~81.25%")
    with c2:
        st.markdown("**Tech Stack:**")
        st.caption("- Python 3.9\n- Streamlit (Frontend)\n- Pandas (Data Processing)\n- Scikit-learn (AI Modeling)")