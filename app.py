import streamlit as st
import joblib
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('credit_risk_model.pkl')

model = load_model()

# Main function
def main():
    st.title("🏦 Loan Approval Predictor")
    st.markdown("Predict whether a loan application will be approved or rejected")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )
            person_emp_length = st.slider(
                "Employment Length (years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=5.0, 
                step=0.5
            )
        
        with col2:
            st.subheader("Loan Details")
            loan_intent = st.selectbox(
                "Loan Purpose",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
            )
            loan_grade = st.select_slider(
                "Loan Grade", 
                options=["A", "B", "C", "D", "E", "F", "G"],
                value="B"
            )
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            loan_int_rate = st.slider(
                "Interest Rate (%)", 
                min_value=0.0, 
                max_value=30.0, 
                value=7.5, 
                step=0.1
            )
        
        submitted = st.form_submit_button("Predict Approval")
    
    if submitted:
        # Prepare input data
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_amnt/person_income,
            'cb_person_default_on_file': 'N',  # Default value
            'cb_person_cred_hist_length': 5    # Default value
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probability of approval
        
        # Display results
        st.divider()
        if prediction:
            st.success(f"✅ Approved (Probability: {probability:.0%})")
            st.balloons()
        else:
            st.error(f"❌ Rejected (Probability of approval: {probability:.0%})")
        
        # Show feature importance (if available)
        try:
            st.subheader("Key Decision Factors")
            importances = pd.Series(model.feature_importances_, index=input_df.columns)
            st.bar_chart(importances.sort_values(ascending=False))
        except:
            pass

if __name__ == '__main__':
    main()