
import streamlit as st
import pandas as pd
import joblib

# Load the saved Decision Tree model
bundle = joblib.load("model_bundle.joblib")
model = bundle["model"]

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")
st.subheader("Using Decision Tree Classifier")

st.write(
    "Enter the applicant details below to predict "
    "whether the loan application will be approved or rejected."
)

st.info("This project is developed for educational purposes.")

st.markdown("---")
st.header("Applicant Information")

with st.form("loan_form"):

    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    gender = st.selectbox(
        "Gender",
        bundle["category_options"]["person_gender"]
    )

    education = st.selectbox(
        "Education",
        bundle["category_options"]["person_education"]
    )

    income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    experience = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=60,
        value=2
    )

    home = st.selectbox(
        "Home Ownership",
        bundle["category_options"]["person_home_ownership"]
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    intent = st.selectbox(
        "Loan Intent",
        bundle["category_options"]["loan_intent"]
    )

    interest_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=0.0,
        value=10.5
    )

    loan_percent_income = st.number_input(
        "Loan Amount / Annual Income (e.g. 0.20 for 20%)",
        min_value=0.0,
        max_value=1.0,
        value=0.20
    )

    credit_history = st.number_input(
        "Credit History Length (Years)",
        min_value=0.0,
        value=4.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=1000,
        value=700
    )

    previous_defaults = st.selectbox(
        "Previous Loan Defaults",
        bundle["category_options"]["previous_loan_defaults_on_file"]
    )

    submit = st.form_submit_button("Predict Loan Approval")

if submit:

    input_data = {
        "person_age": age,
        "person_gender": gender,
        "person_education": education,
        "person_income": income,
        "person_emp_exp": experience,
        "person_home_ownership": home,
        "loan_amnt": loan_amount,
        "loan_intent": intent,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": credit_history,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_defaults
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=bundle["raw_feature_columns"]
    )

    # Handle missing values using saved training values
    for column in bundle["numerical_columns"]:
        input_df[column] = input_df[column].fillna(
            bundle["numerical_medians"][column]
        )

    for column in bundle["categorical_columns"]:
        input_df[column] = input_df[column].fillna(
            bundle["categorical_modes"][column]
        )

    # Apply the same encoding used during model training
    processed_df = pd.get_dummies(input_df, drop_first=True)

    processed_df = processed_df.reindex(
        columns=bundle["encoded_columns"],
        fill_value=0
    )

    # Predict using the trained model
    prediction = model.predict(processed_df)[0]

    st.markdown("---")
    st.header("Prediction Result")

    if prediction == 1:
        st.success("Loan Application: APPROVED")
    else:
        st.error("Loan Application: REJECTED")

    st.caption(
        "This is a machine learning prediction, not an actual "
        "bank approval or financial decision."
    )
