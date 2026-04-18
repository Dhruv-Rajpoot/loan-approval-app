import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Loan Approval System",
    layout="centered"
)

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        text-align: center;
        color: white;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Intelligent Loan Approval System")


df = pd.read_csv("loan_data.csv")

X = df.drop("Approved", axis=1)
y = df["Approved"]

model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 65, 30)
income = st.slider("Income", 10000, 100000, 50000)
credit = st.slider("Credit Score", 300, 900, 700)
loan_amt = st.slider("Loan Amount", 50000, 500000, 150000)

loan_term = st.selectbox(
    "Loan Term (months)",
    [12, 24, 36, 60, 120]
)

existing_loan = st.selectbox(
    "Existing Loan",
    [0, 1]
)

experience = st.slider("Employment Years", 0, 30, 5)


if st.button("Predict Loan Approval"):
    
    input_data = [[age, income, credit, loan_amt, loan_term]]
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")