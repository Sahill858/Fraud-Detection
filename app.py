import streamlit as st 
import pandas as pd
import joblib

# Load the saved XGBoost model
model = joblib.load('xgboost_fraud_model.pkl')

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

# Header

st.title("💳 Real-Time Fraud Transaction Detector")
st.markdown("Enter the transaction details below to check if it's fraudulent or genuine")

# Input form

with st.form("input form"):
    step = st.number_input("Time Step (e.g 1 - 743)", min_value=1, max_value=1000)
    type = st.selectbox("Transaction Type", ['TRANSFER', 'CASHOUT'])
    amount = st.number_input("Transaction Amoount", min_value=0.0)
    oldbalanceOrg = st.number_input("Sender old balance", min_value=0.0)
    newbalanceOrig = st.number_input("Sender new balance", min_value=0.0)
    oldbalanceDest = st.number_input("Receiver old balance", min_value=0.0)
    newbalanceDest = st.number_input("Receiver new balance", min_value=0.0)

    submitted = st.form_submit_button("Check Transaction")

# Prediction

if submitted:
    # One-hot encode transaction type
    type_CASH_OUT = 1 if type == 'CASH_OUT' else 0
    type_TRANSFER = 1 if type == 'TRANSFER' else 0
    type_PAYMENT = 1 if type == 'PAYMENT' else 0
    type_DEBIT = 1 if type == 'DEBIT' else 0

    # Feature engineering (must match training)
    errorOrig = oldbalanceOrg - newbalanceOrig - amount
    errorDest = newbalanceDest - oldbalanceDest - amount
    sender_no_balance = 1 if oldbalanceOrg == 0 else 0
    receiver_no_balance = 1 if oldbalanceDest == 0 else 0

    # Final input in exact feature order
    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER,
        'errorOrig': errorOrig,
        'errorDest': errorDest,
        'sender_no_balance': sender_no_balance,
        'receiver_no_balance': receiver_no_balance
    }])

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    if prediction ==1:
        st.error("This transaction is likely **Fraudulent**!")
    else:
        st.success("This transaction appears to be **Genuine**.")
    
    st.caption("Model trained with XGBoost. High recall ensures most frauds are caught.")