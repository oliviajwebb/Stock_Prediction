import os, warnings
import math
import numpy as np
import pandas as pd
import streamlit as st

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

warnings.simplefilter("ignore")

# ── Secrets ───────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Full feature list (252 features) ─────────────────────────────────────────
ALL_FEATURES = [
    'TransactionID', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
    'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C3', 'C4', 'C5', 'C7',
    'C9', 'C12', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
    'D11', 'D12', 'D13', 'D14', 'D15', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V12', 'V13', 'V14', 'V15', 'V17', 'V19', 'V20',
    'V23', 'V24', 'V25', 'V27', 'V29', 'V31', 'V35', 'V36', 'V37', 'V38',
    'V39', 'V41', 'V44', 'V45', 'V46', 'V47', 'V48', 'V50', 'V51', 'V53',
    'V54', 'V55', 'V56', 'V57', 'V59', 'V61', 'V65', 'V66', 'V68', 'V69',
    'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V82', 'V83', 'V86', 'V87',
    'V88', 'V89', 'V90', 'V92', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100',
    'V101', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111',
    'V112', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121',
    'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130',
    'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139',
    'V141', 'V142', 'V143', 'V144', 'V146', 'V148', 'V153', 'V161', 'V164',
    'V166', 'V167', 'V168', 'V169', 'V170', 'V172', 'V173', 'V174', 'V175',
    'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V183', 'V184', 'V186',
    'V187', 'V188', 'V191', 'V193', 'V194', 'V196', 'V199', 'V200', 'V202',
    'V204', 'V205', 'V206', 'V208', 'V209', 'V214', 'V215', 'V217', 'V218',
    'V220', 'V221', 'V223', 'V226', 'V227', 'V228', 'V234', 'V235', 'V238',
    'V240', 'V241', 'V242', 'V245', 'V250', 'V259', 'V260', 'V263', 'V270',
    'V276', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286',
    'V287', 'V288', 'V289', 'V290', 'V291', 'V293', 'V294', 'V296', 'V297',
    'V299', 'V300', 'V302', 'V303', 'V305', 'V306', 'V307', 'V308', 'V309',
    'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V319', 'V320',
    'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329',
    'V331', 'V332', 'V334', 'V335', 'V336', 'V337', 'V338', 'TransactionAmt_log'
]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor(aws_id, aws_secret, aws_token):
    session = boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )
    sm_session = sagemaker.Session(boto_session=session)
    return Predictor(
        endpoint_name=aws_endpoint,
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer()
    )

# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(feature_dict):
    row = [feature_dict.get(f, 0) for f in ALL_FEATURES]
    csv_input = ','.join(map(str, row))
    try:
        predictor = get_predictor(aws_id, aws_secret, aws_token)
        response  = predictor.predict(csv_input)
        pred = response['predictions'][0]
        prob = response['probabilities'][0]
        return pred, prob, 200
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection")
st.markdown(
    "Enter transaction details to predict whether a transaction is "
    "**fraudulent or legitimate** using a tuned Gradient Boosting pipeline."
)

with st.form("pred_form"):
    st.subheader("Transaction Details")
    st.caption("Fill in the key transaction fields. All other features default to 0.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Transaction Info**")
        transaction_id  = st.number_input("Transaction ID",         min_value=0,   value=2987004, step=1)
        transaction_amt = st.number_input("Transaction Amount ($)",  min_value=0.0, value=100.0,   step=1.0)
        product_cd      = st.number_input("Product CD (encoded)",    min_value=0,   value=1,       step=1)
        dist1           = st.number_input("Distance 1",              min_value=0.0, value=0.0,     step=1.0)
        dist2           = st.number_input("Distance 2",              min_value=0.0, value=0.0,     step=1.0)

        st.markdown("**Card Info**")
        card1 = st.number_input("Card 1 ID",    min_value=0, value=9500, step=1)
        card2 = st.number_input("Card 2 ID",    min_value=0, value=111,  step=1)
        card3 = st.number_input("Card 3 Value", min_value=0, value=150,  step=1)
        card5 = st.number_input("Card 5 Value", min_value=0, value=226,  step=1)

        st.markdown("**Address**")
        addr1 = st.number_input("Billing Address (addr1)", min_value=0, value=299, step=1)
        addr2 = st.number_input("Billing Country (addr2)", min_value=0, value=87,  step=1)

    with col2:
        st.markdown("**Count Features (C)**")
        c1  = st.number_input("C1",  min_value=0.0, value=1.0, step=1.0)
        c3  = st.number_input("C3",  min_value=0.0, value=0.0, step=1.0)
        c4  = st.number_input("C4",  min_value=0.0, value=0.0, step=1.0)
        c5  = st.number_input("C5",  min_value=0.0, value=1.0, step=1.0)
        c7  = st.number_input("C7",  min_value=0.0, value=1.0, step=1.0)
        c9  = st.number_input("C9",  min_value=0.0, value=0.0, step=1.0)
        c12 = st.number_input("C12", min_value=0.0, value=0.0, step=1.0)

        st.markdown("**Days Features (D)**")
        d1  = st.number_input("D1",  min_value=0.0, value=0.0, step=1.0)
        d2  = st.number_input("D2",  min_value=0.0, value=0.0, step=1.0)
        d4  = st.number_input("D4",  min_value=0.0, value=0.0, step=1.0)
        d10 = st.number_input("D10", min_value=0.0, value=0.0, step=1.0)
        d15 = st.number_input("D15", min_value=0.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("🔎 Predict Fraud", use_container_width=True)

if submitted:
    amt_log = math.log1p(transaction_amt)

    feature_dict = {
        'TransactionID':      transaction_id,
        'TransactionAmt':     transaction_amt,
        'TransactionAmt_log': amt_log,
        'ProductCD':          product_cd,
        'card1':              card1,
        'card2':              card2,
        'card3':              card3,
        'card5':              card5,
        'addr1':              addr1,
        'addr2':              addr2,
        'dist1':              dist1,
        'dist2':              dist2,
        'C1':                 c1,
        'C3':                 c3,
        'C4':                 c4,
        'C5':                 c5,
        'C7':                 c7,
        'C9':                 c9,
        'C12':                c12,
        'D1':                 d1,
        'D2':                 d2,
        'D4':                 d4,
        'D10':                d10,
        'D15':                d15,
    }

    pred, prob, status = call_model_api(feature_dict)

    if status == 200:
        st.markdown("## Prediction Result")

        if pred == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Probability", f"{prob:.1%}")
        with col2:
            st.metric("Prediction", "Fraud" if pred == 1 else "Legitimate")

        st.progress(float(prob))
        st.caption(f"Raw probability score: {prob:.4f} | Model: Gradient Boosting Tuned Pipeline")

        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame({
            "Feature": list(feature_dict.keys()),
            "Value":   list(feature_dict.values())
        })
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.error(status)
        st.info("Make sure your SageMaker endpoint is running and AWS credentials are valid.")

st.markdown("---")
st.caption("IEEE-CIS Fraud Detection · Olivia Webb · Milestone 4")
