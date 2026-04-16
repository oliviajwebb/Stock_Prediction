import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

from sklearn.pipeline import Pipeline

from joblib import dump
from joblib import load

# ── Setup ────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

# ── Secrets ──────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "pipeline": "finalized_fraud_model.tar.gz",
    "keys": [
        "TransactionAmt", "TransactionAmt_log",
        "card1", "card2", "card3", "card5",
        "addr1", "addr2", "dist1",
        "C1", "C2", "D1", "V1", "V2", "V3"
    ],
    "inputs": [
        {"name": "TransactionAmt",     "label": "Transaction Amount ($)",   "min": 0.0,   "max": 50000.0, "default": 100.0,  "step": 1.0},
        {"name": "card1",              "label": "Card 1 ID",                 "min": 0.0,   "max": 20000.0, "default": 9500.0, "step": 1.0},
        {"name": "card2",              "label": "Card 2 ID",                 "min": 0.0,   "max": 1000.0,  "default": 111.0,  "step": 1.0},
        {"name": "card3",              "label": "Card 3 Value",              "min": 0.0,   "max": 300.0,   "default": 150.0,  "step": 1.0},
        {"name": "card5",              "label": "Card 5 Value",              "min": 0.0,   "max": 300.0,   "default": 226.0,  "step": 1.0},
        {"name": "addr1",              "label": "Billing Address (addr1)",   "min": 0.0,   "max": 500.0,   "default": 299.0,  "step": 1.0},
        {"name": "addr2",              "label": "Billing Country (addr2)",   "min": 0.0,   "max": 100.0,   "default": 87.0,   "step": 1.0},
        {"name": "dist1",              "label": "Distance 1",                "min": 0.0,   "max": 10000.0, "default": 0.0,    "step": 1.0},
        {"name": "C1",                 "label": "C1 Count Feature",          "min": 0.0,   "max": 2500.0,  "default": 1.0,    "step": 1.0},
        {"name": "C2",                 "label": "C2 Count Feature",          "min": 0.0,   "max": 2500.0,  "default": 1.0,    "step": 1.0},
        {"name": "D1",                 "label": "D1 Days Feature",           "min": 0.0,   "max": 640.0,   "default": 0.0,    "step": 1.0},
        {"name": "V1",                 "label": "V1 (anonymized)",           "min": 0.0,   "max": 1.0,     "default": 1.0,    "step": 0.01},
        {"name": "V2",                 "label": "V2 (anonymized)",           "min": 0.0,   "max": 1.0,     "default": 1.0,    "step": 0.01},
        {"name": "V3",                 "label": "V3 (anonymized)",           "min": 0.0,   "max": 1.0,     "default": 1.0,    "step": 0.01},
    ]
}

# ── Load Pipeline from S3 ─────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)

# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer()
    )

    try:
        csv_input = ','.join(map(str, input_df.values[0]))
        response  = predictor.predict(csv_input)
        pred      = response['predictions'][0]
        prob      = response['probabilities'][0]
        return pred, prob, 200
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection")
st.markdown(
    "Enter transaction details below to predict whether a transaction is "
    "**fraudulent or legitimate** using a tuned K-Nearest Neighbors pipeline."
)

with st.form("pred_form"):
    st.subheader("Transaction Details")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )

    submitted = st.form_submit_button("🔎 Predict Fraud", use_container_width=True)

if submitted:
    import math

    # Compute log-transformed amount and build full feature row
    amt     = user_inputs["TransactionAmt"]
    amt_log = math.log1p(amt)

    data_row = [
        amt, amt_log,
        user_inputs["card1"],  user_inputs["card2"],
        user_inputs["card3"],  user_inputs["card5"],
        user_inputs["addr1"],  user_inputs["addr2"],
        user_inputs["dist1"],
        user_inputs["C1"],     user_inputs["C2"],
        user_inputs["D1"],
        user_inputs["V1"],     user_inputs["V2"],     user_inputs["V3"]
    ]

    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    pred, prob, status = call_model_api(input_df)

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
        st.caption(f"Raw probability score: {prob:.4f} | Model: KNN Tuned Pipeline")

        # Feature summary table
        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame({
            "Feature": MODEL_INFO["keys"],
            "Value":   data_row
        })
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.error(status)
        st.info("Make sure your SageMaker endpoint is running and AWS credentials are valid.")

st.markdown("---")
st.caption("IEEE-CIS Fraud Detection · Olivia Webb · Milestone 4")
