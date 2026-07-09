import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

BASE = os.path.dirname(__file__)

st.set_page_config(page_title="MSME Liquidity Risk Screener", page_icon="💧", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, "model.joblib"))
    with open(os.path.join(BASE, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(BASE, "cat_options.json")) as f:
        cat_options = json.load(f)
    with open(os.path.join(BASE, "num_defaults.json")) as f:
        num_defaults = json.load(f)
    with open(os.path.join(BASE, "columns_meta.json")) as f:
        meta = json.load(f)
    return model, feature_columns, cat_options, num_defaults, meta

model, feature_columns, cat_options, num_defaults, meta = load_artifacts()
cat_cols = meta["cat_cols"]

st.title("💧 MSME Liquidity Risk Screener")
st.caption(
    "Predicts the likelihood that a micro/small/medium enterprise in Kenya, Uganda, or Ghana "
    "currently reports a binding liquidity constraint, using a stacking ensemble "
    "(Random Forest + XGBoost + Gradient Boosting → logistic meta-learner). "
    "This is a research prototype from a DSA 8201 coursework analysis, not a production credit-risk tool."
)

st.divider()
st.subheader("Firm profile")

col1, col2 = st.columns(2)
with col1:
    nationality = st.selectbox("Country", [c for c in cat_options["Nationality"] if c != "DRC"])
    gender = st.selectbox("Owner gender", cat_options.get("Gender", ["Male", "Female"]))
    customer_age = st.number_input("Owner age", min_value=18, max_value=100, value=40)
with col2:
    years_in_operation = st.number_input("Years in operation", min_value=0.0, max_value=60.0, value=5.0)
    gross_income = st.number_input("Gross income (local currency)", min_value=0.0, value=5000.0, step=100.0)
    operating_expense = st.number_input("Operating expense (local currency)", min_value=0.0, value=3000.0, step=100.0)

business_sales = st.number_input("Business sales (local currency)", min_value=0.0, value=6000.0, step=100.0)

st.divider()
st.subheader("Financial product access")

fin_cols = ["mobile_wallet_access", "credit_card_ownership", "active_loan_holder",
            "has_internet_banking", "has_debit_card"]
fin_labels = {
    "mobile_wallet_access": "Mobile wallet access",
    "credit_card_ownership": "Credit card ownership",
    "active_loan_holder": "Active loan holder",
    "has_internet_banking": "Internet banking",
    "has_debit_card": "Debit card",
}
fin_inputs = {}
cols = st.columns(2)
for i, c in enumerate(fin_cols):
    with cols[i % 2]:
        fin_inputs[c] = st.selectbox(fin_labels[c], cat_options[c], key=c)

st.divider()
st.subheader("Business behaviour & risk perception")

behaviour_cols = [c for c in cat_cols if c not in fin_cols + ["Nationality", "Gender"]]
behaviour_labels = {c: c.replace("_", " ").capitalize() for c in behaviour_cols}
behaviour_inputs = {}
cols = st.columns(2)
for i, c in enumerate(behaviour_cols):
    with cols[i % 2]:
        behaviour_inputs[c] = st.selectbox(behaviour_labels[c], cat_options[c], key=c)

st.divider()

def has_now(x):
    return 1 if isinstance(x, str) and x.strip().lower() == "have now" else 0

if st.button("Predict liquidity risk", type="primary", use_container_width=True):
    row = {
        "Nationality": nationality,
        "Gender": gender,
        "customer_age": customer_age,
        "years_in_operation": years_in_operation,
        "gross_income": gross_income,
        "operating_expense": operating_expense,
        "business_sales": business_sales,
    }
    row.update(fin_inputs)
    row.update(behaviour_inputs)

    # Engineered features -- must mirror the training pipeline exactly
    net_profit = gross_income - operating_expense
    profit_margin = net_profit / gross_income if gross_income > 0 else num_defaults["profit_margin"]
    expense_to_income_ratio = operating_expense / gross_income if gross_income > 0 else num_defaults["expense_to_income_ratio"]
    sales_to_expense_ratio = business_sales / operating_expense if operating_expense > 0 else num_defaults["sales_to_expense_ratio"]
    log_gross_income = np.log1p(max(gross_income, 0))
    log_business_sales = np.log1p(max(business_sales, 0))

    row["net_profit"] = net_profit
    row["profit_margin"] = profit_margin
    row["expense_to_income_ratio"] = expense_to_income_ratio
    row["sales_to_expense_ratio"] = sales_to_expense_ratio
    row["log_gross_income"] = log_gross_income
    row["log_business_sales"] = log_business_sales

    incl_cols = ["mobile_wallet_access", "credit_card_ownership", "active_loan_holder",
                 "has_internet_banking", "has_debit_card"]
    score = 0
    for c in incl_cols:
        flag = has_now(row[c])
        row[c + "_now"] = flag
        score += flag
    row["financial_inclusion_score"] = score

    input_df = pd.DataFrame([row])
    input_enc = pd.get_dummies(input_df, columns=cat_cols)

    # Align to the exact training column layout -- add any missing dummy columns as 0, drop extras
    input_aligned = input_enc.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(input_aligned)[0, 1]
    pred = int(proba >= 0.5)

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Predicted: **Liquidity constrained** (probability {proba:.1%})")
    else:
        st.success(f"✅ Predicted: **No binding liquidity constraint** (probability of constraint {proba:.1%})")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "Probability reflects the stacking ensemble's confidence based on patterns in survey data from "
        "7,674 MSMEs across Kenya, Uganda, and Ghana. Test-set ROC-AUC = 0.787."
    )

st.divider()
with st.expander("About this model"):
    st.markdown("""
    - **Data:** MSME survey, 9,618 firms (Kenya, Uganda, Ghana, DRC); liquidity constraint modelled on 7,674
      firms in Kenya, Uganda, and Ghana (DRC excluded — the field was not administered there).
    - **Model:** Stacking ensemble — Random Forest, XGBoost, and Gradient Boosting base learners with a
      logistic regression meta-learner.
    - **Performance:** Test ROC-AUC 0.787; 5-fold CV ROC-AUC 0.775 ± 0.014.
    - **Caveat:** this predicts a *current, self-reported* liquidity constraint — not business failure risk,
      creditworthiness, or a formal solvency ratio. Treat outputs as a screening signal, not a lending decision.
    """)
