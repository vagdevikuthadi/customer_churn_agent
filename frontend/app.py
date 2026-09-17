import sys
import os
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from predict import predict_churn, model


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")

try:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)
except Exception:
    df = pd.DataFrame()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "custom_profile" not in st.session_state:
    st.session_state.custom_profile = {}


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #eef5ef;
}

.block-container {
    max-width: 1050px;
    padding-top: 35px;
    padding-bottom: 40px;
}

.header {
    background: #145a32;
    color: white;
    padding: 25px 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

.header h1 {
    margin: 0;
    font-size: 32px;
}

.header p {
    margin: 8px 0 0 0;
    font-size: 16px;
}

.feature-card {
    background: white;
    min-height: 245px;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #cfe3d2;
    box-shadow: 0 5px 15px rgba(0,0,0,0.06);
    text-align: center;
    margin-bottom: 25px;
}

.feature-card h2 {
    color: #145a32;
    margin-top: 5px;
}

.feature-card p {
    color: #555;
    min-height: 55px;
}

.card-link {
    display: inline-block;
    background: #1b6e35;
    color: white !important;
    text-decoration: none !important;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 600;
}

.inner-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #cfe3d2;
    margin-bottom: 18px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}

.inner-card h3 {
    color: #145a32;
    margin-top: 0;
}

.result-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #cfe3d2;
    margin: 18px 0;
}

.page-title {
    color: #145a32;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 8px;
}

.back-link {
    color: #145a32;
    text-decoration: none;
    font-weight: 600;
}

.metric-box {
    background: #f5faf6;
    border: 1px solid #d5e8d8;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.metric-title {
    color: #666;
    font-size: 13px;
}

.metric-value {
    color: #145a32;
    font-size: 24px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="header">
    <h1>🤖 ChurnGuard AI</h1>
    <p>Customer churn prediction made simple</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
page = st.query_params.get("page", "Home")

if isinstance(page, list):
    page = page[0]


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def customer_to_model_data(customer):
    return {
        "gender": customer["gender"],
        "SeniorCitizen": int(customer["SeniorCitizen"]),
        "Partner": customer["Partner"],
        "Dependents": customer["Dependents"],
        "tenure": int(customer["tenure"]),
        "PhoneService": customer["PhoneService"],
        "MultipleLines": customer["MultipleLines"],
        "InternetService": customer["InternetService"],
        "OnlineSecurity": customer["OnlineSecurity"],
        "OnlineBackup": customer["OnlineBackup"],
        "DeviceProtection": customer["DeviceProtection"],
        "TechSupport": customer["TechSupport"],
        "StreamingTV": customer["StreamingTV"],
        "StreamingMovies": customer["StreamingMovies"],
        "Contract": customer["Contract"],
        "PaperlessBilling": customer["PaperlessBilling"],
        "PaymentMethod": customer["PaymentMethod"],
        "MonthlyCharges": float(customer["MonthlyCharges"]),
        "TotalCharges": float(customer["TotalCharges"])
    }


def get_risk(probability):
    if probability >= 70:
        return "High"
    elif probability >= 40:
        return "Medium"
    return "Low"


def get_risk_reasons(customer):
    reasons = []

    if int(customer["tenure"]) <= 12:
        reasons.append("Short customer tenure")

    if customer["Contract"] == "Month-to-month":
        reasons.append("Month-to-month contract")

    if customer["InternetService"] == "Fiber optic":
        reasons.append("Fiber optic internet service")

    if customer["OnlineSecurity"] == "No":
        reasons.append("No online security")

    if customer["OnlineBackup"] == "No":
        reasons.append("No online backup")

    if customer["TechSupport"] == "No":
        reasons.append("No technical support")

    if float(customer["MonthlyCharges"]) >= 80:
        reasons.append("High monthly charges")

    if customer["PaymentMethod"] == "Electronic check":
        reasons.append("Electronic check payment")

    if not reasons:
        reasons.append("No major rule-based risk factors detected")

    return reasons


def get_recommendations(customer):
    recommendations = []

    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer suitable longer-term contract options."
        )

    if customer["TechSupport"] == "No":
        recommendations.append(
            "Provide proactive technical support assistance."
        )

    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Explain the benefits of online security services."
        )

    if customer["OnlineBackup"] == "No":
        recommendations.append(
            "Recommend online backup based on customer needs."
        )

    if float(customer["MonthlyCharges"]) >= 80:
        recommendations.append(
            "Review pricing and available plan options."
        )

    if int(customer["tenure"]) <= 12:
        recommendations.append(
            "Use an early-stage onboarding and engagement program."
        )

    if not recommendations:
        recommendations.append(
            "Continue regular engagement and monitor future churn risk."
        )

    return recommendations

def grouped_feature_importance():

    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    importances = rf_model.feature_importances_
    grouped = {}

    index = 0

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder":
            continue

        if hasattr(transformer, "categories_"):

            for col, categories in zip(columns, transformer.categories_):
                count = len(categories)
                grouped[col] = importances[index:index + count].sum()
                index += count

        elif hasattr(transformer, "named_steps"):

            encoder = None

            for step in transformer.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step
                    break

            if encoder is not None:

                for col, categories in zip(columns, encoder.categories_):
                    count = len(categories)
                    grouped[col] = importances[index:index + count].sum()
                    index += count

            else:

                for col in columns:
                    grouped[col] = importances[index]
                    index += 1

        else:

            for col in columns:
                grouped[col] = importances[index]
                index += 1

    names = {
        "SeniorCitizen": "Senior Citizen",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "MultipleLines": "Multiple Lines",
        "PhoneService": "Phone Service",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "Dependents": "Dependents",
        "Partner": "Partner",
        "Contract": "Contract",
        "tenure": "Tenure",
        "gender": "Gender"
    }

    data = pd.DataFrame(
        list(grouped.items()),
        columns=["Factor", "Importance"]
    )

    data["Factor"] = data["Factor"].replace(names)

    return data.sort_values(
        "Importance",
        ascending=False
    ).head(8)


def display_customer_details(customer):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Customer ID</div>'
            f'<div class="metric-value">{customer["customerID"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Tenure</div>'
            f'<div class="metric-value">{customer["tenure"]} months</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Monthly Charges</div>'
            f'<div class="metric-value">'
            f'${float(customer["MonthlyCharges"]):.2f}'
            f'</div></div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":

    st.markdown(
        '<div class="page-title">Customer Analytics</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore customer information, behaviour, insights and churn prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="feature-card">
            <h2>👤 Customer Profile</h2>
            <p>Enter and customize customer details.</p>
            <a class="card-link" href="?page=Profile">Open Profile</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h2>💰 Customer Insights</h2>
            <p>Explore financial and account insights available in the customer dataset.</p>
            <a class="card-link" href="?page=Insights">Open Insights</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="feature-card">
            <h2>📊 Customer Behaviour</h2>
            <p>Understand tenure, contracts, subscribed services and customer engagement factors.</p>
            <a class="card-link" href="?page=Behaviour">Open Behaviour</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h2>🔮 Churn Prediction</h2>
            <p>Load a customer, predict churn probability, identify risk and receive retention actions.</p>
            <a class="card-link" href="?page=Prediction">Predict Churn</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">AI-powered customer churn analysis</div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PROFILE - MANUAL ENTRY ONLY
# ---------------------------------------------------------
elif page == "Profile":

    st.markdown(
        '<a class="back-link" href="?page=Home">← Back to Home</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">👤 Customer Profile</div>',
        unsafe_allow_html=True
    )

    st.write("Enter customer information manually.")

    st.markdown(
        '<div class="inner-card"><h3>Customer Details</h3></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input(
            "Customer ID",
            placeholder="Example: C001"
        )

        customer_name = st.text_input(
            "Customer Name",
            placeholder="Enter customer name"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=18
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

    with col2:
        location = st.text_input(
            "Location",
            placeholder="Example: Hyderabad"
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["New Customer", "Regular Customer", "Premium Customer"]
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=1
        )

        contact_preference = st.selectbox(
            "Contact Preference",
            ["Email", "Phone", "SMS", "WhatsApp"]
        )

    st.markdown(
        '<div class="inner-card"><h3>Additional Information</h3></div>',
        unsafe_allow_html=True
    )

    notes = st.text_area(
        "Customer Notes",
        placeholder="Enter any additional customer information..."
    )

    if st.button(
        "💾 Save Customer Profile",
        use_container_width=True
    ):

        if not customer_id or not customer_name:
            st.warning(
                "Please enter at least Customer ID and Customer Name."
            )
        else:

            st.session_state.custom_profile = {
                "Customer ID": customer_id,
                "Customer Name": customer_name,
                "Age": age,
                "Gender": gender,
                "Location": location,
                "Customer Type": customer_type,
                "Tenure": tenure,
                "Contact Preference": contact_preference,
                "Customer Notes": notes
            }

            st.success("Customer profile saved successfully! ✅")

    if st.session_state.custom_profile:

        st.markdown(
            '<div class="inner-card"><h3>Saved Customer Profile</h3></div>',
            unsafe_allow_html=True
        )

        saved = st.session_state.custom_profile

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Customer ID:** {saved['Customer ID']}")
            st.write(f"**Customer Name:** {saved['Customer Name']}")
            st.write(f"**Age:** {saved['Age']}")
            st.write(f"**Gender:** {saved['Gender']}")
            st.write(f"**Location:** {saved['Location']}")

        with col2:
            st.write(f"**Customer Type:** {saved['Customer Type']}")
            st.write(f"**Tenure:** {saved['Tenure']} months")
            st.write(
                f"**Contact Preference:** "
                f"{saved['Contact Preference']}"
            )

        if saved["Customer Notes"]:
            st.write(f"**Notes:** {saved['Customer Notes']}")


# ---------------------------------------------------------
# BEHAVIOUR
# ---------------------------------------------------------
elif page == "Behaviour":

    st.markdown(
        '<a class="back-link" href="?page=Home">← Back to Home</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">📊 Customer Behaviour</div>',
        unsafe_allow_html=True
    )

    if df.empty:
        st.error("Customer dataset could not be loaded.")
        st.stop()

    customer_id = st.selectbox(
        "Select Customer ID",
        df["customerID"].tolist()
    )

    customer = df[df["customerID"] == customer_id].iloc[0]

    st.markdown(
        '<div class="inner-card"><h3>Customer Relationship</h3></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Tenure", f"{customer['tenure']} months")

    with col2:
        st.metric("Contract Type", customer["Contract"])

    st.markdown(
        '<div class="inner-card"><h3>Services & Engagement</h3></div>',
        unsafe_allow_html=True
    )

    services = {
        "Phone Service": customer["PhoneService"],
        "Multiple Lines": customer["MultipleLines"],
        "Internet Service": customer["InternetService"],
        "Online Security": customer["OnlineSecurity"],
        "Online Backup": customer["OnlineBackup"],
        "Device Protection": customer["DeviceProtection"],
        "Tech Support": customer["TechSupport"],
        "Streaming TV": customer["StreamingTV"],
        "Streaming Movies": customer["StreamingMovies"]
    }

    service_df = pd.DataFrame(
        list(services.items()),
        columns=["Service", "Customer Status"]
    )

    st.dataframe(
        service_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This dataset contains subscription and account information. "
        "It does not contain actual monthly usage, complaint counts "
        "or support-call history."
    )


# ---------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------
elif page == "Insights":

    st.markdown(
        '<a class="back-link" href="?page=Home">← Back to Home</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">💰 Customer Insights</div>',
        unsafe_allow_html=True
    )

    if df.empty:
        st.error("Customer dataset could not be loaded.")
        st.stop()

    customer_id = st.selectbox(
        "Select Customer ID",
        df["customerID"].tolist()
    )

    customer = df[df["customerID"] == customer_id].iloc[0]

    st.markdown(
        '<div class="inner-card"><h3>Financial Insights</h3></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Monthly Charges",
            f"${float(customer['MonthlyCharges']):.2f}"
        )

    with col2:
        st.metric(
            "Total Charges",
            f"${float(customer['TotalCharges']):.2f}"
        )

    with col3:
        st.metric(
            "Tenure",
            f"{customer['tenure']} months"
        )

    st.markdown(
        '<div class="inner-card"><h3>Account Insights</h3></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Contract:** {customer['Contract']}")
        st.write(f"**Payment Method:** {customer['PaymentMethod']}")
        st.write(
            f"**Paperless Billing:** {customer['PaperlessBilling']}"
        )

    with col2:
        st.write(
            f"**Internet Service:** {customer['InternetService']}"
        )
        st.write(
            f"**Online Security:** {customer['OnlineSecurity']}"
        )
        st.write(
            f"**Tech Support:** {customer['TechSupport']}"
        )

    st.info(
        "Insights shown here are based on fields available "
        "in the Telco customer dataset."
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
elif page == "Prediction":

    st.markdown(
        '<a class="back-link" href="?page=Home">← Back to Home</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">🔮 Churn Prediction</div>',
        unsafe_allow_html=True
    )

    if df.empty:
        st.error("Customer dataset could not be loaded.")
        st.stop()

    st.write(
        "Select a customer to load their actual information "
        "and run the ML prediction."
    )

    customer_ids = df["customerID"].tolist()

    selected_id = st.selectbox(
        "Customer ID",
        customer_ids,
        index=0
    )

    customer = df[df["customerID"] == selected_id].iloc[0]

    display_customer_details(customer)

    st.markdown(
        '<div class="inner-card"><h3>Customer Information</h3></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Gender:** {customer['gender']}")
        st.write(
            f"**Senior Citizen:** "
            f"{'Yes' if int(customer['SeniorCitizen']) else 'No'}"
        )
        st.write(f"**Partner:** {customer['Partner']}")
        st.write(f"**Dependents:** {customer['Dependents']}")
        st.write(f"**Phone Service:** {customer['PhoneService']}")

    with col2:
        st.write(f"**Multiple Lines:** {customer['MultipleLines']}")
        st.write(
            f"**Internet Service:** {customer['InternetService']}"
        )
        st.write(
            f"**Online Security:** {customer['OnlineSecurity']}"
        )
        st.write(
            f"**Online Backup:** {customer['OnlineBackup']}"
        )
        st.write(
            f"**Device Protection:** {customer['DeviceProtection']}"
        )

    with col3:
        st.write(f"**Tech Support:** {customer['TechSupport']}")
        st.write(f"**Streaming TV:** {customer['StreamingTV']}")
        st.write(
            f"**Streaming Movies:** {customer['StreamingMovies']}"
        )
        st.write(f"**Contract:** {customer['Contract']}")
        st.write(
            f"**Payment Method:** {customer['PaymentMethod']}"
        )

    st.markdown(
        '<div class="inner-card"><h3>Run Prediction</h3></div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🔮 Predict Churn",
        use_container_width=True
    ):

        try:

            customer_data = customer_to_model_data(customer)

            result = predict_churn(customer_data)

            st.session_state.prediction_result = result

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    result = st.session_state.prediction_result

    if result is not None:

        prediction = result["prediction"]
        probability = result["probability"]
        risk = get_risk(probability)

        st.markdown(
            '<div class="result-card"><h3>Prediction Result</h3></div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Classification", prediction)

        with col2:
            st.metric(
                "Churn Probability",
                f"{probability:.2f}%"
            )

        with col3:
            st.metric("Risk Level", risk)

        st.progress(
            min(max(probability / 100, 0), 1),
            text=f"Churn probability: {probability:.2f}%"
        )

        st.markdown(
            '<div class="inner-card"><h3>⚠️ Risk Reasons</h3></div>',
            unsafe_allow_html=True
        )

        reasons = get_risk_reasons(customer)

        for reason in reasons:
            st.write(f"• {reason}")

        st.markdown(
            '<div class="inner-card">'
            '<h3>🎯 Retention Recommendations</h3></div>',
            unsafe_allow_html=True
        )

        recommendations = get_recommendations(customer)

        for recommendation in recommendations:
            st.write(f"• {recommendation}")

        st.markdown(
            '<div class="inner-card">'
            '<h3>📌 Important Churn Factors</h3></div>',
            unsafe_allow_html=True
        )

        feature_data = grouped_feature_importance()

        st.bar_chart(
            feature_data.set_index("Factor")
        )

        st.markdown(
            '<div class="inner-card">'
            '<h3>🔥 High-Risk Customers</h3></div>',
            unsafe_allow_html=True
        )

        try:

            prediction_data = df.drop(
                columns=["customerID", "Churn"]
            ).copy()

            prediction_data["TotalCharges"] = pd.to_numeric(
                prediction_data["TotalCharges"],
                errors="coerce"
            ).fillna(0)

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1] * 100

            high_risk_df = pd.DataFrame({
                "Customer ID": df["customerID"],
                "Churn Probability": probabilities
            })

            high_risk_df["Risk"] = (
                high_risk_df["Churn Probability"]
                .apply(get_risk)
            )

            risk_counts = (
                high_risk_df["Risk"]
                .value_counts()
                .reindex(
                    ["High", "Medium", "Low"],
                    fill_value=0
                )
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "High Risk",
                    int(risk_counts["High"])
                )

            with c2:
                st.metric(
                    "Medium Risk",
                    int(risk_counts["Medium"])
                )

            with c3:
                st.metric(
                    "Low Risk",
                    int(risk_counts["Low"])
                )

            high_risk_only = high_risk_df[
                high_risk_df["Risk"] == "High"
            ].sort_values(
                "Churn Probability",
                ascending=False
            ).head(10)

            if high_risk_only.empty:

                st.success(
                    "No high-risk customers were identified."
                )

            else:

                high_risk_only["Churn Probability"] = (
                    high_risk_only["Churn Probability"]
                    .round(2)
                    .astype(str) + "%"
                )

                st.dataframe(
                    high_risk_only,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.warning(
                f"High-risk analysis could not be completed: {e}"
            )


# ---------------------------------------------------------
# UNKNOWN PAGE
# ---------------------------------------------------------
else:

    st.error("Page not found.")

    st.markdown(
        '<a class="back-link" href="?page=Home">'
        '← Return to Home</a>',
        unsafe_allow_html=True
    )