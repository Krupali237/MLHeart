import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
import os

port = int(os.environ.get("PORT", 8501))



# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="HeartSafe AI",
    page_icon="🫀",
    layout="wide"
)

# ===============================
# LOAD MODEL & DATA
# ===============================
model = joblib.load("model.pkl")

df = pd.read_csv("cardio_train.csv", sep=";")


df["age"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

features = [
    "age", "height", "weight",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi"
]

# ===============================
# MODEL EVALUATION
# ===============================
X = df[features]
y = df["cardio"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)
auc_score = roc_auc_score(y, y_prob)

cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual No Risk", "Actual Risk"],
    columns=["Predicted No Risk", "Predicted Risk"]
)

report_df = pd.DataFrame(
    classification_report(y, y_pred, output_dict=True)
).transpose().round(2)

fpr, tpr, _ = roc_curve(y, y_prob)

# ===============================
# PREMIUM UI CSS
# ===============================
st.markdown("""
<style>

/* =========================
   GOOGLE FONT
========================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================
   PAGE BACKGROUND
========================= */
body {
    background: radial-gradient(circle at top, #1E293B, #020617);
    color: #E5E7EB;
}

.block-container {
    padding-top: 2.5rem;
}

/* =========================
   HERO SECTION
========================= */
.hero {
    background: linear-gradient(135deg, #2563EB, #22C55E);
    padding: 55px;
    border-radius: 30px;
    margin-bottom: 40px;
    animation: fadeSlide 1s ease;
}

.hero h1 {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.hero p {
    font-size: 18px;
    opacity: 0.95;
}

/* =========================
   CARDS (GLASS EFFECT)
========================= */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.4);
    margin-bottom: 28px;
    transition: all 0.4s ease;
}

.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 30px 60px rgba(0,0,0,0.6);
}

/* =========================
   METRICS
========================= */
.metric {
    font-size: 38px;
    font-weight: 800;
    color: #22C55E;
}

.label {
    font-size: 14px;
    color: #CBD5F5;
}

/* =========================
   SECTION TITLES
========================= */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #93C5FD;
    margin: 35px 0 15px;
}

/* =========================
   STORY BOX
========================= */
.story {
    background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(34,197,94,0.25));
    padding: 28px;
    border-radius: 20px;
    font-size: 16px;
    line-height: 1.6;
    animation: fadeSlide 0.8s ease;
}

/* =========================
   ANIMATION
========================= */
@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0F172A);
    border-right: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)


# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🫀 HeartSafe AI")
st.sidebar.caption("ML-powered Cardiac Risk System")

menu = st.sidebar.radio(
    "Navigate",
    ["Overview", "Risk Prediction", "Visual Insights", "About Project"]
)

# ===============================
# OVERVIEW PAGE
# ===============================
if menu == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>HeartSafe AI</h1>
        <p>End-to-end Machine Learning system for heart disease risk analysis with interactive insights.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{len(df):,}</div>
            <div class="label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{df["cardio"].sum():,}</div>
            <div class="label">High Risk Patients</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{accuracy*100:.2f}%</div>
            <div class="label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{auc_score:.2f}</div>
            <div class="label">ROC–AUC Score</div>
        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # AGE DISTRIBUTION
    # ===============================
    st.markdown("<div class='section-title'>📊 Risk Distribution by Age</div>", unsafe_allow_html=True)

    fig_age = px.histogram(
        df,
        x="age",
        color="cardio",
        nbins=35,
        color_discrete_sequence=["#34D399", "#EF4444"]
    )
    fig_age.update_layout(
    bargap=0.1,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title="Age",
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    ),
    yaxis=dict(
        title="Count",
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    ),
    legend=dict(
        font=dict(color="#064E3B")
    )
)

    st.plotly_chart(fig_age, use_container_width=True)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    st.markdown("<div class='section-title'>📌 Confusion Matrix</div>", unsafe_allow_html=True)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Teal"
    )
    fig_cm.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    ),
    yaxis=dict(
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    )
)

    st.plotly_chart(fig_cm, use_container_width=True)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.markdown("<div class='section-title'>📄 Classification Report</div>", unsafe_allow_html=True)
    st.dataframe(report_df, use_container_width=True)

    # ===============================
    # ROC CURVE
    # ===============================
    st.markdown("<div class='section-title'>📈 ROC Curve</div>", unsafe_allow_html=True)

    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    fig_roc = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate"
    )
    fig_roc.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray")
    )
    fig_roc.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title="False Positive Rate",
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    ),
    yaxis=dict(
        title="True Positive Rate",
        title_font=dict(color="#064E3B", size=14),
        tickfont=dict(color="#064E3B")
    )
)

    st.plotly_chart(fig_roc, use_container_width=True)

# ===============================
# RISK PREDICTION (UNIQUE)
# ===============================
elif menu == "Risk Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Personalized Heart Risk Insight</h1>
<p>Interpretable results derived from your health data.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 90)
            height = st.slider("Height (cm)", 140, 200)
            weight = st.slider("Weight (kg)", 40, 150)

        with col2:
            ap_hi = st.slider("Systolic BP", 90, 200)
            ap_lo = st.slider("Diastolic BP", 60, 140)
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3])

        with col3:
            gluc = st.selectbox("Glucose", [1, 2, 3])
            smoke = st.selectbox("Smoking", [0, 1])
            alco = st.selectbox("Alcohol", [0, 1])
            active = st.selectbox("Physical Activity", [0, 1])

        submit = st.form_submit_button("Analyze My Heart Health")


    if submit:
        bmi = weight / ((height / 100) ** 2)

        input_data = np.array([[age, height, weight, ap_hi, ap_lo,
                                cholesterol, gluc, smoke, alco, active, bmi]])

        prob = model.predict_proba(input_data)[0][1] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "#D1FAE5"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#FECACA"},
                ],
                'bar': {'color': "#047857"}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card">
            <h3 style="color:#047857;">Risk Score</h3>
            <div class="metric">{prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # 🧠 UNIQUE FEATURE: RISK STORY
        if prob > 70:
            story = (
        "Your cardiovascular risk is HIGH.\n\n"
        "What you should do:\n"
        "- Consult a cardiologist immediately\n"
        "- Reduce salt & oily food\n"
        "- Start daily walking (30 mins)\n"
        "- Monitor BP regularly"
    )
        elif prob > 40:
            story = (
        "Your cardiovascular risk is MODERATE.\n\n"
        "What you should do:\n"
        "- Improve diet\n"
        "- Exercise 4–5 days/week\n"
        "- Avoid stress & smoking\n"
        "- Regular health checkups"
    )
        else:
            story = (
        "Your cardiovascular profile appears HEALTHY.\n\n"
        "What you should do:\n"
        "- Maintain healthy lifestyle\n"
        "- Balanced diet\n"
        "- Regular exercise\n"
        "- Annual checkup"
    )
        

        st.markdown(f"""
        <div class="story">
            <b>🧠 AI Risk Interpretation</b><br><br>
            {story}
        </div>
        """, unsafe_allow_html=True)
        

# ===============================
# VISUAL INSIGHTS
# ===============================
elif menu == "Visual Insights":
    st.markdown("""
    <div class="hero">
        <h1>Visual Insights</h1>
        <p>Understanding what the model learned in a simple way.</p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # 1️⃣ FEATURE IMPORTANCE BAR
    # ===============================
    importances = model.named_steps["model"].feature_importances_

    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#6EE7B7"],  # soft green
        title="Feature Importance"
    )

    fig.update_layout(
        # plot_bgcolor="white",
        # paper_bgcolor="white",
        xaxis=dict(title="Importance", tickfont=dict(color="#FFFFFF", size=12)),
        yaxis=dict(title="Feature", tickfont=dict(color="#FFFFFF", size=12))
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # 2️⃣ AGE vs BMI SCATTER
    # ===============================
    sample_df = df.sample(800)  # sample to keep plot clean

    fig2 = px.scatter(
        sample_df,
        x="age",
        y="bmi",
        color="cardio",
        color_discrete_map={0: "#34D399", 1: "#EF4444"},
        labels={"cardio": "Heart Risk"},
        title="Age vs BMI vs Risk"
    )

    fig2.update_layout(
        # plot_bgcolor="white",
        # paper_bgcolor="white",
        xaxis=dict(title="Age", tickfont=dict(color="#FFFFFF", size=12)),
        yaxis=dict(title="BMI", tickfont=dict(color="#FFFFFF", size=12))
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # 3️⃣ MODEL CONFIDENCE HISTOGRAM
    # ===============================
    probs = model.predict_proba(df[features])[:, 1] * 100

    fig3 = px.histogram(
        probs,
        nbins=30,
        color_discrete_sequence=["#10B981"],
        title="Model Predicted Risk Distribution (%)"
    )

    fig3.update_layout(
        # plot_bgcolor="white",
        # paper_bgcolor="white",
        xaxis=dict(title="Predicted Risk (%)", tickfont=dict(color="#FFFFFF", size=12)),
        yaxis=dict(title="Number of Patients", tickfont=dict(color="#FFFFFF", size=12))
    )

    st.plotly_chart(fig3, use_container_width=True)


# ===============================
# ABOUT
# ===============================
else:
    st.markdown("""
    <div class="hero">
        <h1>About This Project</h1>
        <p>A designer-first approach to machine learning systems.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **HeartSafe AI** blends:
    - Machine learning accuracy  
    - Human-centered design  
    - Visual explainability  

    This project demonstrates how ML systems can be **understandable, calm, and responsible** — not just accurate.

    ⚠️ Educational & research demonstration only.
    """)
if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")

   


