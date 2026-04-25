import streamlit as st
import pandas as pd
import pickle as pkl
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
pipe = pkl.load(open("CPP.pkl", "rb"))
ds = pd.read_csv("clean_data.csv")

# ---------------- BACKGROUND IMAGE ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("car_pic1.avif")

# ---------------- STYLING ----------------
st.markdown(f"""
<style>

/* -------- BACKGROUND -------- */
.stApp {{
    background-image: url("data:image/avif;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* REMOVE OVERLAY */
.stApp::before {{
    display: none;
}}

/* -------- TITLE -------- */
.title {{
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 12px rgba(0,0,0,0.8);
    margin-bottom: 25px;
}}

/* -------- GLASS CARD -------- */
.card {{
    background: rgba(255, 255, 255, 0.15);
    padding: 30px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 10px 35px rgba(0,0,0,0.4);
}}

/* -------- INPUT LABELS -------- */
label {{
    color: white !important;
    font-weight: 500;
}}

/* -------- BUTTON -------- */
.stButton>button {{
    background: linear-gradient(135deg, #000, #444);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}}

.stButton>button:hover {{
    transform: scale(1.05);
    background: linear-gradient(135deg, #222, #666);
}}

/* -------- RESULT -------- */
.result-box {{
    background: rgba(0, 0, 0, 0.75);
    color: #00ffcc;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    margin-top: 20px;
    backdrop-filter: blur(8px);
}}

/* -------- FOOTER -------- */
footer {{
    visibility: hidden;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Car Price Predictor</div>', unsafe_allow_html=True)

# ---------------- FORM ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

companies = sorted(ds["company"].unique())
company = st.selectbox("Company", companies)

names = sorted(ds[ds["company"] == company]["name"].unique())
name = st.selectbox("Model", names)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1995, max_value=2025)

with col2:
    kms_driven = st.number_input("KM Driven", min_value=0)

fuel_types = sorted(ds["fuel_type"].unique())
fuel_type = st.selectbox("Fuel Type", fuel_types)

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button("Predict Price")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HTML REPORT ----------------
def generate_html(company, name, year, kms, fuel, price):
    return f"""
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h1 style="text-align:center;">Car Price Report</h1>
        <hr>
        <p><b>Company:</b> {company}</p>
        <p><b>Model:</b> {name}</p>
        <p><b>Year:</b> {year}</p>
        <p><b>KMs Driven:</b> {kms}</p>
        <p><b>Fuel Type:</b> {fuel}</p>
        <hr>
        <h2 style="color:green;">Estimated Price: ₹ {price:,}</h2>
    </body>
    </html>
    """

# ---------------- SAFE PREDICT ----------------
def safe_predict(df):
    try:
        return pipe.predict(df)
    except Exception as e:
        st.error("⚠️ Prediction failed due to unseen data or model mismatch.")
        st.warning("👉 Please select valid values from dropdown (training data only).")
        st.text(str(e))
        return None

# ---------------- PREDICTION ----------------
if predict_btn:
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    result = safe_predict(input_df)

    if result is not None:
        try:
            price = int(result.item())   # ✅ FIX

            st.markdown(f"""
            <div class="result-box">
                💰 Estimated Price: ₹ {price:,}
            </div>
            """, unsafe_allow_html=True)

            html = generate_html(company, name, year, kms_driven, fuel_type, price)

            st.download_button(
                label="📄 Download Report",
                data=html,
                file_name="Car_Price_Report.html",
                mime="text/html"
            )

        except Exception as e:
            st.error("⚠️ Error converting prediction result.")
            st.text(str(e))

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center; color:white;'>Thank You❤️</p>
""", unsafe_allow_html=True)
