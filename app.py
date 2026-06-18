import os
import streamlit as st
import pickle
import numpy as np

# ── load model ─────────────────────────────────────────
if not os.path.exists("model.pkl"):
    st.error("model.pkl not found — run training notebook first")
    st.stop()

model  = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ── page config ────────────────────────────────────────
st.set_page_config(page_title="GitHub Popularity Predictor", page_icon="🔥")

# ── title ──────────────────────────────────────────────
st.title("🔥 GitHub Repo Popularity Predictor")
st.write("Fill in repo details to predict if it will be popular!")

# ── inputs ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    forks        = st.number_input("Forks",              min_value=0, value=50)
    issues       = st.number_input("Open Issues",        min_value=0, value=10)
    size_kb      = st.number_input("Size (KB)",          min_value=0, value=1000)
    age_days     = st.number_input("Age (days)",         min_value=0, value=365)

with col2:
    topics_count    = st.number_input("Topics Count",       min_value=0, value=3)
    description_len = st.number_input("Description Length", min_value=0, value=80)
    days_since_update = st.number_input("Days Since Update",min_value=0, value=10)
    language = st.selectbox("Language",
        ["Python","JavaScript","Java","Go","Rust","TypeScript","Unknown"])

# ── encode language ────────────────────────────────────
lang_map = {"Python":0,"JavaScript":1,"Java":2,"Go":3,
            "Rust":4,"TypeScript":5,"Unknown":6}
lang_enc = lang_map[language]

# ── predict ────────────────────────────────────────────
if st.button("Predict 🚀"):
    X = np.array([[forks, issues, size_kb, lang_enc,
                   topics_count, description_len,
                   age_days, days_since_update]])
    X_scaled = scaler.transform(X)
    pred     = model.predict(X_scaled)[0]
    prob     = model.predict_proba(X_scaled)[0][1] * 100

    st.divider()
    if pred == 1:
        st.success(f"🔥 Popular Repo! — {prob:.1f}% confidence")
    else:
        st.error(f"📦 Not Popular — {prob:.1f}% confidence")
