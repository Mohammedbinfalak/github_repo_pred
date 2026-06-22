import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── train model on startup ─────────────────────────────
@st.cache_resource
def train_model():
    HEADERS = {"Accept": "application/vnd.github+json"}
    BASE    = "https://api.github.com"

    def fetch(query, pages=3):
        repos = []
        for page in range(1, pages + 1):
            try:
                r = requests.get(f"{BASE}/search/repositories",
                    params={"q": query, "sort": "stars",
                            "order": "desc", "per_page": 100, "page": page},
                    headers=HEADERS)
                items = r.json().get("items", [])
                if not items:
                    break
                repos.extend(items)
                time.sleep(2)
            except:
                break
        return repos

    all_repos = []
    for lang in ["python", "javascript", "java", "go"]:
        all_repos.extend(fetch(f"stars:>50 language:{lang}", pages=2))

    rows = []
    for r in all_repos:
        rows.append({
            "stars":       r["stargazers_count"],
            "forks":       r["forks_count"],
            "issues":      r["open_issues_count"],
            "size_kb":     r["size"],
            "language":    r["language"] or "Unknown",
            "topics":      len(r.get("topics", [])),
            "desc_len":    len(r.get("description") or ""),
        })

    df = pd.DataFrame(rows).drop_duplicates()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df["language"] = df["language"].fillna("Unknown")

    le = LabelEncoder()
    df["lang_enc"] = le.fit_transform(df["language"])

    df["is_popular"] = (df["stars"] > df["stars"].median()).astype(int)

    X = df[["forks","issues","size_kb","lang_enc","topics","desc_len"]]
    y = df["is_popular"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, sc, le

# ── app ────────────────────────────────────────────────
st.title("🔥 Repo Popularity Predictor")
st.write("Predicts if a repo will be popular based on its features.")

with st.spinner("Training model on live data..."):
    model, scaler, le = train_model()

st.success("Model ready!")

col1, col2 = st.columns(2)
with col1:
    forks    = st.number_input("Forks",      min_value=0, value=50)
    issues   = st.number_input("Issues",     min_value=0, value=10)
    size_kb  = st.number_input("Size (KB)",  min_value=0, value=1000)
with col2:
    topics   = st.number_input("Topics",     min_value=0, value=3)
    desc_len = st.number_input("Desc length",min_value=0, value=80)
    language = st.selectbox("Language",
        ["Python","JavaScript","Java","Go","Unknown"])

if st.button("Predict 🚀"):
    lang_enc = le.transform([language])[0] if language in le.classes_ else 0
    X = np.array([[forks, issues, size_kb, lang_enc, topics, desc_len]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1] * 100
    if pred == 1:
        st.success(f"🔥 Popular! — {prob:.1f}% confidence")
    else:
        st.error(f"📦 Not Popular — {prob:.1f}% confidence")
