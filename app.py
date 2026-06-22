import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Repo Popularity Predictor", page_icon="🔥", layout="wide")

# ── custom css ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding: 2rem 3rem; }
    h1 { color: #58a6ff !important; font-size: 2.5rem !important; }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        background: #238636;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        width: 100%;
    }
    .stButton>button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ── train model ────────────────────────────────────────
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
            "stars":    r["stargazers_count"],
            "forks":    r["forks_count"],
            "issues":   r["open_issues_count"],
            "size_kb":  r["size"],
            "language": r["language"] or "Unknown",
            "topics":   len(r.get("topics", [])),
            "desc_len": len(r.get("description") or ""),
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
    return model, sc, le, df

# ── header ─────────────────────────────────────────────
st.title("🔥 Repo Popularity Predictor")
st.caption("Predict if a GitHub repo will go viral using ML")
st.divider()

with st.spinner("⚙️ Training model on live GitHub data..."):
    model, scaler, le, df = train_model()

# ── metrics row ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Repos",   f"{len(df):,}")
col2.metric("⭐ Avg Stars",     f"{int(df['stars'].mean()):,}")
col3.metric("🍴 Avg Forks",     f"{int(df['forks'].mean()):,}")
col4.metric("🎯 Threshold",     f"{int(df['stars'].median()):,} stars")

st.divider()

# ── inputs ─────────────────────────────────────────────
st.subheader("📝 Enter Repo Details")
col1, col2, col3 = st.columns(3)

with col1:
    forks    = st.number_input("🍴 Forks",       min_value=0, value=50)
    issues   = st.number_input("🐛 Open Issues", min_value=0, value=10)

with col2:
    size_kb  = st.number_input("📁 Size (KB)",   min_value=0, value=1000)
    topics   = st.number_input("🏷️ Topics",      min_value=0, value=3)

with col3:
    desc_len = st.number_input("📝 Desc Length", min_value=0, value=80)
    language = st.selectbox("💻 Language",
        ["Python","JavaScript","Java","Go","Unknown"])

st.divider()

# ── predict ────────────────────────────────────────────
if st.button("🚀 Predict Popularity"):
    lang_enc = le.transform([language])[0] if language in le.classes_ else 0
    X = np.array([[forks, issues, size_kb, lang_enc, topics, desc_len]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1] * 100

    st.divider()
    if pred == 1:
        st.success(f"🔥 Popular Repo! — {prob:.1f}% confidence")
        st.balloons()
    else:
        st.error(f"📦 Not Popular — {prob:.1f}% confidence")

    # gauge bar
    st.subheader("Confidence Score")
    st.progress(int(prob))
    st.caption(f"{prob:.1f}% chance of being popular")
