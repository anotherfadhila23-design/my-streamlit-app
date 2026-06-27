
import streamlit as st
import torch
import numpy as np
import re
import joblib
import time
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

# ════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindScan — Deteksi Depresi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0D1117;
    color: #E6EDF3;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* ── HERO SECTION ── */
.hero-container {
    background: linear-gradient(135deg, #161B22 0%, #0D1117 50%, #161B22 100%);
    border: 1px solid #21262D;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(79,142,247,0.06) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(255,79,107,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.3);
    color: #4F8EF7;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: #F0F6FF;
    margin: 0 0 0.6rem 0;
}
.hero-title span {
    background: linear-gradient(135deg, #4F8EF7, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #8B949E;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
}
.stat-item {
    display: flex;
    flex-direction: column;
}
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4F8EF7;
}
.stat-label {
    font-size: 0.75rem;
    color: #8B949E;
    margin-top: 2px;
}

/* ── Cards ── */
.card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #30363D; }
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}

/* ── Result Cards ── */
.result-depresi {
    background: linear-gradient(135deg, #1A0A0D 0%, #1C0E12 100%);
    border: 1px solid rgba(255,79,107,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-depresi::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,79,107,0.1) 0%, transparent 70%);
}
.result-sehat {
    background: linear-gradient(135deg, #081A12 0%, #0C1F16 100%);
    border: 1px solid rgba(46,204,143,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-sehat::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(46,204,143,0.1) 0%, transparent 70%);
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    display: block;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-label-dep { color: #FF4F6B; }
.result-label-sehat { color: #2ECC8F; }
.result-sub { color: #8B949E; font-size: 0.85rem; }

/* ── Prob Bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-label { font-size: 0.8rem; color: #8B949E; width: 120px; flex-shrink: 0; }
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #21262D;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill-dep {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF4F6B, #FF8FA3);
    transition: width 0.8s ease;
}
.prob-bar-fill-sehat {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2ECC8F, #6EE7B7);
    transition: width 0.8s ease;
}
.prob-pct { font-size: 0.8rem; font-weight: 600; width: 45px; text-align: right; }

/* ── Model Badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.25);
    color: #A78BFA;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Confidence Ring ── */
.conf-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1rem 0;
}
.conf-ring {
    width: 120px;
    height: 120px;
    position: relative;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important;
}

/* ── Buttons ── */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    border: none !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #4F8EF7, #A78BFA) !important;
    color: white !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(79,142,247,0.35) !important;
}
.stButton button[kind="secondary"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
}
.stButton button[kind="secondary"]:hover {
    background: #2D333B !important;
    border-color: #4F8EF7 !important;
}

/* ── Radio ── */
.stRadio label { color: #E6EDF3 !important; }
.stRadio div[role="radio"] { accent-color: #4F8EF7; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
}

/* ── Info box ── */
.disclaimer-box {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    font-size: 0.82rem;
    color: #D4A017;
    line-height: 1.6;
}

/* ── Keyword chips ── */
.chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.5rem;
}
.chip-dep {
    background: rgba(255,79,107,0.15);
    border: 1px solid rgba(255,79,107,0.3);
    color: #FF8FA3;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.chip-safe {
    background: rgba(46,204,143,0.15);
    border: 1px solid rgba(46,204,143,0.3);
    color: #6EE7B7;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22 !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8B949E !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}

/* ── Divider ── */
hr { border-color: #21262D !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #E6EDF3 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #8B949E !important; }

/* ── Animated pulse ── */
@keyframes pulse-dep { 0%,100%{box-shadow:0 0 0 0 rgba(255,79,107,0.4)} 50%{box-shadow:0 0 0 12px rgba(255,79,107,0)} }
@keyframes pulse-sehat { 0%,100%{box-shadow:0 0 0 0 rgba(46,204,143,0.4)} 50%{box-shadow:0 0 0 12px rgba(46,204,143,0)} }
.pulse-dep { animation: pulse-dep 2s infinite; border-radius: 50%; display: inline-block; }
.pulse-sehat { animation: pulse-sehat 2s infinite; border-radius: 50%; display: inline-block; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_tfidf_models():
    tfidf = joblib.load("tfidf_model.pkl")
    lr    = joblib.load("lr_model.pkl")
    return tfidf, lr

@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained("best_roberta_model")
    model = AutoModelForSequenceClassification.from_pretrained("best_roberta_model")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)

def predict_tfidf_lr(text, tfidf, lr):
    cleaned = clean_text(text)
    vec  = tfidf.transform([cleaned])
    prob = lr.predict_proba(vec)[0]
    pred = int(lr.predict(vec)[0])
    return pred, prob, cleaned

def predict_roberta(text, tokenizer, model):
    enc = tokenizer(
        text, max_length=MAX_LEN,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs, text

DEP_KEYWORDS = [
    "hopeless","worthless","empty","sad","depressed","alone","crying","numb",
    "anxious","tired","exhausted","lonely","broken","miserable","despair",
    "darkness","suicidal","give up","no point","cant go on"
]

def highlight_keywords(text):
    words = text.lower().split()
    found = [w.strip(".,!?") for w in words if w.strip(".,!?") in DEP_KEYWORDS]
    return list(set(found))

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; color:#F0F6FF;">MindScan</div>
        <div style="font-size:0.72rem; color:#8B949E; letter-spacing:0.05em;">DEPRESSION DETECTION AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "",
        ["📊 Logistic Regression", "🤖 RoBERTa"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Performa Model</div>', unsafe_allow_html=True)

    models_perf = [
        ("Logistic Reg.", "94.2%", "94.1%"),
        ("Naive Bayes",   "91.3%", "91.0%"),
        ("Linear SVM",    "94.5%", "94.3%"),
        ("RoBERTa ★",     "97.1%", "97.0%"),
    ]
    for name, acc, f1 in models_perf:
        is_active = ("RoBERTa" in name and "RoBERTa" in model_choice) or \
                    ("Logistic" in name and "Logistic" in model_choice)
        border = "border-left: 3px solid #4F8EF7;" if is_active else "border-left: 3px solid #21262D;"
        st.markdown(f"""
        <div style="background:#0D1117;{border}padding:0.5rem 0.7rem;border-radius:0 6px 6px 0;margin-bottom:6px;">
            <div style="font-size:0.8rem;font-weight:{"600" if is_active else "400"};color:{"#F0F6FF" if is_active else "#8B949E"};">{name}</div>
            <div style="display:flex;gap:1rem;margin-top:2px;">
                <span style="font-size:0.7rem;color:#8B949E;">Acc <b style="color:{"#4F8EF7" if is_active else "#8B949E"};">{acc}</b></span>
                <span style="font-size:0.7rem;color:#8B949E;">F1 <b style="color:{"#4F8EF7" if is_active else "#8B949E"};">{f1}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);border-radius:8px;padding:0.8rem;font-size:0.75rem;color:#8B949E;line-height:1.6;">
        📡 <b style="color:#4F8EF7;">Device:</b> """ + str(DEVICE).upper() + """<br>
        📦 <b style="color:#4F8EF7;">Dataset:</b> Reddit Cleaned<br>
        🏫 <b style="color:#4F8EF7;">Institut:</b> ITS Surabaya
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ EAS Pembelajaran Mesin — ITS</div>
    <h1 class="hero-title">Deteksi <span>Indikasi Depresi</span><br>pada Teks Media Sosial</h1>
    <p class="hero-sub">Analisis teks menggunakan TF-IDF Baseline & RoBERTa Fine-tuned dengan LIME Explainability</p>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-num">97.1%</span>
            <span class="stat-label">Best Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">7.7K</span>
            <span class="stat-label">Data Training</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">4</span>
            <span class="stat-label">Model Diuji</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">2</span>
            <span class="stat-label">Kelas Label</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔍 Analisis Teks", "📊 Perbandingan Model", "ℹ️ Tentang"])

# ══════════════════════════════════════════
# TAB 1 — Deteksi
# ══════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">✏️ Input Teks</div>', unsafe_allow_html=True)

        EXAMPLES = {
            "😔 Contoh Depresi": "I feel so hopeless and empty inside. Nothing brings me joy anymore. I wake up every day feeling worthless and I just want everything to stop. I am exhausted of pretending to be okay when I am not.",
            "😊 Contoh Non-Depresi": "Had an amazing weekend! Went hiking with friends and the view was breathtaking. Feeling grateful for all the good things in life. Can not wait for our next adventure together!",
            "😐 Contoh Ambigu": "I have been really tired lately and not sleeping well. Work has been stressful and I feel like I need a break. Hoping things will get better soon.",
        }

        ex_choice = st.selectbox(
            "Muat contoh teks:",
            ["— Ketik sendiri —"] + list(EXAMPLES.keys()),
            label_visibility="visible"
        )

        default_text = EXAMPLES.get(ex_choice, "") if ex_choice != "— Ketik sendiri —" else ""
        user_input = st.text_area(
            "Teks dari media sosial (bahasa Inggris):",
            value=default_text,
            height=180,
            placeholder="Tulis atau paste teks di sini...\nContoh: I feel hopeless and empty lately...",
            label_visibility="visible"
        )

        char_count = len(user_input)
        word_count = len(user_input.split()) if user_input.strip() else 0
        st.markdown(f'<div style="text-align:right;font-size:0.72rem;color:#8B949E;margin-top:-0.5rem;">{word_count} kata · {char_count} karakter</div>', unsafe_allow_html=True)

        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        with col_b1:
            predict_btn = st.button("🔍  Analisis Teks", type="primary", use_container_width=True)
        with col_b2:
            clear_btn = st.button("🗑️  Hapus", type="secondary", use_container_width=True)
        with col_b3:
            model_tag = "RoBERTa" if "RoBERTa" in model_choice else "LR+TF-IDF"
            st.markdown(f'<div class="model-badge" style="margin-top:0.45rem;">⚡ {model_tag}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if clear_btn:
            st.rerun()

        # ── Keyword Hints ──
        if user_input.strip():
            found_kw = highlight_keywords(user_input)
            if found_kw:
                chips_html = "".join([f'<span class="chip-dep">{w}</span>' for w in found_kw])
                st.markdown(f"""
                <div class="card" style="margin-top:0;">
                    <div class="card-title">⚠️ Kata Indikator Terdeteksi</div>
                    <div class="chips-container">{chips_html}</div>
                    <div style="font-size:0.75rem;color:#8B949E;margin-top:0.7rem;">
                        Kata-kata ini sering berkorelasi dengan konten depresi pada dataset Reddit.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card" style="min-height:340px;"><div class="card-title">📊 Hasil Analisis</div>', unsafe_allow_html=True)

        if predict_btn:
            if not user_input.strip():
                st.warning("⚠️ Masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Menganalisis pola teks..."):
                    time.sleep(0.4)
                    try:
                        if "Logistic" in model_choice:
                            tfidf_m, lr_m = load_tfidf_models()
                            pred, probs, cleaned = predict_tfidf_lr(user_input, tfidf_m, lr_m)
                            model_label = "Logistic Regression + TF-IDF"
                        else:
                            tok, rob = load_roberta()
                            pred, probs, cleaned = predict_roberta(user_input, tok, rob)
                            model_label = "RoBERTa Fine-tuned"
                    except FileNotFoundError as e:
                        st.error(f"❌ File model tidak ditemukan: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {e}")
                        st.stop()

                prob_dep    = float(probs[1]) * 100
                prob_nondep = float(probs[0]) * 100
                confidence  = max(prob_dep, prob_nondep)

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-depresi">
                        <span class="result-icon">🔴</span>
                        <div class="result-label result-label-dep">Terindikasi Depresi</div>
                        <div class="result-sub">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-sehat">
                        <span class="result-icon">🟢</span>
                        <div class="result-label result-label-sehat">Non-Depresi</div>
                        <div class="result-sub">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div class="prob-row">
                        <span class="prob-label">🔴 Depresi</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill-dep" style="width:{prob_dep:.1f}%"></div>
                        </div>
                        <span class="prob-pct" style="color:#FF8FA3;">{prob_dep:.1f}%</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">🟢 Non-Depresi</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill-sehat" style="width:{prob_nondep:.1f}%"></div>
                        </div>
                        <span class="prob-pct" style="color:#6EE7B7;">{prob_nondep:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔎 Detail Preprocessing & Token"):
                    c1, c2 = st.columns(2)
                    c1.metric("Total Kata", word_count)
                    c2.metric("Kata Setelah Clean", len(cleaned.split()))
                    st.markdown("**Teks setelah preprocessing:**")
                    st.code(cleaned[:400] + ("..." if len(cleaned) > 400 else ""), language=None)

        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:200px;color:#30363D;">
                <div style="font-size:3rem;">🧠</div>
                <div style="font-size:0.85rem;margin-top:0.5rem;color:#8B949E;">Hasil analisis akan muncul di sini</div>
                <div style="font-size:0.75rem;color:#30363D;margin-top:0.3rem;">Masukkan teks lalu klik Analisis Teks</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <b>Disclaimer:</b> Aplikasi ini adalah prototipe akademik untuk keperluan EAS Pembelajaran Mesin ITS.
            Hasil prediksi <b>bukan diagnosis medis</b>. Jika Anda atau orang sekitar mengalami gejala depresi,
            segera hubungi profesional kesehatan mental atau hotline <b>119 ext 8</b> (Indonesia).
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — Perbandingan Model
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><div class="card-title">🏆 Perbandingan Performa Model</div>', unsafe_allow_html=True)

    model_data = [
        {"Model": "Logistic Regression", "Accuracy": 94.2, "F1-Score": 94.1, "AUC-ROC": 98.1, "Speed": "⚡ Sangat Cepat", "Tipe": "Baseline"},
        {"Model": "Naive Bayes",         "Accuracy": 91.3, "F1-Score": 91.0, "AUC-ROC": 97.2, "Speed": "⚡ Sangat Cepat", "Tipe": "Baseline"},
        {"Model": "Linear SVM",          "Accuracy": 94.5, "F1-Score": 94.3, "AUC-ROC": 98.3, "Speed": "✅ Cepat",        "Tipe": "Baseline"},
        {"Model": "RoBERTa Fine-tuned",  "Accuracy": 97.1, "F1-Score": 97.0, "AUC-ROC": 99.2, "Speed": "🐢 Lambat (GPU)", "Tipe": "Deep Learning"},
    ]

    for m in model_data:
        is_best = m["Model"] == "RoBERTa Fine-tuned"
        border_style = "border:1px solid rgba(79,142,247,0.4);" if is_best else "border:1px solid #21262D;"
        badge = '<span style="background:rgba(79,142,247,0.2);color:#4F8EF7;font-size:0.65rem;padding:2px 7px;border-radius:4px;font-weight:600;margin-left:8px;">BEST</span>' if is_best else ""

        acc_pct  = m["Accuracy"]
        f1_pct   = m["F1-Score"]
        auc_pct  = m["AUC-ROC"]
        bar_color = "#4F8EF7" if is_best else "#30363D"
        fill_color = "linear-gradient(90deg,#4F8EF7,#A78BFA)" if is_best else "linear-gradient(90deg,#30363D,#3D444E)"

        st.markdown(f"""
        <div style="background:#0D1117;{border_style}border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.7rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem;">
                <div>
                    <span style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:0.95rem;color:#F0F6FF;">{m["Model"]}</span>
                    {badge}
                    <span style="font-size:0.72rem;color:#8B949E;margin-left:8px;">{m["Tipe"]}</span>
                </div>
                <span style="font-size:0.8rem;color:#8B949E;">{m["Speed"]}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;">
                <div>
                    <div style="font-size:0.7rem;color:#8B949E;margin-bottom:3px;">Accuracy</div>
                    <div style="height:6px;background:#21262D;border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{acc_pct}%;background:{fill_color};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:0.78rem;font-weight:600;color:{"#4F8EF7" if is_best else "#E6EDF3"};margin-top:2px;">{acc_pct}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#8B949E;margin-bottom:3px;">F1-Score</div>
                    <div style="height:6px;background:#21262D;border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{f1_pct}%;background:{fill_color};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:0.78rem;font-weight:600;color:{"#4F8EF7" if is_best else "#E6EDF3"};margin-top:2px;">{f1_pct}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#8B949E;margin-bottom:3px;">AUC-ROC</div>
                    <div style="height:6px;background:#21262D;border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{auc_pct}%;background:{fill_color};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:0.78rem;font-weight:600;color:{"#4F8EF7" if is_best else "#E6EDF3"};margin-top:2px;">{auc_pct}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:0;"><div class="card-title">🔬 Pipeline NLP</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Load Dataset", "Depression Reddit Cleaned — 7.7K post dari Kaggle", "#4F8EF7"),
        ("02", "EDA & Visualisasi", "Distribusi kelas, word cloud, analisis panjang teks", "#A78BFA"),
        ("03", "Preprocessing", "Lowercase, hapus URL/mention, stopword removal, tokenisasi", "#2ECC8F"),
        ("04", "TF-IDF Baseline", "Logistic Regression, Naive Bayes, Linear SVM", "#F59E0B"),
        ("05", "RoBERTa Fine-tune", "3 epoch, AdamW, class weighting, warmup scheduler", "#FF4F6B"),
        ("06", "LIME Explainability", "Interpretasi lokal & global feature importance", "#4F8EF7"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc, color) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid #21262D;border-radius:10px;padding:1rem;margin-bottom:0.7rem;">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:{color};margin-bottom:0.3rem;">{num}</div>
                <div style="font-weight:600;font-size:0.85rem;color:#F0F6FF;margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.75rem;color:#8B949E;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — Tentang
# ══════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        st.markdown('<div class="card"><div class="card-title">📖 Tentang Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.9rem;color:#C9D1D9;line-height:1.8;">
            <p>Proyek ini merupakan implementasi <b style="color:#4F8EF7;">End-of-Semester Assessment (EAS)</b>
            Mata Kuliah Pembelajaran Mesin di Institut Teknologi Sepuluh Nopember (ITS) Surabaya.</p>
            <p>Sistem ini mendeteksi <b style="color:#FF4F6B;">indikasi depresi</b> pada teks media sosial
            menggunakan pendekatan bertahap: dimulai dari model baseline berbasis TF-IDF hingga
            model deep learning state-of-the-art yaitu <b style="color:#A78BFA;">RoBERTa</b>.</p>
            <p>Model dilatih menggunakan dataset <b>Depression Reddit Cleaned</b> dari Kaggle,
            yang berisi ribuan post Reddit berlabel depresi dan non-depresi.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🛠️ Teknologi</div>', unsafe_allow_html=True)
        techs = [
            ("🐍", "Python 3.10+", "Bahasa pemrograman utama"),
            ("🤗", "HuggingFace Transformers", "RoBERTa fine-tuning & tokenizer"),
            ("🔥", "PyTorch", "Framework deep learning"),
            ("📊", "Scikit-learn", "TF-IDF, baseline models, metrics"),
            ("💡", "LIME", "Model explainability & interpretability"),
            ("🎈", "Streamlit", "Web app framework"),
        ]
        for icon, name, desc in techs:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;padding:0.5rem 0;border-bottom:1px solid #21262D;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <div style="font-size:0.85rem;font-weight:600;color:#E6EDF3;">{name}</div>
                    <div style="font-size:0.75rem;color:#8B949E;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card"><div class="card-title">📞 Sumber Bantuan</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem;color:#C9D1D9;line-height:1.8;margin-bottom:1rem;">
            Jika Anda atau orang sekitar membutuhkan bantuan terkait kesehatan mental:
        </div>
        """, unsafe_allow_html=True)
        hotlines = [
            ("🇮🇩", "Into The Light Indonesia", "119 ext 8"),
            ("🇮🇩", "Yayasan Pulih", "(021) 788-42580"),
            ("🌐", "Crisis Text Line", "Text HOME to 741741"),
        ]
        for flag, name, num in hotlines:
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid #21262D;border-radius:8px;padding:0.7rem;margin-bottom:0.5rem;">
                <div style="font-size:0.8rem;color:#8B949E;">{flag} {name}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;color:#4F8EF7;">{num}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">⚠️ Disclaimer Penting</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem;color:#8B949E;line-height:1.7;">
            Aplikasi ini <b style="color:#FF4F6B;">BUKAN alat diagnosis medis</b>.
            Dibuat semata-mata untuk keperluan akademik dan penelitian NLP.<br><br>
            Jangan gunakan hasil prediksi ini sebagai dasar keputusan medis.
            Selalu konsultasikan kondisi Anda dengan profesional kesehatan mental berlisensi.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
