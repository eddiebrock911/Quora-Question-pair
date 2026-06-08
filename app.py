import streamlit as st
import pickle
import helpe

# streamlit run app.py

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Question Similarity Detector",
    page_icon="🔍",
    layout="centered",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0d0f14;
}

/* Header badge */
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* Question cards */
.q-card {
    background: #13151c;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}

.q-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

/* Result cards */
.result-dup {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.result-notdup {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.result-emoji { font-size: 36px; }

.result-title-dup   { color: #10b981; font-size: 22px; font-weight: 700; margin: 8px 0 4px; }
.result-title-notdup{ color: #ef4444; font-size: 22px; font-weight: 700; margin: 8px 0 4px; }

.result-sub { color: #64748b; font-size: 13px; font-family: 'JetBrains Mono', monospace; }

/* VS divider */
.vs-wrap {
    display: flex; align-items: center; gap: 12px;
    margin: 6px 0;
}

.vs-line { flex: 1; height: 1px; background: rgba(99,102,241,0.15); }
.vs-text {
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* Streamlit overrides */
div[data-testid="stTextArea"] textarea {
    background: #13151c !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}

.stButton > button {
    width: 100%;
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: #4338ca !important;
}

h1, h2, h3 { color: #f1f5f9 !important; }
p, label    { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">🔍 NLP · ML Model · Question Similarity</div>', unsafe_allow_html=True)
st.title("Question Pair Similarity")
st.markdown(
    "<p style='color:#64748b; font-family:JetBrains Mono,monospace; font-size:13px;'>"
    "Paste two questions to find out if they ask the same thing.</p>",
    unsafe_allow_html=True
)
st.divider()

# ─── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="q-label">Question 01</div>', unsafe_allow_html=True)
q1 = st.text_area("", placeholder="e.g. How do I learn machine learning?", key="q1", label_visibility="collapsed")

st.markdown(
    '<div class="vs-wrap"><div class="vs-line"></div>'
    '<span class="vs-text">VS</span><div class="vs-line"></div></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="q-label">Question 02</div>', unsafe_allow_html=True)
q2 = st.text_area("", placeholder="e.g. What is the best way to start with ML?", key="q2", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction ────────────────────────────────────────────────────────────────
if st.button("⚡ Run Prediction"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions before predicting.")
    else:
        with st.spinner("Analyzing question pair..."):
            query_point = helpe.query_point_creator(q1, q2)
            prediction  = model.predict(query_point)
            is_dup      = int(prediction[0]) == 1

            # Confidence (use predict_proba if available)
            try:
                prob = model.predict_proba(query_point)[0]
                confidence = round(max(prob) * 100, 1)
            except AttributeError:
                confidence = None

        if is_dup:
            st.markdown("""
            <div class="result-dup">
                <div class="result-emoji">✅</div>
                <div class="result-title-dup">Duplicate</div>
                <div class="result-sub">These questions convey the same intent.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-notdup">
                <div class="result-emoji">❌</div>
                <div class="result-title-notdup">Not Duplicate</div>
                <div class="result-sub">These questions are semantically different.</div>
            </div>""", unsafe_allow_html=True)

        if confidence is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{confidence}%")
            st.progress(confidence / 100)

        # Expandable debug info
        with st.expander("🔬 Debug Info"):
            st.code(f"Input shape : {query_point.shape}", language="python")
            st.code(f"Raw output  : {prediction}", language="python")
            if confidence:
                st.code(f"Probabilities: {prob}", language="python")