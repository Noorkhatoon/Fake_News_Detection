import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "YOUR_HUGGINGFACE_USERNAME/YOUR_MODEL_REPO"

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e3a5f);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.result-card {
    background: #172033;
    border: 1px solid #334155;
    padding: 24px;
    border-radius: 15px;
    text-align: center;
    margin-top: 15px;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
}

.fake {color: #fb7185;}
.real {color: #34d399;}

.info-card {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.info-card h3 {
    color: #94a3b8;
    font-size: 15px;
}

.info-card p {
    color: white;
    font-size: 25px;
    font-weight: bold;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

textarea {
    border-radius: 12px !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


try:
    tokenizer, model, device = load_model()
except Exception as e:
    st.error("Model loading failed. Please check the Hugging Face model ID.")
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📰 News Intelligence")
    st.markdown("---")

    st.subheader("About the System")
    st.write(
        "An AI-powered news classification system "
        "using a fine-tuned DistilBERT model."
    )

    st.markdown("---")
    st.subheader("Technology")
    st.write("• Python")
    st.write("• DistilBERT")
    st.write("• PyTorch")
    st.write("• Hugging Face")
    st.write("• Streamlit")

    st.markdown("---")
    st.caption("AI-based classification | Version 1.0")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>📰 Fake News Detection</h1>
    <p>Intelligent News Analysis & Classification Platform</p>
    <p>Analyze news content using Artificial Intelligence.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- NEWS INPUT ----------------

st.subheader("News Analysis")

st.write("Enter a news article below to analyze its classification.")

news = st.text_area(
    "News Article",
    height=230,
    placeholder="Paste your complete news article here..."
)

analyze = st.button("🔍 Analyze News")


# ---------------- PREDICTION ----------------

if analyze:

    if len(news.strip()) < 30:
        st.warning("Please enter a complete news article.")

    else:
        with st.spinner("Analyzing news with AI..."):

            inputs = tokenizer(
                news,
                return_tensors="pt",
                truncation=True,
                max_length=384
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(
                    outputs.logits, dim=1
                )[0]

            prediction = torch.argmax(probabilities).item()

            fake_probability = probabilities[0].item() * 100
            real_probability = probabilities[1].item() * 100

        st.markdown("---")
        st.subheader("Analysis Results")

        if prediction == 0:
            result = "FAKE NEWS DETECTED"
            result_class = "fake"
            icon = "⚠️"
        else:
            result = "REAL NEWS PREDICTED"
            result_class = "real"
            icon = "✅"

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:42px">{icon}</div>
            <div class="result-title {result_class}">{result}</div>
            <p style="color:#94a3b8">
                AI classification result
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h3>FAKE Probability</h3>
                <p>{fake_probability:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h3>REAL Probability</h3>
                <p>{real_probability:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.progress(fake_probability / 100)

        st.caption(
            "Probabilities represent the model's estimates, "
            "not proof that a news claim is true or false. "
            "Verify important information through reliable sources."
        )


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<center style='color:#94a3b8'>"
    "Fake News Detection System | Powered by DistilBERT"
    "</center>",
    unsafe_allow_html=True
)
