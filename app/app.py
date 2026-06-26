"""
app.py
======
Streamlit prediction UI for the T5 dialogue summarizer.

Users can paste a dialogue/long text OR upload a .txt file, then the
fine-tuned model produces and displays a summary.

Run locally:
    streamlit run app/app.py
Run via Docker:
    docker build -t t5-summarizer-app:1.0 .
    docker run -p 8501:8501 t5-summarizer-app:1.0
"""
import os
import sys

import streamlit as st

# Make src/ importable regardless of where streamlit is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import load_config, resolve_model_dir, summarize  # noqa: E402

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

st.set_page_config(page_title="T5 Dialogue Summarizer", page_icon="📝", layout="wide")


@st.cache_resource
def get_config():
    return load_config(CONFIG_PATH)


config = get_config()
model_dir = resolve_model_dir(config)

st.title("📝 T5 Dialogue Summarization")
st.caption(
    "Fine-tuned **t5-small** on the SAMSum dataset. Paste a conversation or "
    "upload a `.txt` file to generate an abstractive summary."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Model:** `{os.path.basename(model_dir)}`")
    st.write(f"**Base checkpoint:** `{config['model']['checkpoint']}`")
    max_length = st.slider("Max summary length", 16, 128,
                           config["model"]["max_target_length"], 8)
    num_beams = st.slider("Beam search width", 1, 8, 4, 1)
    st.markdown("---")
    st.markdown(
        "**How to use**\n\n"
        "1. Paste text or upload a `.txt` file.\n"
        "2. Click **Summarize**.\n"
        "3. View the generated summary."
    )

SAMPLE = (
    "Amanda: I baked cookies. Do you want some?\n"
    "Jerry: Sure!\n"
    "Amanda: I'll bring you tomorrow :-)"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input dialogue")
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    default_text = SAMPLE
    if uploaded is not None:
        default_text = uploaded.read().decode("utf-8", errors="ignore")
    dialogue = st.text_area("Or paste your dialogue here", value=default_text, height=300)
    run = st.button("Summarize", type="primary", use_container_width=True)

with col2:
    st.subheader("Generated summary")
    if run:
        if not dialogue.strip():
            st.warning("Please provide some dialogue text first.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize(
                    dialogue,
                    config=config,
                    model_dir=model_dir,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            st.success("Done!")
            st.markdown(f"> {summary}")
            st.markdown("---")
            st.metric("Input words", len(dialogue.split()))
            st.metric("Summary words", len(summary.split()))
    else:
        st.info("The summary will appear here after you click **Summarize**.")

st.markdown("---")
st.caption("Built with HuggingFace Transformers, MLflow, Streamlit & Docker.")
