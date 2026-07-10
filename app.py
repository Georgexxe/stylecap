"""Streamlit demo for the StyleCap production pipeline."""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from src import config, llm, pipeline

st.set_page_config(page_title="StyleCap", page_icon="SC", layout="wide")
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background: #fcfcfa;
        color: #17191c;
    }
    [data-testid="stSidebar"] {
        background: #e9eef0;
    }
    [data-testid="stHeader"] {
        background: rgba(252, 252, 250, 0.94);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p,
    .stApp label, [data-testid="stSidebar"] * {
        color: #17191c;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("StyleCap")
st.caption("Grounded four-style video captioning with Gemma")

with st.sidebar:
    st.subheader("Run mode")
    configured = all(
        [
            os.environ.get("FIREWORKS_API_KEY"),
            config.PERCEPTION_MODEL,
            config.STYLE_MODEL,
            config.JUDGE_MODELS,
        ]
    )
    live_mode = st.toggle(
        "Live Gemma inference",
        value=configured,
        disabled=not configured,
    )
    mock_mode = not live_mode
    st.caption("Live generation ready" if configured else "Recorded benchmark mode")

gallery_tab, upload_tab = st.tabs(["Live benchmark", "Try your clip"])

with gallery_tab:
    gallery_path = Path(__file__).parent / "examples" / "demo_gallery.json"
    gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
    selected_title = st.selectbox("Official evaluation clip", [item["title"] for item in gallery])
    selected = next(item for item in gallery if item["title"] == selected_title)

    video_col, facts_col = st.columns([3, 2], gap="large")
    with video_col:
        st.video(selected["video_url"])
    with facts_col:
        st.subheader("Grounding facts")
        st.write(selected["facts"])

    left, right = st.columns(2, gap="large")
    for index, (style, caption) in enumerate(selected["captions"].items()):
        target = left if index % 2 == 0 else right
        with target:
            st.subheader(style.replace("_", " ").title())
            st.write(caption)

with upload_tab:
    uploaded = st.file_uploader(
        "Video clip",
        type=["mp4", "mov", "mkv", "webm"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        st.video(uploaded)
        run = st.button(
            "Generate captions",
            type="primary",
            icon=":material/auto_awesome:",
            use_container_width=True,
        )
        if run:
            llm.MOCK = mock_mode
            if not mock_mode:
                try:
                    config.validate_runtime()
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

            suffix = Path(uploaded.name).suffix.lower() or ".mp4"
            with tempfile.TemporaryDirectory() as directory:
                clip_path = Path(directory) / f"upload{suffix}"
                clip_path.write_bytes(uploaded.getbuffer())
                try:
                    with st.status("Processing video", expanded=True) as status:
                        st.write("Extracting representative frames")
                        result = pipeline.process_clip(str(clip_path))
                        status.update(label="Captions ready", state="complete", expanded=False)
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    st.stop()

            caption_tab, facts_tab = st.tabs(["Captions", "Grounding facts"])
            with caption_tab:
                for caption in result.captions:
                    st.subheader(caption.style.replace("_", " ").title())
                    st.write(caption.caption)
            with facts_tab:
                st.json(result.facts.model_dump(mode="json"))
