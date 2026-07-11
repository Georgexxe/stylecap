"""Streamlit demo for the StyleCap production pipeline."""

import html
import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from src import config, ingest, llm, pipeline

st.set_page_config(page_title="StyleCap", page_icon="SC", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ink: #16191d;
        --muted: #667078;
        --paper: #f7f8f6;
        --panel: #ffffff;
        --line: #dfe3e5;
        --red: #cf493c;
        --blue: #2563a8;
        --green: #18745d;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--paper);
        color: var(--ink);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #edf0f1;
        border-right: 1px solid #d8dddf;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 246, 0.94);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p,
    .stApp label, [data-testid="stSidebar"] * {
        color: var(--ink);
    }
    h1, h2, h3, p, label, button { letter-spacing: 0 !important; }
    h1 { font-size: 2.15rem !important; margin-bottom: 0.1rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 178px;
        border: 2px dashed #aeb8bd;
        border-radius: 6px;
        background: var(--panel);
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--red) !important;
        border-color: var(--red) !important;
        border-radius: 6px !important;
    }
    .stButton > button[kind="primary"],
    .stDownloadButton > button {
        background: var(--red);
        border-color: var(--red);
        color: #ffffff;
        border-radius: 6px;
        min-height: 2.8rem;
    }
    .stButton > button[kind="primary"] * ,
    .stDownloadButton > button * { color: #ffffff !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.4rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding-left: 0;
        padding-right: 0;
    }
    .status-line {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.35rem 0 1rem 0;
        color: var(--green);
        font-size: 0.88rem;
        font-weight: 650;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
    }
    .source-summary {
        border-top: 3px solid var(--blue);
        background: var(--panel);
        border-radius: 6px;
        padding: 1.05rem 1.1rem;
        border-left: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        min-height: 178px;
    }
    .source-summary strong { font-size: 1rem; }
    .source-summary p { color: var(--muted); margin: 0.55rem 0 0 0; }
    .caption-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 3px solid var(--blue);
        border-radius: 6px;
        padding: 1rem 1.05rem;
        min-height: 142px;
        margin-bottom: 1rem;
    }
    .caption-card.sarcastic { border-top-color: var(--red); }
    .caption-card.humorous-tech { border-top-color: var(--green); }
    .caption-card.humorous-non-tech { border-top-color: #8a6726; }
    .caption-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 750;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .caption-card p { margin: 0; line-height: 1.55; }
    .fact-strip {
        background: #edf3f5;
        border-left: 3px solid var(--blue);
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1.25rem 0;
        border-radius: 0 6px 6px 0;
    }
    .fact-strip p { margin: 0; line-height: 1.5; }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 0.85rem 2rem; }
        h1 { font-size: 1.75rem !important; }
        .caption-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_name(style: str) -> str:
    """Return a concise display label for an evaluator style key."""
    labels = {
        "formal": "Formal",
        "sarcastic": "Sarcastic",
        "humorous_tech": "Humorous · Tech",
        "humorous_non_tech": "Humorous · Everyday",
    }
    return labels.get(style, style.replace("_", " ").title())


def render_caption_card(style: str, caption: str) -> None:
    """Render one safe, compact caption card."""
    css_style = style.replace("_", "-")
    st.markdown(
        f"""
        <div class="caption-card {css_style}">
            <div class="caption-label">{html.escape(style_name(style))}</div>
            <p>{html.escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


configured = all(
    [
        os.environ.get("FIREWORKS_API_KEY"),
        config.PERCEPTION_MODEL,
        config.STYLE_MODEL,
        config.JUDGE_MODELS,
    ]
)

st.title("StyleCap")
st.caption("Four voices. One grounded view of the video.")
st.markdown(
    '<div class="status-line"><span class="status-dot"></span>'
    + ("Live Gemma inference ready" if configured else "Recorded benchmark mode")
    + "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Inference")
    live_mode = st.toggle("Live Gemma", value=configured, disabled=not configured)
    mock_mode = not live_mode
    st.caption("Gemma 4 · Fireworks AI")
    st.divider()
    st.caption("AMD Developer Hackathon: ACT II")
    st.caption("Track 2 · Video Captioning")

create_tab, benchmark_tab = st.tabs(["Create captions", "Benchmark clips"])

with create_tab:
    st.subheader("Choose a video")
    source_mode = st.radio(
        "Video source",
        ["Upload video", "Paste video link"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded = None
    source_url = ""
    if source_mode == "Upload video":
        uploaded = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "mkv", "webm"],
            accept_multiple_files=False,
            help="MP4, MOV, MKV, or WebM up to 512 MB",
        )
    else:
        source_url = st.text_input(
            "Direct video link",
            placeholder="https://example.com/video.mp4",
        ).strip()
        st.caption("Public direct links to MP4, MOV, MKV, or WebM files")

    has_source = uploaded is not None or bool(source_url)
    source_key = (
        f"upload:{uploaded.name}:{uploaded.size}"
        if uploaded is not None
        else f"url:{source_url}"
    )
    if st.session_state.get("stylecap_source_key") != source_key:
        st.session_state.pop("stylecap_result", None)
        st.session_state.pop("stylecap_elapsed", None)
        st.session_state["stylecap_source_key"] = source_key

    if has_source:
        preview_col, action_col = st.columns([3, 2], gap="large")
        with preview_col:
            st.video(uploaded if uploaded is not None else source_url)
        with action_col:
            source_label = uploaded.name if uploaded is not None else source_url
            st.markdown(
                f"""
                <div class="source-summary">
                    <strong>Ready to analyze</strong>
                    <p>{html.escape(source_label)}</p>
                    <p>Observe · Generate · Verify · Select</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            run = st.button(
                "Generate four captions",
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

            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as directory:
                work_dir = Path(directory)
                try:
                    with st.status("Watching the video", expanded=True) as status:
                        if uploaded is not None:
                            suffix = Path(uploaded.name).suffix.lower() or ".mp4"
                            clip_path = work_dir / f"upload{suffix}"
                            clip_path.write_bytes(uploaded.getbuffer())
                            st.write("Video received")
                        else:
                            st.write("Fetching the linked video")
                            clip_path = ingest.download_video(source_url, work_dir)

                        progress_labels = {
                            "sample": "Sampling representative frames",
                            "perceive": "Building grounded visual facts",
                            "style": "Writing four distinct voices",
                            "select": "Final captions verified and selected",
                        }

                        def report(stage: str) -> None:
                            st.write(progress_labels[stage])

                        result = pipeline.process_clip(str(clip_path), progress=report)
                        elapsed = time.perf_counter() - started
                        status.update(
                            label=f"Four captions ready in {elapsed:.1f}s",
                            state="complete",
                            expanded=False,
                        )
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    st.stop()

            st.session_state["stylecap_result"] = result
            st.session_state["stylecap_elapsed"] = elapsed

    result = st.session_state.get("stylecap_result")
    if result is not None:
        st.divider()
        st.subheader("Grounded result")
        facts = result.facts
        action_summary = "; ".join(facts.actions[:3]) or "No action detected"
        st.markdown(
            f"""
            <div class="fact-strip">
                <p><strong>{html.escape(facts.setting)}</strong> ·
                {html.escape(action_summary)} · {html.escape(facts.mood)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")
        for index, caption in enumerate(result.captions):
            with (left if index % 2 == 0 else right):
                render_caption_card(caption.style, caption.caption)

        payload = {
            "facts": facts.model_dump(mode="json"),
            "captions": {
                caption.style: caption.caption for caption in result.captions
            },
        }
        download_col, detail_col = st.columns([1, 2], gap="large")
        with download_col:
            st.download_button(
                "Download result",
                json.dumps(payload, indent=2),
                file_name="stylecap-result.json",
                mime="application/json",
                icon=":material/download:",
                use_container_width=True,
            )
        with detail_col:
            with st.expander("Grounding details"):
                st.json(facts.model_dump(mode="json"))

with benchmark_tab:
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
        with (left if index % 2 == 0 else right):
            render_caption_card(style, caption)
