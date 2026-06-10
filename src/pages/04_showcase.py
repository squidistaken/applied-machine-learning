import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.utils import api_client

st.title("🔬 Showcase")
st.markdown(
    "Upload a chest X-ray to see how it is preprocessed and how the selected "
    "model classifies it — results update live as you switch models."
)

if not api_client.is_api_running():
    st.error(
        "The backend API is not reachable. Start it before using this page."
    )
    st.stop()

CLASS_HELP = {
    "NORMAL": "No signs of pneumonia.",
    "BACTERIA": "Bacterial pneumonia.",
    "VIRUS": "Viral pneumonia.",
}
MODEL_LABELS = {"cnn": "CNN", "resnet": "ResNet-18", "lgbm": "LightGBM"}


@st.cache_data(show_spinner=False)
def _preview(file_bytes: bytes, filename: str):
    """Preprocess an uploaded image. Cached on the raw bytes."""
    resp = api_client.preview_preprocessing(file_bytes, filename)
    return resp.content if resp.status_code == 200 else None


@st.cache_data(show_spinner=False)
def _predict(model_name: str, file_bytes: bytes, filename: str):
    """Run inference. Cached per (model, image) so switching is instant."""
    resp = api_client.predict_image(model_name, file_bytes, filename)
    if resp.status_code != 200:
        return None, resp.text
    return resp.json(), None


uploaded = st.file_uploader(
    "Chest X-ray image",
    type=["png", "jpg", "jpeg", "pgm"],
    help="Upload a single grayscale chest X-ray.",
)

if uploaded is None:
    st.info("⬆️ Upload an image to begin.")
    st.stop()

file_bytes = uploaded.getvalue()

st.divider()
left, right = st.columns([1, 1], gap="large")

# region Images
with left:
    st.subheader("X-ray")
    after_tab, before_tab = st.tabs(
        ["After (model input)", "Before (uploaded)"]
    )
    with after_tab:
        preview = _preview(file_bytes, uploaded.name)
        if preview is not None:
            st.image(Image.open(io.BytesIO(preview)), use_container_width=True)
        else:
            st.error("Preprocessing failed for this image.")
    with before_tab:
        st.image(Image.open(io.BytesIO(file_bytes)), use_container_width=True)
# endregion

# region Prediction
with right:
    st.subheader("📊 Results")
    model_name = st.selectbox(
        "Model",
        options=["cnn", "resnet", "lgbm"],
        format_func=lambda m: MODEL_LABELS[m],
        help="Switch models to compare verdicts — results update instantly.",
    )

    with st.spinner("Running inference (Monte-Carlo dropout uncertainty)..."):
        result, err = _predict(model_name, file_bytes, uploaded.name)

    if result is None:
        st.error(f"Prediction failed: {err}")
        st.stop()

    probabilities = result["probabilities"]
    predicted = result["predicted_class"]
    confidence = probabilities.get(predicted, 0.0)
    is_uncertain = result.get("is_uncertain", False)

    # Clear good/bad verdict.
    if is_uncertain:
        st.error(
            f"⚠️ **{predicted}** - but this prediction is **UNRELIABLE**. "
            "The predictive entropy is above the clinical safety threshold; a "
            "human expert should review this scan.",
            icon="🚨",
        )
    elif confidence >= 0.70:
        st.success(
            f"✅ **{predicted}** - confident prediction "
            f"({confidence:.0%} probability). {CLASS_HELP.get(predicted, '')}",
            icon="🩺",
        )
    else:
        st.warning(
            f"🤔 **{predicted}** - moderate confidence "
            f"({confidence:.0%} probability). Consider review.",
            icon="⚖️",
        )

    vcol1, vcol2 = st.columns(2)
    verdict = "🚨 Unreliable" if is_uncertain else "🟢 Reliable"
    vcol1.metric("Reliability verdict", verdict)
    vcol2.metric("Confidence", f"{confidence:.1%}")

    unc = result.get("uncertainty")
    if unc is not None:
        st.metric(
            "Total uncertainty (entropy)",
            f"{unc:.3f}",
            help="Predictive entropy in nats. Higher = less certain. "
            "Flagged unreliable at ≥ 0.75.",
        )
    ale = result.get("aleatoric_uncertainty")
    epi = result.get("epistemic_uncertainty")
    if ale is not None and epi is not None:
        acol1, acol2 = st.columns(2)
        acol1.metric(
            "Aleatoric", f"{ale:.3f}", help="Data (irreducible) uncertainty."
        )
        acol2.metric(
            "Epistemic", f"{epi:.3f}", help="Model (knowledge) uncertainty."
        )

    st.markdown("##### Class probabilities")
    prob_df = pd.DataFrame(
        {
            "class": list(probabilities.keys()),
            "probability": list(probabilities.values()),
        }
    ).set_index("class")
    st.bar_chart(prob_df, horizontal=True)
    st.dataframe(
        prob_df.style.format({"probability": "{:.1%}"}),
        use_container_width=True,
    )

    st.caption(f"Model used: **{result.get('model_used', model_name)}**")
# endregion
