import io
import time

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from src.utils import api_client

st.title("🧪 Data & Preprocessing")
st.markdown(
    "Download the raw dataset, run the preprocessing pipeline, and inspect what "
    "the transformations actually do to each X-ray."
)

if not api_client.is_api_running():
    st.error(
        "The backend API is not reachable. Start it before using this page."
    )
    st.stop()


def _poll_job(job: str, label: str) -> dict:
    """Poll a data job until it finishes, animating a progress bar.

    Args:
        job (str): The job id ('download' or 'preprocess').
        label (str): A human-friendly label for the spinner text.

    Returns:
        dict: The final job-state payload.
    """
    bar = st.progress(0.0, text=f"Starting {label}...")
    tick = 0
    while True:
        try:
            data = api_client.get_data_status(job).json()
        except Exception as e:  # noqa: BLE001 - surface any polling error
            bar.empty()
            st.error(f"Could not read {label} status: {e}")
            return {"status": "failed", "error": str(e)}

        status = data.get("status", "idle")
        progress = data.get("progress")
        message = data.get("message") or f"Running {label}..."

        if progress is None:
            # Indeterminate stage: animate a capped, oscillating bar.
            tick += 1
            bar.progress(min(0.9, 0.1 + (tick % 9) * 0.1), text=message)
        else:
            bar.progress(float(progress), text=message)

        if status in ("completed", "failed"):
            bar.progress(1.0 if status == "completed" else 0.0, text=message)
            return data

        time.sleep(1.0)


# region Download
st.header("1 · Download the dataset")
dcol1, dcol2 = st.columns([1, 2])
with dcol1:
    force = st.checkbox(
        "Force re-download",
        value=False,
        help="Re-download even if the data already exists locally.",
    )
    start_download = st.button("⬇️ Download data", type="primary")

with dcol2:
    if start_download:
        resp = api_client.download_data(force=force)
        if resp.status_code != 200:
            st.error(f"Failed to start download: {resp.text}")
        else:
            result = _poll_job("download", "download")
            if result.get("status") == "completed":
                st.success(result.get("message", "Download complete."))
            else:
                st.error(result.get("error") or "Download failed.")
    else:
        st.caption(
            "The dataset is fetched from Kaggle (~1.2 GB). Kaggle credentials "
            "must be configured in `config.yaml`."
        )

# endregion

st.divider()

# region Preprocess
st.header("2 · Run preprocessing")
st.markdown(
    "Preprocessing crops to the **lung region**, resizes to **224×224**, and "
    "applies **CLAHE** contrast enhancement. The LightGBM pipeline additionally "
    "extracts HOG + statistical features."
)
pcol1, pcol2 = st.columns([1, 2])
with pcol1:
    pipeline = st.selectbox(
        "Pipeline",
        options=["all", "pytorch", "lightgbm"],
        help="`pytorch` produces image tensors; `lightgbm` extracts tabular "
        "features (and requires the pytorch pipeline to have run first).",
    )
    lgb_size = st.number_input(
        "LightGBM edge size",
        min_value=16,
        max_value=256,
        value=64,
        step=16,
        help="Downsampling size for LightGBM feature extraction.",
    )
    start_preprocess = st.button("🧹 Run preprocessing", type="primary")

with pcol2:
    if start_preprocess:
        resp = api_client.preprocess_data(
            pipeline=pipeline, lgb_size=int(lgb_size)
        )
        if resp.status_code != 200:
            st.error(f"Failed to start preprocessing: {resp.text}")
        else:
            result = _poll_job("preprocess", "preprocessing")
            if result.get("status") == "completed":
                st.success(result.get("message", "Preprocessing complete."))
            else:
                st.error(result.get("error") or "Preprocessing failed.")
    else:
        st.caption("Progress is reported per image as the pipeline runs.")

# endregion

st.divider()

# region Before / after
st.header("3 · Visualise before & after")
st.markdown(
    "Pick a sample to compare the **raw** X-ray with its **preprocessed** "
    "version, exactly as the models receive it."
)

# Split & sample picker sit directly above the image.
ccol1, ccol2 = st.columns([1, 3])
with ccol1:
    split = st.selectbox("Split", options=["train", "test"], key="viz_split")

# Fetch metadata to know how many images are available for this split.
meta_resp = api_client.get_data_metadata(
    data_type="raw", split=split, page=1, limit=1
)
if meta_resp.status_code != 200:
    st.warning("No raw data found. Download and preprocess the dataset first.")
    st.stop()

total_items = meta_resp.json().get("total_items", 0)
if total_items == 0:
    st.warning("No images available for this split yet.")
    st.stop()

with ccol2:
    if total_items > 1:
        index = st.slider(
            "Image index",
            min_value=0,
            max_value=total_items - 1,
            value=0,
            step=1,
            help=f"{total_items} images available in the '{split}' split.",
        )
    else:
        index = 0
        st.caption("Only one image available in this split.")

# Resolve the metadata (filename + label) for the chosen index.
item_resp = api_client.get_data_metadata(
    data_type="raw", split=split, page=int(index) + 1, limit=1
)
item = item_resp.json()["items"][0] if item_resp.status_code == 200 else {}
if item:
    st.markdown(
        f"**File:** `{item.get('filename', '?')}` · "
        f"**Label:** `{item.get('label', '?')}`"
    )


def _load_image(data_type: str):
    resp = api_client.get_image(data_type, split, int(index))
    if resp.status_code != 200:
        return None
    return Image.open(io.BytesIO(resp.content))


raw_img = _load_image("raw")
processed_img = _load_image("processed")

if raw_img is None:
    st.info("Raw image unavailable.")
elif processed_img is None:
    st.info("Preprocessed image unavailable — run the preprocessing pipeline.")
else:
    # Draggable on-image divider: grab the handle and wipe between the raw
    # scan (left) and the preprocessed input (right). Resize the raw image to
    # the processed canvas so the two line up pixel-for-pixel.
    raw_aligned = raw_img.convert("L").resize(processed_img.size)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        image_comparison(
            img1=raw_aligned,
            img2=processed_img.convert("L"),
            label1="Before (raw)",
            label2="After (preprocessed)",
            width=500,
            starting_position=50,
            in_memory=True,
        )
        st.caption("Drag the divider to compare raw vs. preprocessed.")
# endregion
