import streamlit as st

st.title("🫁 Pneumonia Classification from Chest X-Rays")
st.caption("Applied Machine Learning (WBAI065-05) · University of Groningen")

st.markdown(
    """
This dashboard is the interactive front-end for a chest X-ray classification
system. It walks through the **entire machine learning lifecycle** — from raw
data all the way to live predictions — and talks to a FastAPI backend that
serves the data pipeline, training jobs, and trained models.
"""
)

st.subheader("The task")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
We classify chest X-ray images into **three clinically meaningful classes**:

- **NORMAL:** No signs of pneumonia.
- **BACTERIA:** Bacterial pneumonia.
- **VIRUS:** Viral pneumonia.

The dataset is the
[Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)
collection from Kaggle. Because misdiagnosis carries real clinical cost, the
system also reports **uncertainty quantification (UQ)** so a prediction can be
flagged as *unreliable* rather than silently trusted.
"""
    )
with col2:
    st.info(
        "**Three models** are supported:\n\n"
        "- **CNN:** a compact baseline convolutional network.\n"
        "- **ResNet** — a fine-tuned ResNet-18.\n"
        "- **LightGBM** — gradient-boosted trees on HOG features.",
        icon="📦",
    )

st.subheader("How to use this dashboard")
st.markdown(
    """
| Page | What you can do |
| --- | --- |
| **Data & Preprocessing** | Download the dataset and run the preprocessing pipeline with live progress, then compare images *before* and *after* preprocessing. |
| **Model Training** | Configure a model (with model-specific hyperparameters), launch training, watch live metrics, and review the final evaluation plots. |
| **Showcase** | Upload your own X-ray, see how it is preprocessed, and get a classification with calibrated scores and a clear reliability verdict. |
"""
)

st.divider()
st.caption(
    "Tip: the sidebar shows whether the backend API is currently reachable. "
    "If it is offline, start it with `uvicorn src.api.router:app --reload`."
)
