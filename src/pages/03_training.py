import io
import time

import pandas as pd
import streamlit as st
from PIL import Image

from src.utils import api_client

st.title("⚙️ Model Training")
st.markdown(
    "Configure a model, launch a training run, and watch the validation "
    "metrics evolve live before reviewing the final evaluation plots."
)

if not api_client.is_api_running():
    st.error(
        "The backend API is not reachable. Start it before using this page."
    )
    st.stop()

# Per-model sensible defaults.
MODEL_DEFAULTS = {
    "cnn": {"epochs": 20, "learning_rate": 0.0001},
    "resnet": {"epochs": 10, "learning_rate": 0.0001},
    "lgbm": {"epochs": 100, "learning_rate": 0.1},
}
IS_PYTORCH = {"cnn": True, "resnet": True, "lgbm": False}

PLOT_TITLES = {
    "training_history": "Training History",
    "confusion_matrix": "Confusion Matrix (Test Set)",
    "reliability_diagram": "Reliability Diagram (Test Set)",
    "selective_prediction": "Selective Prediction (Test Set)",
}

# region Model selection & status
model_name = st.selectbox(
    "Model architecture",
    options=["cnn", "resnet", "lgbm"],
    format_func=lambda m: {
        "cnn": "CNN",
        "resnet": "ResNet-18",
        "lgbm": "LightGBM",
    }[m],
)

status_resp = api_client.get_model_status(model_name)
if status_resp.status_code == 200:
    info = status_resp.json()
    badge = "Trained" if info.get("status") == "completed" else "Not Trained"
    st.caption(f"Current status: **{badge}**")
# endregion

st.divider()

# region Configuration
st.header("Configuration")
defaults = MODEL_DEFAULTS[model_name]
is_torch = IS_PYTORCH[model_name]

c1, c2, c3 = st.columns(3)
with c1:
    epochs = st.number_input(
        "Epochs" if is_torch else "Boosting rounds",
        min_value=1,
        max_value=500,
        value=defaults["epochs"],
        step=1,
    )
    patience = st.number_input(
        "Early-stopping patience",
        min_value=1,
        max_value=50,
        value=3,
        step=1,
    )
with c2:
    learning_rate = st.number_input(
        "Learning rate",
        min_value=1e-5,
        max_value=1.0,
        value=float(defaults["learning_rate"]),
        step=1e-4,
        format="%.5f",
    )
with c3:
    if is_torch:
        batch_size = st.number_input(
            "Batch size", min_value=1, max_value=256, value=32, step=1
        )
        weight_decay = st.number_input(
            "Weight decay (L2)",
            min_value=0.0,
            max_value=1.0,
            value=0.0001,
            step=0.0001,
            format="%.5f",
        )
        num_leaves, max_depth = 31, -1
    else:
        num_leaves = st.number_input(
            "Number of leaves", min_value=2, max_value=255, value=31, step=1
        )
        max_depth = st.number_input(
            "Max tree depth (-1 = unlimited)",
            min_value=-1,
            max_value=64,
            value=-1,
            step=1,
        )
        batch_size, weight_decay = 32, 0.0

if is_torch:
    st.caption(
        "🌳 Tree parameters are hidden — they don't apply to neural networks."
    )
else:
    st.caption(
        "🧠 Batch size & weight decay are hidden — they don't apply to LightGBM."
    )

with st.container(border=True):
    ucol1, ucol2 = st.columns([4, 1], vertical_alignment="center")
    with ucol1:
        st.markdown(
            "**🎯 Uncertainty quantification**  \n"
            "Measure *how much to trust* each prediction: calibration (ECE), "
            "predictive entropy, Brier score and NLL on the validation set."
        )
    with ucol2:
        enable_uq = st.toggle("Enable", value=True, key="enable_uq")

start_training = st.button("🚀 Start training", type="primary")
# endregion

# region Live Training
if start_training:
    payload = {
        "model_name": model_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "patience": int(patience),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "weight_decay": float(weight_decay),
        "enable_uq": bool(enable_uq),
    }
    resp = api_client.train_model(payload)
    if resp.status_code != 200:
        st.error(f"Failed to start training: {resp.text}")
    else:
        st.success("Training started. Tracking live progress below.")
        bar = st.progress(0.0, text="Initialising...")
        loss_area = st.empty()
        metric_area = st.empty()

        while True:
            try:
                data = api_client.get_training_status(model_name).json()
            except Exception as e:
                st.error(f"Could not read training status: {e}")
                break

            status = data.get("status", "idle")
            progress = data.get("progress")
            message = data.get("message") or "Training..."
            history = data.get("history", [])

            bar.progress(
                float(progress) if progress is not None else 0.0, text=message
            )

            if history:
                hist_df = pd.DataFrame(history).set_index("epoch")
                loss_cols = [
                    c for c in ["train_loss", "eval_loss"] if c in hist_df
                ]
                if loss_cols:
                    with loss_area.container():
                        st.markdown("**Loss**")
                        st.line_chart(hist_df[loss_cols])
                metric_cols = [
                    c
                    for c in ["macro_f1", "precision", "recall", "ece"]
                    if c in hist_df
                ]
                if metric_cols:
                    with metric_area.container():
                        st.markdown("**Validation metrics**")
                        st.line_chart(hist_df[metric_cols])

            if status in ("completed", "failed"):
                if status == "completed":
                    bar.progress(1.0, text=message)
                    st.success(message)
                else:
                    st.error(data.get("error") or "Training failed.")
                break

            time.sleep(1.5)
# endregion

st.divider()

# region Results
st.header("Results")
st.caption(f"Showing the latest saved results for **{model_name}**.")

metrics_resp = api_client.get_metrics(model_name)
if metrics_resp.status_code == 200:
    m = metrics_resp.json()
    mc = st.columns(4)
    mc[0].metric(
        "Macro F1",
        f"{m['macro_f1']:.3f}",
        help="Harmonic mean of precision and recall, averaged equally across "
        "classes. Robust to class imbalance - higher is better (max 1.0).",
    )
    mc[1].metric(
        "Precision",
        f"{m['precision']:.3f}",
        help="Of all the cases the model flagged for a class, the fraction "
        "that were actually that class. High precision = few false alarms.",
    )
    mc[2].metric(
        "Recall",
        f"{m['recall']:.3f}",
        help="Of all the true cases of a class, the fraction the model caught. "
        "High recall = few missed cases (critical in a clinical setting).",
    )
    mc[3].metric(
        "Loss",
        f"{m['loss']:.3f}" if m.get("loss") is not None else "—",
        help="Cross-entropy on the test set — how far predicted probabilities "
        "sit from the truth. Lower is better.",
    )

    if any(
        m.get(k) is not None
        for k in ("ece", "predictive_entropy", "brier_score", "nll")
    ):
        uc = st.columns(4)
        uc[0].metric(
            "ECE",
            f"{m['ece']:.3f}" if m.get("ece") is not None else "—",
            help="Expected Calibration Error: the gap between the model's stated "
            "confidence and its real accuracy. 0 = perfectly calibrated.",
        )
        uc[1].metric(
            "Pred. entropy",
            f"{m['predictive_entropy']:.3f}"
            if m.get("predictive_entropy") is not None
            else "—",
            help="Average predictive entropy (nats) over the test set. Higher "
            "means the model is less certain about its predictions overall.",
        )
        uc[2].metric(
            "Brier",
            f"{m['brier_score']:.3f}"
            if m.get("brier_score") is not None
            else "—",
            help="Mean squared error between predicted probabilities and the "
            "true labels. Rewards both accuracy and calibration — lower is better.",
        )
        uc[3].metric(
            "NLL",
            f"{m['nll']:.3f}" if m.get("nll") is not None else "—",
            help="Negative log-likelihood of the true classes. Punishes confident "
            "mistakes heavily — lower means better-calibrated probabilities.",
        )
else:
    st.info("No metrics available yet — train this model to generate them.")

plots_resp = api_client.list_plots(model_name)
available = (
    plots_resp.json().get("plots", []) if plots_resp.status_code == 200 else []
)

if available:
    st.subheader("Evaluation plots")
    cols = st.columns(2)
    for i, plot in enumerate(available):
        img_resp = api_client.get_plot(model_name, plot)
        if img_resp.status_code == 200:
            with cols[i % 2]:
                st.markdown(f"**{PLOT_TITLES.get(plot, plot)}**")
                st.image(
                    Image.open(io.BytesIO(img_resp.content)),
                    use_container_width=True,
                )
else:
    st.caption("Evaluation plots will appear here once training has completed.")
# endregion
