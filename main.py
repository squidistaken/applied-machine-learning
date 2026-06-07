from pathlib import Path
import streamlit as st
from src.utils.api_client import is_api_running

ROOT_DIR = Path(__file__).parent
LOGO_PATH = ROOT_DIR / "assets" / "rug_logo.png"

st.set_page_config(
    page_title="Pneumonia Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")

pages = [
    st.Page(
        "src/pages/01_introduction.py",
        title="Introduction",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "src/pages/02_preprocessing.py",
        title="Data & Preprocessing",
        icon="🧪",
    ),
    st.Page(
        "src/pages/03_training.py",
        title="Model Training",
        icon="⚙️",
    ),
    st.Page(
        "src/pages/04_showcase.py",
        title="Showcase",
        icon="🔬",
    ),
]

navigation = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    if is_api_running():
        st.success("API connected", icon="✅")
    else:
        st.error("API unreachable", icon="🚫")
        st.caption("Start the backend with `uvicorn src.api.router:app`.")

navigation.run()
