import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="DeepSea AI | Underwater Fish Classification",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# --------------------------------------------------
# TITLE SECTION
# --------------------------------------------------
st.title("DeepSea AI")
st.subheader("Real-time Underwater Fish Classification using YOLOv8")
st.markdown("---")

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
tab_demo, tab_research, tab_contact = st.tabs([
    "🐟 Model Demo",
    "📄 Research",
    "📬 Contact"
])

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# --------------------------------------------------
# TAB 1: MODEL DEMO
# --------------------------------------------------
with tab_demo:
    st.markdown("## 🔍 Try the Model")

    st.info("Upload an underwater image and detect fish species using a YOLOv8 deep learning model.")

    st.markdown("### ⚙️ Inference Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    uploaded_file = st.file_uploader(
        "📂 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image = Image.open(uploaded_file)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(tmp.name)
            image_path = tmp.name

        results = model.predict(image_path, conf=conf_threshold)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🧠 Detection Result")
            for r in results:
                st.image(r.plot(), use_container_width=True)

                if r.boxes is not None and len(r.boxes) > 0:
                    st.markdown("### 📊 Detected Species")
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        species = r.names[cls_id]

                        st.markdown(
                            f"- **{species.title()}** — `{confidence:.2%}`"
                        )
                else:
                    st.warning("No fish detected at this confidence level.")

# --------------------------------------------------
# TAB 2: RESEARCH
# --------------------------------------------------
with tab_research:
    st.markdown("## 📄 Research & Publication")

    st.markdown("### 🎯 Research Objective")
    st.write("This project investigates automated underwater fish classification using real-time object detection to support marine biodiversity monitoring and intelligent aquaculture systems.")

    st.markdown("### 🧪 Methodology")
    st.markdown("""
    - YOLOv8 object detection framework
    - Custom-labeled underwater fish dataset
    - Confidence-based prediction filtering
    """)

    st.markdown("### 🚀 Future Work")
    st.markdown("""
    - IEEE research paper submission
    - Video & live-stream inference
    - Species population analytics
    """)

# --------------------------------------------------
# TAB 3: CONTACT
# --------------------------------------------------
with tab_contact:
    st.markdown("## 📬 Contact")

    st.write("**Developer:** Nikith")
    st.write("**Specialization:** Computer Vision • Deep Learning • AI Research")
    st.write("**Email:** your_email@example.com")
    st.write("**GitHub:** https://github.com/yourusername")
    st.write("**LinkedIn:** https://linkedin.com/in/yourprofile")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("Built with YOLOv8 • Streamlit • Deep Learning")
st.markdown("© 2026 DeepSea AI")
