"""
Streamlit frontend for the AI Skin Disease Detection System.
Provides image upload, prediction display, risk assessment, LLM analysis,
follow-up chat, and PDF report download.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
from typing import Any

from utils.config import Config
from utils.logger import get_logger
from utils.pdf_report import generate_pdf_report
from models.classifier import SkinClassifier
from models.risk_assessment import RiskAssessor

logger = get_logger(__name__)

# ── Page Configuration ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    .main-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #a8b2d1;
        font-size: 1rem;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .risk-low { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border-left: 5px solid #28a745; }
    .risk-moderate { background: linear-gradient(135deg, #fff3cd, #ffeaa7); color: #856404; border-left: 5px solid #ffc107; }
    .risk-high { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; border-left: 5px solid #dc3545; }
    .risk-critical { background: linear-gradient(135deg, #721c24, #a71d2a); color: #fff; border-left: 5px solid #ff0000; }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .metric-card h3 { color: #495057; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #212529; }

    .warning-banner {
        background: linear-gradient(135deg, #fff3cd, #ffc107);
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 5px solid #e0a800;
    }

    .critical-banner {
        background: linear-gradient(135deg, #f8d7da, #dc3545);
        color: #fff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 5px solid #a71d2a;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    .disclaimer-footer {
        background: #f8f9fa;
        border-top: 2px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d;
    }

    .upload-area {
        border: 2px dashed #6c757d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: #e94560;
        background: #fff5f7;
    }

    .stSpinner > div > div {
        border-top-color: #e94560 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 AI Skin Disease Detection</h1>
    <p>Upload a skin lesion image for AI-powered analysis • Cloud-based inference • Evidence-based insights</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────

def render_sidebar() -> tuple[Config, dict[str, Any]]:
    """Render sidebar with auto-loaded API keys and patient info form."""

    with st.sidebar:
        # ── API Keys — auto-loaded from .env ───────────────────────
        config = Config.from_env()
        key_status = config.validate()

        st.header("🔑 API Keys")
        all_set = all(key_status.values())
        if all_set:
            st.success("All keys loaded from `.env`")
        else:
            st.caption("Keys auto-loaded from `.env` file")

        for name, is_set in key_status.items():
            icon = "✅" if is_set else "❌"
            st.caption(f"{icon} {name}")

        # Optional override expander
        with st.expander("Override API keys manually", expanded=not all_set):
            hf_override = st.text_input(
                "HuggingFace API Key",
                type="password",
                value="",
                help="Leave blank to use .env value",
                placeholder="Using .env" if config.has_huggingface else "Enter key...",
            )
            groq_override = st.text_input(
                "Groq API Key",
                type="password",
                value="",
                help="Leave blank to use .env value",
                placeholder="Using .env" if config.has_groq else "Enter key...",
            )
            tavily_override = st.text_input(
                "Tavily API Key",
                type="password",
                value="",
                help="Leave blank to use .env value",
                placeholder="Using .env" if config.has_tavily else "Enter key...",
            )

            # If user typed something, override the env values
            if hf_override:
                config.huggingface_api_key = hf_override
            if groq_override:
                config.groq_api_key = groq_override
            if tavily_override:
                config.tavily_api_key = tavily_override

        st.divider()

        # ── Patient Information Form ───────────────────────────────
        st.header("👤 Patient Information")
        st.caption("Used for multi-modal risk assessment")

        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        duration = st.selectbox(
            "How long have you had this lesion?",
            ["Less than 1 month", "1-6 months", "6-12 months", "1-2 years", "More than 2 years"],
        )

        st.markdown("**Symptoms**")
        itching = st.checkbox("Itching")
        bleeding = st.checkbox("Bleeding")
        size_change = st.checkbox("Change in size")
        family_history = st.checkbox("Family history of skin cancer")
        pain_level = st.slider("Pain level", 1, 10, 1)

        patient_info = {
            "age": age,
            "gender": gender,
            "duration": duration,
            "itching": itching,
            "bleeding": bleeding,
            "size_change": size_change,
            "family_history": family_history,
            "pain_level": pain_level,
        }

        return config, patient_info


# ── Risk Color Helpers ─────────────────────────────────────────────

def get_risk_class(level: str) -> str:
    return {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Critical": "risk-critical",
    }.get(level, "risk-moderate")


def get_risk_emoji(level: str) -> str:
    return {
        "Low": "🟢",
        "Moderate": "🟡",
        "High": "🟠",
        "Critical": "🔴",
    }.get(level, "⚪")


# ── Main App ───────────────────────────────────────────────────────

def main() -> None:
    config, patient_info = render_sidebar()

    # Initialize session state
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "risk_result" not in st.session_state:
        st.session_state.risk_result = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── Image Upload ───────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 Upload Skin Lesion Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear photo of the skin lesion for analysis",
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with col_preview:
            st.subheader("🖼️ Uploaded Image")
            st.image(image, use_container_width=True, caption="Uploaded skin lesion")

        # ── Classification ─────────────────────────────────────────
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if not config.has_groq:
                st.error("⚠️ Please enter your Groq API key in the sidebar to proceed (used for image classification via LLaMA 4 Vision).")
                return

            # Step 1: Classify
            with st.spinner("🧠 Classifying skin lesion via HuggingFace AI..."):
                classifier = SkinClassifier(config)
                prediction = classifier.classify(image)
                st.session_state.prediction = prediction

            # Step 2: Risk assessment
            with st.spinner("📊 Computing multi-modal risk assessment..."):
                assessor = RiskAssessor()
                risk_result = assessor.assess(prediction, patient_info)
                st.session_state.risk_result = risk_result

            # Step 3: LLM Analysis
            if config.has_groq or config.has_huggingface:
                with st.spinner("🤖 Running AI analysis: vision diagnosis → internet search → combined analysis..."):
                    try:
                        from agent.llm_agent import analyze_prediction
                        analysis = analyze_prediction(config, prediction, risk_result, patient_info, image=image)
                        st.session_state.analysis_result = analysis
                    except Exception as e:
                        logger.error(f"LLM analysis failed: {e}")
                        st.session_state.analysis_result = (
                            f"⚠️ AI analysis could not be completed: {str(e)}\n\n"
                            "Please check your API keys and try again."
                        )
            else:
                st.session_state.analysis_result = (
                    "ℹ️ Provide a Groq or HuggingFace API key to enable AI-powered analysis."
                )

            st.rerun()

    # ── Display Results ────────────────────────────────────────────
    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        risk_result = st.session_state.risk_result
        analysis = st.session_state.analysis_result

        st.divider()

        # Warnings
        for warning in prediction.get("warnings", []):
            if "URGENT" in warning or "cancer" in warning.lower():
                st.markdown(f'<div class="critical-banner">{warning}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-banner">{warning}</div>', unsafe_allow_html=True)

        # ── Metrics Row ────────────────────────────────────────────
        st.subheader("📊 Prediction Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Condition</h3>
                <div class="value" style="font-size: 1.1rem;">{prediction.get('label', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            confidence = prediction.get("confidence", 0)
            conf_color = "#28a745" if confidence >= 0.7 else "#ffc107" if confidence >= 0.5 else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Model Confidence</h3>
                <div class="value" style="color: {conf_color};">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            cancer_prob = prediction.get("cancer_probability", 0)
            cancer_color = "#dc3545" if cancer_prob >= 0.5 else "#ffc107" if cancer_prob >= 0.2 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cancer Probability</h3>
                <div class="value" style="color: {cancer_color};">{cancer_prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            risk_level = risk_result.get("risk_level", "Unknown") if risk_result else "Unknown"
            risk_emoji = get_risk_emoji(risk_level)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Level</h3>
                <div class="value">{risk_emoji} {risk_level}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence Bar ─────────────────────────────────────────
        st.markdown("**All Predictions:**")
        for pred in prediction.get("all_predictions", [])[:5]:
            st.progress(pred["score"], text=f"{pred['label']} — {pred['score']:.1%}")

        # ── Risk Assessment Details ────────────────────────────────
        if risk_result:
            st.divider()
            risk_class = get_risk_class(risk_result.get("risk_level", ""))

            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h3>{get_risk_emoji(risk_result.get('risk_level', ''))} Risk Assessment: {risk_result.get('risk_level', 'Unknown')}</h3>
                <p><strong>Score:</strong> {risk_result.get('risk_score', 0):.3f} / 1.000</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Detailed Risk Reasoning", expanded=False):
                st.markdown(risk_result.get("reasoning", "N/A"))

                components = risk_result.get("components", {})
                if components:
                    st.markdown("**Component Scores:**")
                    for key, value in components.items():
                        label = key.replace("_", " ").title()
                        st.progress(min(value, 1.0), text=f"{label}: {value:.3f}")

        # ── LLM Analysis ──────────────────────────────────────────
        if analysis:
            st.divider()
            with st.expander("🤖 AI Medical Analysis", expanded=True):
                st.markdown(analysis)

        # ── PDF Download ──────────────────────────────────────────
        st.divider()
        if st.button("📥 Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        prediction=prediction,
                        risk_result=risk_result or {},
                        patient_info=patient_info,
                        llm_analysis=analysis,
                    )
                    st.download_button(
                        label="⬇️ Click to download report",
                        data=pdf_bytes,
                        file_name="skin_disease_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

        # ── Follow-up Chat ────────────────────────────────────────
        st.divider()
        st.subheader("💬 Follow-up Questions")
        st.caption("Ask questions about your analysis")

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_question := st.chat_input("Ask a question about your results..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from agent.llm_agent import chat_followup
                        context = analysis or "No previous analysis available."
                        response = chat_followup(config, user_question, context)
                    except Exception as e:
                        response = f"⚠️ Could not process your question: {e}"

                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

    else:
        # No image uploaded yet — show instructions
        st.info(
            "👆 **Upload a skin lesion image** to get started.\n\n"
            "The AI will analyze the image and provide:\n"
            "- Disease classification with confidence scores\n"
            "- Cancer risk assessment\n"
            "- Evidence-based medical analysis\n"
            "- Downloadable PDF report"
        )

    # ── Footer Disclaimer ──────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-footer">
        ⚕️ <strong>Medical Disclaimer:</strong> This AI system is for <strong>educational purposes only</strong>.
        It does not provide medical advice, diagnosis, or treatment recommendations.
        Always consult a qualified dermatologist or healthcare provider for skin concerns.
        AI predictions may be inaccurate and should never be used as the sole basis for medical decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
