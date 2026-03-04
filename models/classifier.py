"""
Skin lesion image classifier using Groq Vision API (LLaMA 4 Scout).
No local model weights — all inference runs in the cloud via Groq's
multimodal vision endpoint.
"""

import base64
import io
import json
import re
from typing import Any

from PIL import Image

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Mapping of common disease labels to metadata
LABEL_MAP: dict[str, dict[str, Any]] = {
    "actinic keratosis": {"display": "Actinic Keratosis", "is_cancerous": False, "cancer_risk": 0.35},
    "basal cell carcinoma": {"display": "Basal Cell Carcinoma", "is_cancerous": True, "cancer_risk": 0.85},
    "dermatofibroma": {"display": "Dermatofibroma", "is_cancerous": False, "cancer_risk": 0.05},
    "melanoma": {"display": "Melanoma", "is_cancerous": True, "cancer_risk": 0.95},
    "melanocytic nevus": {"display": "Melanocytic Nevus (Mole)", "is_cancerous": False, "cancer_risk": 0.05},
    "nevus": {"display": "Melanocytic Nevus (Mole)", "is_cancerous": False, "cancer_risk": 0.05},
    "pigmented benign keratosis": {"display": "Pigmented Benign Keratosis", "is_cancerous": False, "cancer_risk": 0.05},
    "seborrheic keratosis": {"display": "Seborrheic Keratosis", "is_cancerous": False, "cancer_risk": 0.03},
    "squamous cell carcinoma": {"display": "Squamous Cell Carcinoma", "is_cancerous": True, "cancer_risk": 0.90},
    "vascular lesion": {"display": "Vascular Lesion", "is_cancerous": False, "cancer_risk": 0.05},
    # Short aliases from HAM10000
    "akiec": {"display": "Actinic Keratosis", "is_cancerous": False, "cancer_risk": 0.35},
    "bcc": {"display": "Basal Cell Carcinoma", "is_cancerous": True, "cancer_risk": 0.85},
    "bkl": {"display": "Pigmented Benign Keratosis", "is_cancerous": False, "cancer_risk": 0.05},
    "df": {"display": "Dermatofibroma", "is_cancerous": False, "cancer_risk": 0.05},
    "mel": {"display": "Melanoma", "is_cancerous": True, "cancer_risk": 0.95},
    "nv": {"display": "Melanocytic Nevus (Mole)", "is_cancerous": False, "cancer_risk": 0.05},
    "vasc": {"display": "Vascular Lesion", "is_cancerous": False, "cancer_risk": 0.05},
}

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CLASSIFICATION_PROMPT = """You are a dermatology AI assistant analyzing a skin lesion image.

Analyze this image carefully and classify the skin condition.

Return ONLY valid JSON (no markdown fences, no extra text):
{
  "predictions": [
    {"label": "disease_name", "confidence": 0.XX},
    {"label": "disease_name", "confidence": 0.XX}
  ],
  "analysis": "brief visual description of what you observe"
}

Choose labels ONLY from this list:
- melanoma
- basal cell carcinoma
- squamous cell carcinoma
- actinic keratosis
- melanocytic nevus
- dermatofibroma
- vascular lesion
- pigmented benign keratosis
- seborrheic keratosis

List your top 3-5 predictions with confidence scores (0.0 to 1.0, must sum to ~1.0).
Order from highest to lowest confidence.
Respond with ONLY the JSON object."""


class SkinClassifier:
    """
    Classifies skin lesion images using Groq's multimodal Vision API.

    All inference is performed in the cloud — no model weights are
    downloaded or stored locally.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazily initialize the Groq client."""
        if self._client is None:
            if not self.config.has_groq:
                raise ValueError(
                    "Groq API key is required for image classification. "
                    "Please set GROQ_API_KEY in .env or enter it in the sidebar."
                )
            from groq import Groq
            self._client = Groq(api_key=self.config.groq_api_key)
        return self._client

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded JPEG string."""
        img_resized = image.copy()
        # Resize to 224x224 for consistency and speed
        img_resized = img_resized.resize((224, 224), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img_resized.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def classify(self, image: Image.Image) -> dict[str, Any]:
        """
        Classify a skin lesion image using Groq Vision API.

        Args:
            image: PIL Image of the skin lesion.

        Returns:
            Dict with keys:
                - label: Human-readable predicted disease name
                - raw_label: Original model output label
                - confidence: Float 0-1, top prediction confidence
                - cancer_probability: Float 0-1, estimated cancer probability
                - is_cancerous: Boolean flag
                - all_predictions: List of all predictions with scores
                - warnings: List of warning strings
        """
        try:
            client = self._get_client()
            img_b64 = self._image_to_base64(image)

            logger.info(f"Sending image to Groq Vision model: {VISION_MODEL}")

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": CLASSIFICATION_PROMPT,
                        },
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
            )

            raw_content = response.choices[0].message.content
            logger.info(f"Received response from Groq Vision API")

            return self._parse_response(raw_content)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "label": "Classification Failed",
                "raw_label": "error",
                "confidence": 0.0,
                "cancer_probability": 0.0,
                "is_cancerous": False,
                "all_predictions": [],
                "warnings": [f"Classification error: {str(e)}. Please check your API key and try again."],
            }

    def _parse_response(self, raw_content: str) -> dict[str, Any]:
        """Parse the JSON response from the vision model."""
        # Strip markdown code fences if present
        cleaned = raw_content.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from model response: {raw_content[:200]}")
            return {
                "label": "Parse Error",
                "raw_label": "parse_error",
                "confidence": 0.0,
                "cancer_probability": 0.0,
                "is_cancerous": False,
                "all_predictions": [],
                "warnings": ["Could not parse AI response. Please try again."],
            }

        predictions = data.get("predictions", [])
        if not predictions:
            return {
                "label": "No Prediction",
                "raw_label": "none",
                "confidence": 0.0,
                "cancer_probability": 0.0,
                "is_cancerous": False,
                "all_predictions": [],
                "warnings": ["Model returned no predictions."],
            }

        return self._process_predictions(predictions)

    def _process_predictions(self, predictions: list[dict]) -> dict[str, Any]:
        """Process parsed predictions into a structured result dict."""
        # Sort by confidence descending
        sorted_preds = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)
        top = sorted_preds[0]

        raw_label = top.get("label", "unknown").lower().strip()
        confidence = float(top.get("confidence", 0.0))

        # Map label
        label_info = LABEL_MAP.get(raw_label, {
            "display": raw_label.replace("_", " ").title(),
            "is_cancerous": False,
            "cancer_risk": 0.10,
        })

        # Cancer probability: primary prediction risk + weighted secondary
        cancer_probability = label_info["cancer_risk"] * confidence
        for pred in sorted_preds[1:]:
            pred_label = pred.get("label", "").lower().strip()
            pred_info = LABEL_MAP.get(pred_label, {"cancer_risk": 0.0})
            pred_score = float(pred.get("confidence", 0.0))
            cancer_probability += pred_info.get("cancer_risk", 0.0) * pred_score * 0.5
        cancer_probability = min(cancer_probability, 1.0)

        # Warnings
        warnings: list[str] = []
        if confidence < self.config.low_confidence_threshold:
            warnings.append(
                "Low model certainty -- this prediction may be unreliable. "
                "Please consult a dermatologist for accurate diagnosis."
            )
        if cancer_probability > self.config.high_cancer_threshold:
            warnings.append(
                "URGENT: High cancer probability detected. "
                "Immediate dermatologist consultation is strongly recommended."
            )

        # All predictions for display
        all_predictions = []
        for pred in sorted_preds:
            pred_lbl = pred.get("label", "unknown").lower().strip()
            pred_info = LABEL_MAP.get(pred_lbl, {"display": pred_lbl.title()})
            all_predictions.append({
                "label": pred_info["display"],
                "score": float(pred.get("confidence", 0.0)),
            })

        return {
            "label": label_info["display"],
            "raw_label": raw_label,
            "confidence": confidence,
            "cancer_probability": cancer_probability,
            "is_cancerous": label_info["is_cancerous"],
            "all_predictions": all_predictions,
            "warnings": warnings,
        }
