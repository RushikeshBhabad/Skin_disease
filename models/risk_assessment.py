"""
Multi-modal risk assessment engine for the Skin Disease Detection System.
Combines AI model output with patient metadata using a weighted medical scoring algorithm.
"""

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Disease classes with inherent risk weights
DISEASE_RISK_WEIGHTS: dict[str, float] = {
    "melanoma": 1.0,
    "mel": 1.0,
    "squamous cell carcinoma": 0.9,
    "basal cell carcinoma": 0.8,
    "bcc": 0.8,
    "actinic keratosis": 0.5,
    "akiec": 0.5,
    "pigmented benign keratosis": 0.1,
    "bkl": 0.1,
    "seborrheic keratosis": 0.05,
    "dermatofibroma": 0.05,
    "df": 0.05,
    "melanocytic nevus (mole)": 0.05,
    "nevus": 0.05,
    "nv": 0.05,
    "vascular lesion": 0.05,
    "vasc": 0.05,
}


class RiskAssessor:
    """
    Calculates overall risk level by combining model predictions
    with structured patient metadata.
    """

    # Weight distribution for final score (must sum to 1.0)
    W_MODEL: float = 0.35       # Model confidence × disease risk
    W_CANCER_PROB: float = 0.25  # Cancer probability from classifier
    W_SYMPTOMS: float = 0.25    # Patient-reported symptoms
    W_HISTORY: float = 0.15     # Demographic and family history factors

    def assess(
        self,
        prediction: dict[str, Any],
        patient_info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a full risk assessment.

        Args:
            prediction: Output dict from SkinClassifier.classify().
            patient_info: Dict with keys:
                - age: int
                - gender: str
                - duration: str   (e.g. "Less than 1 month", "1-6 months", etc.)
                - itching: bool
                - bleeding: bool
                - size_change: bool
                - family_history: bool
                - pain_level: int   (1-10)

        Returns:
            Dict with keys:
                - risk_level: str ("Low" / "Moderate" / "High" / "Critical")
                - risk_score: float (0-1)
                - reasoning: str (human-readable explanation)
                - components: dict of sub-scores for transparency
        """
        try:
            model_score = self._compute_model_score(prediction)
            cancer_score = self._compute_cancer_score(prediction)
            symptom_score = self._compute_symptom_score(patient_info)
            history_score = self._compute_history_score(patient_info)

            final_score = (
                self.W_MODEL * model_score
                + self.W_CANCER_PROB * cancer_score
                + self.W_SYMPTOMS * symptom_score
                + self.W_HISTORY * history_score
            )

            risk_level = self._score_to_level(final_score)
            reasoning = self._generate_reasoning(
                prediction, patient_info, model_score, cancer_score,
                symptom_score, history_score, final_score, risk_level,
            )

            result = {
                "risk_level": risk_level,
                "risk_score": round(final_score, 3),
                "reasoning": reasoning,
                "components": {
                    "model_score": round(model_score, 3),
                    "cancer_score": round(cancer_score, 3),
                    "symptom_score": round(symptom_score, 3),
                    "history_score": round(history_score, 3),
                },
            }

            logger.info(f"Risk assessment complete: {risk_level} (score={final_score:.3f})")
            return result

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                "risk_level": "Unknown",
                "risk_score": 0.0,
                "reasoning": f"Risk assessment could not be completed: {e}",
                "components": {},
            }

    # ── Sub-score calculators ──────────────────────────────────────

    def _compute_model_score(self, prediction: dict[str, Any]) -> float:
        """Score based on model confidence weighted by disease severity."""
        confidence = prediction.get("confidence", 0.0)
        raw_label = prediction.get("raw_label", "").lower()
        disease_weight = DISEASE_RISK_WEIGHTS.get(raw_label, 0.1)
        return confidence * disease_weight

    def _compute_cancer_score(self, prediction: dict[str, Any]) -> float:
        """Score directly from the classifier's cancer probability."""
        return prediction.get("cancer_probability", 0.0)

    def _compute_symptom_score(self, patient_info: dict[str, Any]) -> float:
        """
        Score from ABCDE-inspired symptom analysis.

        Factors:
            - Bleeding       → +0.30  (strong melanoma indicator)
            - Size change     → +0.25  (ABCDE: Evolving)
            - Itching         → +0.10
            - Pain level      → +0.20  (scaled from 1-10)
            - Duration        → +0.15  (longer = more concern for malignancy)
        """
        score = 0.0

        if patient_info.get("bleeding", False):
            score += 0.30

        if patient_info.get("size_change", False):
            score += 0.25

        if patient_info.get("itching", False):
            score += 0.10

        pain = patient_info.get("pain_level", 1)
        score += 0.20 * (min(max(pain, 1), 10) / 10)

        duration_map = {
            "Less than 1 month": 0.02,
            "1-6 months": 0.05,
            "6-12 months": 0.08,
            "1-2 years": 0.10,
            "More than 2 years": 0.15,
        }
        score += duration_map.get(patient_info.get("duration", ""), 0.05)

        return min(score, 1.0)

    def _compute_history_score(self, patient_info: dict[str, Any]) -> float:
        """
        Score from demographics and family history.

        Factors:
            - Family history of skin cancer → +0.40
            - Age > 50                      → +0.20
            - Age > 65                      → +0.10 (additional)
            - Male gender                   → +0.10 (slight epidemiological bias)
        """
        score = 0.0

        if patient_info.get("family_history", False):
            score += 0.40

        age = patient_info.get("age", 30)
        if age > 50:
            score += 0.20
        if age > 65:
            score += 0.10

        if patient_info.get("gender", "").lower() in ("male", "m"):
            score += 0.10

        return min(score, 1.0)

    # ── Score interpretation ────────────────────────────────────────

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map a 0-1 score to a risk level string."""
        if score >= 0.70:
            return "Critical"
        elif score >= 0.45:
            return "High"
        elif score >= 0.25:
            return "Moderate"
        else:
            return "Low"

    @staticmethod
    def _generate_reasoning(
        prediction: dict[str, Any],
        patient_info: dict[str, Any],
        model_score: float,
        cancer_score: float,
        symptom_score: float,
        history_score: float,
        final_score: float,
        risk_level: str,
    ) -> str:
        """Generate a human-readable explanation of the risk assessment."""
        parts: list[str] = []

        label = prediction.get("label", "Unknown")
        confidence = prediction.get("confidence", 0.0)
        parts.append(
            f"The AI model predicted '{label}' with {confidence:.0%} confidence."
        )

        cancer_prob = prediction.get("cancer_probability", 0.0)
        if cancer_prob > 0.5:
            parts.append(f"Cancer probability is elevated at {cancer_prob:.0%}.")
        elif cancer_prob > 0.2:
            parts.append(f"Cancer probability is moderate at {cancer_prob:.0%}.")
        else:
            parts.append(f"Cancer probability is low at {cancer_prob:.0%}.")

        # Symptom flags
        symptom_flags = []
        if patient_info.get("bleeding"):
            symptom_flags.append("bleeding")
        if patient_info.get("size_change"):
            symptom_flags.append("change in size")
        if patient_info.get("itching"):
            symptom_flags.append("itching")
        pain = patient_info.get("pain_level", 1)
        if pain >= 5:
            symptom_flags.append(f"significant pain (level {pain}/10)")

        if symptom_flags:
            parts.append(
                f"Concerning symptoms reported: {', '.join(symptom_flags)}."
            )
        else:
            parts.append("No alarming symptoms reported.")

        if patient_info.get("family_history"):
            parts.append("Family history of skin cancer increases the risk profile.")

        age = patient_info.get("age", 30)
        if age > 50:
            parts.append(f"Patient age ({age}) is a risk factor for skin malignancies.")

        parts.append(
            f"\nOverall risk score: {final_score:.2f}/1.00 -- {risk_level} risk."
        )

        if risk_level == "Critical":
            parts.append(
                "IMMEDIATE dermatologist consultation is strongly recommended."
            )
        elif risk_level == "High":
            parts.append(
                "Please schedule a dermatologist appointment promptly."
            )
        elif risk_level == "Moderate":
            parts.append(
                "Consider consulting a dermatologist for further evaluation."
            )
        else:
            parts.append(
                "Risk appears low, but routine skin checks are always recommended."
            )

        return "\n".join(parts)
