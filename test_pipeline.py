#!/usr/bin/env python3
"""
End-to-end test script for the Skin Disease Detection System.
Tests the full pipeline: classification -> risk assessment -> PDF report.
Runs against multiple test images.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from utils.config import Config
from utils.logger import get_logger
from models.classifier import SkinClassifier
from models.risk_assessment import RiskAssessor
from utils.pdf_report import generate_pdf_report

logger = get_logger("test_pipeline")

# Test patient profiles for varied risk scenarios
PATIENT_PROFILES = [
    {
        "name": "High-risk patient",
        "age": 62, "gender": "Male", "duration": "6-12 months",
        "itching": True, "bleeding": True, "size_change": True,
        "family_history": True, "pain_level": 7,
    },
    {
        "name": "Low-risk patient",
        "age": 25, "gender": "Female", "duration": "Less than 1 month",
        "itching": False, "bleeding": False, "size_change": False,
        "family_history": False, "pain_level": 1,
    },
    {
        "name": "Moderate-risk patient",
        "age": 45, "gender": "Male", "duration": "1-6 months",
        "itching": True, "bleeding": False, "size_change": True,
        "family_history": False, "pain_level": 4,
    },
]

TEST_IMAGES = [
    "test_images/skin_lesion_1.jpg",
    "test_images/synthetic_mole.jpg",
    "test_images/synthetic_dark_lesion.jpg",
]


def test_image(image_path: str, patient_info: dict, config: Config) -> dict:
    """Test a single image through the full pipeline."""
    print(f"\n{'='*70}")
    print(f"  IMAGE: {os.path.basename(image_path)}")
    print(f"  PATIENT: {patient_info.get('name', 'Unknown')} (Age {patient_info['age']}, {patient_info['gender']})")
    print(f"{'='*70}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size}")

    # Step 1: Classification
    print("\n  [Step 1] Classifying with HuggingFace API...")
    classifier = SkinClassifier(config)
    prediction = classifier.classify(image)

    print(f"  ✅ Prediction: {prediction['label']}")
    print(f"     Confidence: {prediction['confidence']:.1%}")
    print(f"     Cancer Probability: {prediction['cancer_probability']:.1%}")
    print(f"     Is Cancerous: {prediction['is_cancerous']}")

    if prediction.get("all_predictions"):
        print(f"     All predictions:")
        for p in prediction["all_predictions"][:5]:
            print(f"       - {p['label']}: {p['score']:.1%}")

    for w in prediction.get("warnings", []):
        print(f"     ⚠️  {w}")

    # Step 2: Risk Assessment
    print("\n  [Step 2] Computing risk assessment...")
    assessor = RiskAssessor()
    risk_result = assessor.assess(prediction, patient_info)

    print(f"  ✅ Risk Level: {risk_result['risk_level']}")
    print(f"     Risk Score: {risk_result['risk_score']:.3f}/1.000")
    print(f"     Components: {risk_result['components']}")

    # Step 3: PDF Report
    print("\n  [Step 3] Generating PDF report...")
    try:
        pdf_bytes = generate_pdf_report(
            prediction=prediction,
            risk_result=risk_result,
            patient_info=patient_info,
            llm_analysis="Test analysis - LLM analysis would appear here in production.",
        )
        pdf_path = f"test_images/report_{os.path.splitext(os.path.basename(image_path))[0]}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"  ✅ PDF saved: {pdf_path} ({len(pdf_bytes)} bytes)")
    except Exception as e:
        print(f"  ❌ PDF generation failed: {e}")

    return {
        "image": image_path,
        "prediction": prediction,
        "risk": risk_result,
    }


def main():
    print("🔬 AI Skin Disease Detection — End-to-End Test")
    print("=" * 70)

    # Load config
    config = Config.from_env()
    keys = config.validate()
    print(f"\nAPI Key Status: {keys}")

    if not config.has_groq:
        print("\n❌ ERROR: GROQ_API_KEY not set!")
        print("   Please create a .env file in the project root with:")
        print("   GROQ_API_KEY=your_groq_api_key")
        sys.exit(1)

    # Find test images
    available_images = [img for img in TEST_IMAGES if os.path.exists(img)]
    print(f"\nTest images found: {len(available_images)}")
    for img in available_images:
        size = os.path.getsize(img)
        print(f"  - {img} ({size} bytes)")

    if not available_images:
        print("❌ No test images found!")
        sys.exit(1)

    # Run tests
    results = []
    for i, image_path in enumerate(available_images):
        patient = PATIENT_PROFILES[i % len(PATIENT_PROFILES)]
        result = test_image(image_path, patient, config)
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        pred = r["prediction"]
        risk = r["risk"]
        print(f"  {os.path.basename(r['image']):30s} | "
              f"{pred['label']:30s} | "
              f"Conf: {pred['confidence']:.0%} | "
              f"Cancer: {pred['cancer_probability']:.0%} | "
              f"Risk: {risk['risk_level']}")

    print(f"\n✅ All {len(results)} tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
