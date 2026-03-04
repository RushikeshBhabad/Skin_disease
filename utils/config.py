"""
Configuration management for the Skin Disease Detection System.
Loads API keys from .env file or Streamlit session state.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root (parent of utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Central configuration holding all API keys and system settings."""

    groq_api_key: Optional[str] = field(default=None)
    huggingface_api_key: Optional[str] = field(default=None)
    tavily_api_key: Optional[str] = field(default=None)

    # Model settings
    classification_model: str = "Anwarkh1/Skin_Cancer-Image_Classification"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1

    # RAG settings
    rag_top_k: int = 3

    # Safety thresholds
    low_confidence_threshold: float = 0.60
    high_cancer_threshold: float = 0.70

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )

    @classmethod
    def from_session(
        cls,
        groq_key: Optional[str] = None,
        hf_key: Optional[str] = None,
        tavily_key: Optional[str] = None,
    ) -> "Config":
        """Create config from user-provided keys (Streamlit sidebar), falling back to env."""
        return cls(
            groq_api_key=groq_key or os.getenv("GROQ_API_KEY"),
            huggingface_api_key=hf_key or os.getenv("HUGGINGFACE_API_KEY"),
            tavily_api_key=tavily_key or os.getenv("TAVILY_API_KEY"),
        )

    def validate(self) -> dict[str, bool]:
        """Check which API keys are present. Returns a dict of key_name -> is_set."""
        return {
            "HuggingFace": bool(self.huggingface_api_key),
            "Groq": bool(self.groq_api_key),
            "Tavily": bool(self.tavily_api_key),
        }

    @property
    def has_huggingface(self) -> bool:
        return bool(self.huggingface_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)
