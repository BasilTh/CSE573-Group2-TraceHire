"""Paths and settings for the extraction code.

LLM settings come from environment variables or the .env file at the repo root.
Extraction runs Llama 3.1 8B on local Ollama by default (Llama 3.1 8B is not on
the ASU API). It has its own endpoint and model settings, so the team's
OPENAI_BASE_URL (used by the agents) does not redirect it by accident:
  EXTRACTION_BASE_URL  default http://localhost:11434/v1 (Ollama)
  EXTRACTION_MODEL     default llama3.1:8b
  OPENAI_API_KEY       the same key the agents use; Ollama ignores it
To run extraction on the ASU API instead, set EXTRACTION_BASE_URL to the
OPENAI_BASE_URL value and EXTRACTION_MODEL to a model the API offers.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "DATA"
SLICE = DATA / "slice"
CACHE = DATA / "cache"
GOLD = DATA / "gold"
EXTRACTED = DATA / "extracted"
LIGHTCAST_SKILLS = DATA / "raw" / "lightcast" / "skills.json"
EVAL = ROOT / "EVALUATIONS"

LLM_BASE_URL = os.getenv("EXTRACTION_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("EXTRACTION_MODEL", "llama3.1:8b")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

SEED = 573
