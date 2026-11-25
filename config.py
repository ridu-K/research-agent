"""All settings in one place. Change models and limits here, not inside the code."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(ENV_PATH)

# --- API keys (all free). Read ONLY from the local .env file; restart the app after editing. ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or None
IEEE_API_KEY = os.getenv("IEEE_API_KEY") or None                          # optional
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None  # optional
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL") or None  # optional: OpenAlex "polite pool"

# --- Models ---
FAST_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")    # Groq: ranking, chat answers
# Used for extraction only when every Gemini model is busy (each has its own 8k tokens/min)
GROQ_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b").split(",") if m.strip()]
FALLBACK_TEXT_CHARS = 18000   # start + end of the paper, fits Groq's free per-minute limit
# Gemini: extraction, synopsis, chat fallback. Tried in order; a model that is retired (404)
# or overloaded (503/429) is skipped automatically. Override with a comma-separated list.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS",
    "gemini-flash-latest,gemini-3.6-flash,gemini-3.5-flash,gemini-flash-lite-latest",
).split(",") if m.strip()]
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768                                                # smaller vectors = smaller DB

# --- Storage ---
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
DB_PATH = DATA_DIR / "research.db"

# --- Discovery ---
RESULTS_PER_SOURCE = 20
RANK_BATCH_SIZE = 10        # abstracts per relevance-scoring call
ABSTRACT_CHARS_FOR_RANKING = 900
HTTP_TIMEOUT = 20           # seconds
IEEE_DOI_PREFIX = "10.1109/"

# --- Papers ---
MAX_UPLOADS = 30                 # PDFs per upload batch
MAX_PDF_MB = 25                  # skip bigger downloads
MAX_PAPER_CHARS = 120_000        # text sent to Gemini per paper (~30k tokens)
MIN_TEXT_CHARS = 1_500           # less than this = probably a scanned PDF -> send the PDF itself

# --- Chat (RAG) ---
CHUNK_CHARS = 1_200              # characters per chunk
CHUNK_OVERLAP = 200              # characters shared between neighbouring chunks
TOP_K = 8                        # chunks given to the LLM per question
MAX_CHUNKS_PER_PAPER = 3         # so one paper can't crowd out the others
HISTORY_TURNS = 3                # previous Q&A pairs remembered in a chat

# --- Synopsis ---
SYNOPSIS_WORDS = 650             # body length that fits one page
