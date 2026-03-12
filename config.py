import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure Blob Storage ──────────────────────────────────────────────
BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# ── Azure OpenAI ────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# ── LLM Parameters (Ingestion) ──────────────────────────────────────
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 50

# ── Chunking Parameters ─────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

# ── Embedding Parameters ────────────────────────────────────────────
EMBEDDING_BATCH_SIZE = 10

# ── ChromaDB ────────────────────────────────────────────────────────
CHROMA_HR_COLLECTION = "hr_collection"
CHROMA_IT_COLLECTION = "it_collection"
CHROMA_FINANCE_COLLECTION = "finance_collection"
CHROMA_OPERATIONS_COLLECTION = "operations_collection"

# ── Domains ─────────────────────────────────────────────────────────
SUPPORTED_DOMAINS = ["HR", "IT", "Finance", "Operations"]

# ── Document Defaults ───────────────────────────────────────────────
DEFAULT_DOC_VERSION = "1.0"

# ════════════════════════════════════════════════════════════════════
# RAG SERVING PARAMETERS
# ════════════════════════════════════════════════════════════════════

# ── RAG Retrieval ────────────────────────────────────────────────────
RAG_TOP_K_INITIAL = 5          # Chunks retrieved on first attempt
RAG_TOP_K_RETRY = 10           # Chunks retrieved on retry attempt

# ── Classifier LLM ──────────────────────────────────────────────────
CLASSIFIER_TEMPERATURE = 0.0
CLASSIFIER_MAX_TOKENS = 50

# ── Summarizer LLM ──────────────────────────────────────────────────
SUMMARIZER_TEMPERATURE = 0.2
SUMMARIZER_MAX_TOKENS = 1000

# ── Validator LLM ───────────────────────────────────────────────────
VALIDATOR_TEMPERATURE = 0.0
VALIDATOR_MAX_TOKENS = 10

# ── Reranker ─────────────────────────────────────────────────────────
ENABLE_RERANKER = True                                    # Toggle ON/OFF
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # HuggingFace cross-encoder
RERANKER_TOP_N = 3             # Chunks to keep after reranking

# ── Guardrails ───────────────────────────────────────────────────────
ENABLE_INPUT_GUARDRAIL = True   # Toggle input guardrail ON/OFF
ENABLE_OUTPUT_GUARDRAIL = True  # Toggle output guardrail ON/OFF

# ── Retry Logic ──────────────────────────────────────────────────────
MAX_RETRY_COUNT = 1             # Max retries before fallback response

HF_HUB_DISABLE_SYMLINKS_WARNING=1

CHROMA_DB_PATH = "./chroma_db"

