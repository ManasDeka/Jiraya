"""
Enterprise Document Intelligence Platform
-------------------------------------------
Streamlit UI Application with Auto-Ingestion on Startup
"""
import torch
torch.classes.__path__ = []

import os
import sys
import warnings
import logging
import uuid

# ── Silence ALL terminal noise ───────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]          = "3"
os.environ["TORCH_CPP_LOG_LEVEL"]           = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["SENTENCE_TRANSFORMERS_HOME"]    = "./.cache/sentence_transformers"

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("azure").setLevel(logging.ERROR)

import streamlit as st
from ui.styles import get_styles
from ui.chat import run_rag_pipeline, stream_answer

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Enterprise Document Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_styles(), unsafe_allow_html=True)

def _clean_citations(citations: list) -> list:
    """
    Strips answer text that leaks into citations.
    Keeps only: filename.pdf | Page: X

    Input  : ["hr_s2.pdf | Page: 1). Additionally, a **minimum..."]
    Output : ["hr_s2.pdf | Page: 1"]
    """
    import re
    cleaned = []
    seen    = set()

    for citation in citations:
        # ── Extract just filename + page number ──────────────────────
        match = re.search(
            r'([\w\-\.]+\.(pdf|docx|pptx|txt))\s*[\|,]?\s*(Page[:\s]*\d+)?',
            citation,
            re.IGNORECASE,
        )
        if match:
            filename = match.group(1).strip()
            page     = match.group(3).strip() if match.group(3) else ""
            clean    = f"{filename} | {page}" if page else filename

            if clean not in seen:
                seen.add(clean)
                cleaned.append(clean)

    return cleaned



# ══════════════════════════════════════════════════════════════════════
# CHROMADB HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════
def is_chroma_populated() -> bool:
    try:
        import posthog
        posthog.disabled = True
        from ingestion.chroma_store import get_collection_stats
        stats = get_collection_stats()
        return any(count > 0 for count in stats.values())
    except Exception as e:
        print(f"[Startup] ChromaDB check failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# AUTO INGESTION
# ══════════════════════════════════════════════════════════════════════
def run_auto_ingestion() -> dict:
    """
    Runs the full ingestion pipeline.
    ingested_hashes.json ensures already processed files are skipped.
    """
    print("\n[Ingestion] Starting pipeline...")

    try:
        from azure.storage.blob import BlobServiceClient
        from config import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME
        from ingestion.blob_reader import read_blob_file
        from ingestion.text_extractor import extract_text
        from ingestion.pii_masker import mask_pii
        from ingestion.domain_classifier import classify_domain
        from ingestion.chunker import chunk_document
        from ingestion.embedder import generate_embeddings
        from ingestion.chroma_store import store_chunks
        from ingestion.hash_tracker import (
            compute_hash,
            is_already_ingested,
            mark_as_ingested,
        )

        SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt"}

        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client    = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        all_blobs  = list(container_client.list_blobs())
        blob_names = [
            blob.name for blob in all_blobs
            if blob.name.rsplit(".", 1)[-1].lower() in SUPPORTED_EXTENSIONS
        ]

        print(f"[Ingestion] Found {len(blob_names)} supported documents")

        if not blob_names:
            return {"success": True, "ingested": 0, "skipped": 0, "failed": 0, "total": 0}

        ingested = skipped = failed = 0

        for blob_name in blob_names:
            try:
                file_bytes, extension = read_blob_file(blob_name)
                file_hash             = compute_hash(file_bytes)

                if is_already_ingested(blob_name, file_hash):
                    skipped += 1
                    continue

                raw_text    = extract_text(file_bytes, extension)
                if not raw_text.strip():
                    failed += 1
                    continue

                masked_text = mask_pii(raw_text)
                domain      = classify_domain(masked_text)
                doc_id      = str(uuid.uuid4())
                doc_name    = blob_name.rsplit("/", 1)[-1]

                chunks          = chunk_document(masked_text, doc_id, doc_name, domain)
                embedded_chunks = generate_embeddings(chunks)
                store_chunks(embedded_chunks, domain)
                mark_as_ingested(blob_name, file_hash)

                print(f"[Ingestion] ✅ {doc_name} → {domain}")
                ingested += 1

            except Exception as e:
                print(f"[Ingestion] ❌ {blob_name}: {str(e)}")
                failed += 1
                continue

        print(f"[Ingestion] Done — Ingested: {ingested} | Skipped: {skipped} | Failed: {failed}\n")

        return {
            "success":  True,
            "ingested": ingested,
            "skipped":  skipped,
            "failed":   failed,
            "total":    len(blob_names),
        }

    except Exception as e:
        print(f"[Ingestion] ❌ Error: {str(e)}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════
# STARTUP CHECK — Runs once per session
# ══════════════════════════════════════════════════════════════════════
if "startup_check_done" not in st.session_state:
    st.session_state.startup_check_done = False

if not st.session_state.startup_check_done:

    populated = is_chroma_populated()

    if not populated:
        with st.spinner("🔄 Knowledge base is empty. Indexing documents from Azure Blob..."):
            result = run_auto_ingestion()

        if result["success"]:
            if result["total"] == 0:
                st.warning("⚠️ No documents found in Azure Blob Storage.")
            else:
                st.success(
                    f"✅ Knowledge base ready! "
                    f"📄 Total: **{result['total']}** | "
                    f"✅ Indexed: **{result['ingested']}** | "
                    f"⏭️ Skipped: **{result['skipped']}** | "
                    f"❌ Failed: **{result['failed']}**"
                )
        else:
            st.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")

    st.session_state.startup_check_done = True


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════
# LOAD RAG APP ONCE
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_rag_app():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    from rag.graph import build_rag_graph
    app        = build_rag_graph()
    sys.stdout = old_stdout
    return app


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — LEFT PANEL
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
            <div style="width:44px; height:44px; background:#1a73e8; border-radius:10px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:22px;">🏢</div>
            <div class="platform-title">Enterprise<br>Document<br>Intelligence<br>Platform</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:13px; color:#666; padding: 0 4px; line-height:1.8;">
            <b>Supported Domains:</b><br>
            🟢 HR &nbsp;&nbsp; 🔵 IT &nbsp;&nbsp; 🟡 Finance &nbsp;&nbsp; 🟠 Operations
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── ChromaDB Live Stats ───────────────────────────────────────────
    st.markdown("#### 📊 Knowledge Base")
    try:
        from ingestion.chroma_store import get_collection_stats
        stats = get_collection_stats()

        domain_icons = {"HR": "🟢", "IT": "🔵", "Finance": "🟡", "Operations": "🟠"}

        for domain, count in stats.items():
            icon = domain_icons.get(domain, "⚪")
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                            font-size:13px; color:#444; padding:4px 6px;
                            background:#f8f9fa; border-radius:6px; margin:3px 0;">
                    <span>{icon} {domain}</span>
                    <span style="font-weight:600; color:#1a73e8;">{count} chunks</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown("<p style='font-size:12px; color:#999;'>Stats unavailable</p>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:12px; color:#999; padding: 0 4px; line-height:1.7;">
            🔒 <b>Guardrails Active</b><br>
            PII detection & profanity filter enabled.
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='chat-header'>Hi, How Can I Help? 👋</div>",
    unsafe_allow_html=True,
)

# ── Render Chat History ───────────────────────────────────────────────
for message in st.session_state.messages:
    role      = message["role"]
    content   = message["content"]
    citations = message.get("citations", [])

    if role == "user":
        st.markdown(
            f'<div class="user-bubble"><div class="user-bubble-inner">{content}</div></div>',
            unsafe_allow_html=True,
        )

    elif role == "assistant":
        st.markdown(
            f'<div class="bot-bubble"><div class="bot-bubble-inner">{content}</div></div>',
            unsafe_allow_html=True,
        )
        # ── Clean citations — filename + page only ────────────────────
        if citations:
            clean_citations = _clean_citations(citations)
            if clean_citations:
                citation_text = " &nbsp;|&nbsp; ".join([f"📄 {c}" for c in clean_citations])
                st.markdown(
                    f'<div class="citation-box">📚 Sources: {citation_text}</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════
# CITATION CLEANER — Filename + Page Only
# ══════════════════════════════════════════════════════════════════════
# def _clean_citations(citations: list) -> list:
#     """
#     Strips answer text that leaks into citations.
#     Keeps only: filename.pdf | Page: X

#     Input  : ["hr_s2.pdf | Page: 1). Additionally, a **minimum..."]
#     Output : ["hr_s2.pdf | Page: 1"]
#     """
#     import re
#     cleaned = []
#     seen    = set()

#     for citation in citations:
#         # ── Extract just filename + page number ──────────────────────
#         match = re.search(
#             r'([\w\-\.]+\.(pdf|docx|pptx|txt))\s*[\|,]?\s*(Page[:\s]*\d+)?',
#             citation,
#             re.IGNORECASE,
#         )
#         if match:
#             filename = match.group(1).strip()
#             page     = match.group(3).strip() if match.group(3) else ""
#             clean    = f"{filename} | {page}" if page else filename

#             if clean not in seen:
#                 seen.add(clean)
#                 cleaned.append(clean)

#     return cleaned


# ══════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            label="Message",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )
    with col2:
        send_clicked = st.form_submit_button("Send →")


# ══════════════════════════════════════════════════════════════════════
# HANDLE SEND
# ══════════════════════════════════════════════════════════════════════
if send_clicked and user_input.strip():

    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    st.markdown(
        f'<div class="user-bubble"><div class="user-bubble-inner">{user_input.strip()}</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.rag_app is None:
        with st.spinner("Initializing AI engine..."):
            st.session_state.rag_app = load_rag_app()

    with st.spinner("Thinking..."):
        result = run_rag_pipeline(
            question=user_input.strip(),
            rag_app=st.session_state.rag_app,
            session_id=st.session_state.session_id,
        )

    answer    = result["answer"]
    citations = result.get("citations", [])

    with st.container():
        st.markdown('<div class="bot-bubble"><div class="bot-bubble-inner">', unsafe_allow_html=True)
        st.write_stream(stream_answer(answer))
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Show clean citations — filename + page only ───────────────────
    if citations and not result.get("guardrail_triggered"):
        clean_citations = _clean_citations(citations)
        if clean_citations:
            citation_text = " &nbsp;|&nbsp; ".join([f"📄 {c}" for c in clean_citations])
            st.markdown(
                f'<div class="citation-box">📚 Sources: {citation_text}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "citations": citations,
    })

    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; margin-top:60px; color:#9aa0a6;">
            <div style="font-size:48px;">💬</div>
            <div style="font-size:16px; margin-top:12px;">
                Ask a question about your enterprise documents.
            </div>
            <div style="font-size:13px; margin-top:6px; color:#bbb;">
                Supports HR, IT, Finance and Operations domains.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
