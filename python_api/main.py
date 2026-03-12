from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import json

# Reuse your existing modules
from rag.graph import build_rag_graph
from ui.chat import run_rag_pipeline, stream_answer
from ingestion.chroma_store import get_collection_stats
from python_api.services.ingestion_service import is_chroma_populated, run_auto_ingestion

app = FastAPI(title="Enterprise Document Intelligence API", version="1.0.0")

# CORS: allow your Node frontend origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # TODO: add prod URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_app = None  # compiled graph, built once


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.on_event("startup")
def on_startup():
    global rag_app
    # 1) Build RAG graph once
    rag_app = build_rag_graph()
    # 2) Auto-ingest if Chroma is empty
    try:
        if not is_chroma_populated():
            # Run synchronously at startup; can be moved to a background task if long
            run_auto_ingestion()
    except Exception as e:
        # Do not crash API if ingestion fails; log and keep running
        print(f"[Startup] Ingestion error: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/kb/stats")
def kb_stats():
    return get_collection_stats()


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        return {
            "answer": "Please provide a question.",
            "domain": "N/A",
            "citations": [],
            "guardrail_triggered": False,
            "validation": "ERROR",
        }
    result = run_rag_pipeline(
        question=req.question.strip(),
        rag_app=rag_app,
        session_id=req.session_id,
    )
    # result already matches your UI contract
    return result


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        # Expect a single JSON message with question (+ optional session_id)
        data = await ws.receive_text()
        payload = json.loads(data)
        question = (payload.get("question") or "").strip()
        session_id = payload.get("session_id")

        if not question:
            await ws.send_text(json.dumps({"type": "error", "message": "Empty question"}))
            await ws.close()
            return

        # Run full pipeline first (same as non-streaming)
        result = run_rag_pipeline(
            question=question, rag_app=rag_app, session_id=session_id
        )

        # Stream word-by-word, same as Streamlit's st.write_stream(stream_answer)
        for token in stream_answer(result.get("answer", "")):
            await ws.send_text(json.dumps({"type": "token", "value": token}))

        # Final payload
        final_payload = {
            "type": "done",
            "answer": result.get("answer", ""),
            "domain": result.get("domain", "N/A"),
            "citations": result.get("citations", []),
            "guardrail_triggered": result.get("guardrail_triggered", False),
            "validation": result.get("validation", "N/A"),
        }
        await ws.send_text(json.dumps(final_payload))
        await ws.close()

    except WebSocketDisconnect:
        # client disconnected
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass