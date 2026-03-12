"""
Enterprise Document Intelligence Chatbot
------------------------------------------
CLI entry point for the RAG-powered chatbot.
Accepts user queries and runs them through the
full LangGraph agentic RAG pipeline.

Usage:
    python chatbot.py
"""

from rag.graph import build_rag_graph
from rag.state import RAGState


def run_chatbot():
    """
    Interactive CLI chatbot loop.
    Builds the RAG graph once and reuses it for all queries.
    """
    print("\n" + "="*60)
    print("   ENTERPRISE DOCUMENT INTELLIGENCE CHATBOT")
    print("="*60)
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    # Build graph once — reused for all queries
    app = build_rag_graph()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break

            # ── Initialize State ─────────────────────────────────────
            initial_state: RAGState = {
                "question": user_input,
                "cleaned_question": "",
                "domain": "",
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "answer": "",
                "validation_result": "",
                "retry_count": 0,
                "guardrail_triggered": False,
                "output_flagged": False,
            }

            print("\n" + "-"*60)

            # ── Run Pipeline ─────────────────────────────────────────
            final_state = app.invoke(initial_state)

            # ── Display Answer ───────────────────────────────────────
            print("\n🤖 Assistant:")
            print(final_state["answer"])
            print(f"\n📁 Domain   : {final_state.get('domain', 'N/A')}")
            print(f"🔁 Retries  : {final_state.get('retry_count', 0)}")
            print(f"✅ Validated: {final_state.get('validation_result', 'N/A')}")
            print("-"*60 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n[Chatbot] ❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    run_chatbot()
