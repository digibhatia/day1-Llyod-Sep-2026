"""
02_basic_rag.py — Day 1, Step 2: RAG on its own, before any agent.

"RAG" (Retrieval-Augmented Generation) is just two steps glued together:
  1. RETRIEVE the most relevant chunks from the vector store (Step 1's job)
  2. GENERATE an answer by asking the model to answer USING those chunks

There is no agent here yet — no tool-calling, no decision by the model
about what to do next. We hand-wire the retrieve-then-generate sequence
ourselves, in plain Python, so the shape of RAG is visible before app.py
hides the same two steps behind a tool the model calls on its own.

RUN IT:
    python 02_basic_rag.py "How much does it cost to go over my agreed limit?"
    python 02_basic_rag.py           (uses a built-in example question)
"""

from __future__ import annotations

import sys

from llm_provider import chat, get_provider
from vectorstore import format_hits, query

RAG_PROMPT_TEMPLATE = """Answer the customer's question using ONLY the policy \
context below. If the context does not contain the answer, say so plainly \
rather than guessing.

POLICY CONTEXT:
{context}

CUSTOMER QUESTION: {question}"""


def answer_with_rag(question: str) -> str:
    # Step 1: retrieve.
    hits = query(question, n_results=3)
    context = format_hits(hits)
    print("--- Retrieved context (this is what the model actually sees) ---")
    print(context)
    print("-" * 66)

    # Step 2: generate, grounded in that context. No tools involved —
    # just one plain user message built from a template.
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    turn = chat([{"role": "user", "content": prompt}])
    return turn.text


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much does it cost to go over my agreed overdraft limit?"
    print(f"(Using provider: {get_provider()})")
    print(f"Question: {question}\n")
    print(f"Answer: {answer_with_rag(question)}")
