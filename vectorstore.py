"""
vectorstore.py — chunk the knowledge base, embed it, and store it in
ChromaDB so we can search it by *meaning* rather than by exact words.

WHY CHROMA, AND WHY OPENAI EMBEDDINGS SPECIFICALLY
Chroma is a vector database: it stores each chunk of text alongside a
vector (a list of a few hundred numbers) that represents its meaning, and
can find the stored vectors closest to a new query's vector almost
instantly. Turning text into that vector is called "embedding" it.

Anthropic does not currently offer an embeddings API — only OpenAI does
of the two providers this programme uses. So this file always uses
OpenAI's embedding model for the vector step, even on a run where you are
chatting with Claude. That means OPENAI_API_KEY must be set in your .env
file no matter which provider you pick for the chat/agent step in
02_basic_rag.py and app.py. See README.md, Step 3.

PUBLIC SURFACE:
    build_or_load_collection(reset=False) -> a Chroma collection, ready to query
    query(question, n_results=3) -> list of {text, source, distance}
    format_hits(hits) -> one context string, ready to paste into a prompt
"""

from __future__ import annotations

import glob
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_store")
KNOWLEDGE_BASE_DIR = os.path.join(HERE, "knowledge_base")
COLLECTION_NAME = "lloyds_policies"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _chunk_document(path: str, max_chars: int = 700) -> list[str]:
    """
    Split one knowledge-base file into paragraph-sized chunks. We chunk on
    blank lines (each policy document is already written in short numbered
    sections) and only hard-cut a single paragraph if it is unusually long.

    Chunk size is a real design decision in RAG, not an implementation
    detail: a chunk that is too big drowns out its own best sentence when
    embedded; a chunk that is too small loses the context around it.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
    return chunks


def get_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The vector store needs it to create "
            "embeddings, even if you plan to chat with Claude — Anthropic has "
            "no embeddings API. Add OPENAI_API_KEY to your .env file. "
            "See README.md, Step 3."
        )
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)


def build_or_load_collection(reset: bool = False):
    """
    Create the Chroma collection on first run (embedding and storing every
    chunk), or simply reopen it on later runs — embedding costs a small
    amount of money and time per chunk, so we do it once and persist the
    result to disk under chroma_store/.

    Pass reset=True to wipe and rebuild from scratch, e.g. after editing
    the knowledge_base/ files.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a first run

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

    if collection.count() == 0:
        ids, docs, metas = [], [], []
        paths = sorted(glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt")))
        if not paths:
            raise ValueError(f"No .txt files found in {KNOWLEDGE_BASE_DIR!r}.")
        for path in paths:
            source = os.path.basename(path)
            for i, chunk in enumerate(_chunk_document(path)):
                ids.append(f"{source}::{i}")
                docs.append(chunk)
                metas.append({"source": source})
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"(Embedded and stored {len(docs)} chunks from {len(paths)} document(s) in {PERSIST_DIR})")

    return collection


def query(question: str, n_results: int = 3, collection=None) -> list[dict]:
    """
    Return the n_results chunks whose embeddings are closest to the
    question's embedding, each as {"text", "source", "distance"}.

    IMPORTANT: this is a DISTANCE, not a similarity score — lower means
    more alike (0 would be an exact match), the opposite of the 0-to-1
    "higher is better" similarity scores you may have seen elsewhere.
    """
    collection = collection or build_or_load_collection()
    result = collection.query(query_texts=[question], n_results=n_results)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits


def format_hits(hits: list[dict], max_distance: float = 1.1) -> str:
    """
    Turn query() results into the context block we hand to the model.
    Hits above max_distance are dropped — a weak match is worse than no
    match, because it makes a wrong answer look sourced. This is the same
    "stale or missing context" failure mode the AgentOps modules from
    Day 2 onward teach you to watch for in production.
    """
    kept = [h for h in hits if h["distance"] <= max_distance]
    if not kept:
        return "(No sufficiently relevant policy document was found for this query.)"
    return "\n\n".join(
        f"[Source: {h['source']} | distance: {h['distance']:.3f}]\n{h['text']}" for h in kept
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the maximum loan to value for a mortgage?"
    print(f"Query: {q}\n")
    for hit in query(q):
        print(f"  distance={hit['distance']:.3f}  [{hit['source']}]")
        print(f"    {hit['text'][:120]}...")
