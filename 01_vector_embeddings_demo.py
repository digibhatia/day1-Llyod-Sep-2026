"""
01_vector_embeddings_demo.py — Day 1, Step 1: what is an embedding, really?

Run this BEFORE looking at any agent code. It does two things, in order,
with nothing else going on so each is easy to see clearly:

  PART A — turns a handful of plain sentences into embedding vectors and
  shows that sentences with similar MEANING end up with similar vectors,
  even when they don't share many of the same words.

  PART B — builds the real Chroma vector store from the Lloyds Financial
  knowledge_base/ folder (embedding and storing every chunk, once) and
  runs a few example searches against it, printing what comes back.

No agent, no tool-calling, no chat loop yet — just embeddings and vector
search on their own. 02_basic_rag.py adds a single LLM call on top of this;
app.py adds the full tool-calling agent on top of that.

RUN IT:
    python 01_vector_embeddings_demo.py
"""

from __future__ import annotations

import numpy as np

from vectorstore import build_or_load_collection, get_embedding_function, query


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def part_a_embeddings_by_hand() -> None:
    print("=" * 70)
    print("PART A — three sentences, embedded, compared")
    print("=" * 70)

    sentences = [
        "What's the interest rate on my overdraft?",          # 0
        "How much does it cost to go over my agreed limit?",  # 1: same meaning as 0, different words
        "What time does the Leeds branch close on Saturday?", # 2: unrelated meaning
    ]
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")

    embed = get_embedding_function()
    vectors = embed(sentences)  # one call, one vector per sentence

    dims = len(vectors[0])
    print(f"\nEach sentence became a vector of {dims} numbers. First 8 numbers of sentence [0]:")
    print(" ", [round(x, 4) for x in vectors[0][:8]], "...")

    sim_0_1 = cosine_similarity(vectors[0], vectors[1])
    sim_0_2 = cosine_similarity(vectors[0], vectors[2])
    print(f"\nSimilarity([0] overdraft rate, [1] cost of going over limit) = {sim_0_1:.3f}  <- same topic, different words")
    print(f"Similarity([0] overdraft rate, [2] branch opening hours)      = {sim_0_2:.3f}  <- unrelated topic")
    print(
        "\nThis is the whole idea of a vector search: sentence [1] scores higher "
        "than [2] even though it does not share the word 'overdraft' or 'rate' "
        "with sentence [0] at all — the embedding captured the MEANING, not "
        "just the words."
    )


def part_b_build_and_search_the_real_store() -> None:
    print("\n" + "=" * 70)
    print("PART B — building the Lloyds Financial vector store")
    print("=" * 70)
    collection = build_or_load_collection()
    print(f"Collection '{collection.name}' now holds {collection.count()} embedded chunks.\n")

    demo_queries = [
        "How much does it cost to go over my agreed limit?",
        "Can I take money out of my fixed rate ISA early?",
        "What's the weather like in Leeds today?",  # deliberately irrelevant — watch the distances
    ]
    for q in demo_queries:
        print(f"Query: {q!r}")
        for hit in query(q, n_results=2, collection=collection):
            print(f"    distance={hit['distance']:.3f}  [{hit['source']}]  {hit['text'][:90]}...")
        print()

    print(
        "Notice the third query's distances are much larger (worse) than the "
        "first two — there is no branch-hours document in the knowledge base, "
        "and the vector search correctly cannot find a close match for it. "
        "format_hits() in vectorstore.py uses exactly this distance to decide "
        "when to say 'no good match found' instead of guessing."
    )


if __name__ == "__main__":
    part_a_embeddings_by_hand()
    part_b_build_and_search_the_real_store()
