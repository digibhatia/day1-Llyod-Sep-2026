# Day 1 — RAG, Vector Embeddings, and Your First Agent

**Scenario:** Lloyds Financial (a training scenario — not a real bank or real
policy) wants a customer-service assistant that answers questions about
mortgages, overdrafts, fraud handling, ISAs and savings, using its own
policy documents rather than whatever the model happens to already "know."

Today builds that assistant in three small, separate steps, so each idea
is visible on its own before the next one is stacked on top of it:

| Step | File | What it shows |
|---|---|---|
| 1 | `01_vector_embeddings_demo.py` | What an embedding is, and how vector search finds meaning, not just matching words |
| 2 | `02_basic_rag.py` | RAG on its own: retrieve, then generate — no agent yet |
| 3 | `app.py` | The full single agent: the model itself decides when to call the retrieval tool |

Every later module in this programme (starting Day 2) instruments,
monitors, or operates a system built on exactly this pattern — so today is
the foundation the rest of the 60 hours sits on.

**This build uses OpenAI only** — one `OPENAI_API_KEY` covers both the chat/agent
step and the embeddings step, so there is only one key to manage today.

---

## Step 0 — What's in this folder

day1/
knowledge_base/ 4 synthetic Lloyds Financial policy documents
vectorstore.py chunks + embeds the knowledge base into Chroma
tools.py the two tools the agent can call
llm_provider.py calls the OpenAI model for chat/agent turns
01_vector_embeddings_demo.py Step 1
02_basic_rag.py Step 2
app.py Step 3 (the main deliverable)
.env.example copy this to .env and fill in your key
requirements.txt


The `knowledge_base/` documents are clearly labelled as synthetic training
content — they are not real Lloyds policy, rates, or figures. Do not put
any real customer data, real policy documents, or anything confidential
into this folder or into any prompt.

---

## Step 1 — Install and set up (10 minutes)

1. **Create a virtual environment** (recommended, keeps this separate from
   anything else on your machine):

python -m venv .venv

   Activate it:
   - Windows: `.venv\Scripts\Activate.ps1`
   - Mac/Linux: `source .venv/bin/activate`

2. **Install the dependencies:**

pip install -r requirements.txt


3. **Create your `.env` file:**

cp .env.example .env

   Open `.env` and paste in your `OPENAI_API_KEY`. That one key covers
   everything today: the chat/agent step in `02_basic_rag.py` and `app.py`,
   and the vector-embedding step in all three scripts.

4. **Check your setup** by running Step 1's demo (see below). If it
   prints vector numbers and search results without an error, you're set
   up correctly.

---

## Step 2 — Run Step 1: what is an embedding? (`01_vector_embeddings_demo.py`)

python 01_vector_embeddings_demo.py


**What it does, and why, in order:**

- **Part A** turns three plain sentences into embedding vectors (a list of
  numbers representing meaning) and compares them. Two of the sentences
  ask about the same thing in different words; the third asks about
  something unrelated. The point: a vector search can find the sentence
  that means the same thing even when it shares almost no words with the
  query — which is exactly what a customer's real, messy phrasing needs.
- **Part B** builds the real vector store from `knowledge_base/*.txt` (this
  costs a small amount of money and a few seconds — it only happens once;
  the result is saved to `chroma_store/` and reused on every later run,
  including by `02_basic_rag.py` and `app.py`) and runs three example
  searches, including one deliberately irrelevant question, so you can see
  what a "no good match" case looks like in the raw distance numbers
  before any answer-generation is layered on top.

**Try this:** open `vectorstore.py` and change `max_chars` in
`_chunk_document` from 700 to something much smaller, like 150. Delete the
`chroma_store/` folder and re-run the script. Do the search results get
better or worse? This is the "chunk size" decision every real RAG system
has to make.

---

## Step 3 — Run Step 2: basic RAG (`02_basic_rag.py`)

python 02_basic_rag.py "How much does it cost to go over my agreed overdraft limit?"


Or just `python 02_basic_rag.py` to use a built-in example question.

**What it does:** exactly two steps, wired together by hand in plain
Python so you can see the whole pipeline in about 15 lines
(`answer_with_rag()` in the file):
1. `vectorstore.query()` retrieves the closest chunks (Step 1's machinery).
2. That context is pasted into a prompt template, and `llm_provider.chat()`
   is called **once**, with no tools — the model is simply asked to answer
   using the text it was given.

The script prints the retrieved context **before** the answer, on purpose
— in a real incident, "what did the model actually see" is the first
question you need to answer, and RAG systems that hide this are much
harder to debug.

**Try this:** ask a question with no good answer in the knowledge base
(e.g. `"What are your branch opening hours in Leeds?"`). Watch the
retrieved context degrade to the "no sufficiently relevant document"
message, and see whether the model still tries to guess an answer anyway
despite the prompt's instruction not to — this is a live look at the
"hallucination" failure class the programme names precisely later.

---

## Step 4 — Run Step 3: the single agent (`app.py`)

python app.py --demo


This runs three scripted example questions end to end. For an interactive
chat instead:

python app.py


Type a question, or `exit` to quit.

**What's new here, compared to Step 2:** the model is no longer just
answering from a context we handed it — it decides, on its own, when to
call `lookup_policy` (the same retrieval you just watched happen by hand)
and when to call the second tool, `calculate_monthly_repayment`, for
arithmetic. Every time it calls a tool, the script prints
`→ calling tool: ...` so you can see the decision as it happens.

Open `app.py` and read `run_agent_turn()` — at under 40 lines, this loop
IS the agent. Everything else in this file is either the system prompt
(the instructions telling the model when to use which tool) or plumbing
around printing and the chat REPL.

**Note the safety cap:** `MAX_AGENT_STEPS` stops the loop after 6 rounds
of tool calls with no final answer, rather than looping (and spending
tokens) forever. Later in the programme, this exact failure class gets a
name: a **runaway loop**.

---

## Step 5 — How `llm_provider.py` fits in

Open `llm_provider.py`. It exposes one function the other two scripts call:
`chat(messages, tools=None, system=None)`. Internally it builds an OpenAI
client from `OPENAI_API_KEY`, sends the conversation and any tool
definitions, and returns a plain `AgentTurn` object with `.text`,
`.tool_calls`, and `.stop_reason` — so `app.py`'s loop only ever deals with
that one simple shape, never the raw SDK response.

Keeping that translation in one small file, separate from `app.py`'s own
logic, is the habit worth noticing today: it's the same reason a later
module can swap in a different model or provider without touching the
agent loop itself.

---

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `RuntimeError: No model key found` | `OPENAI_API_KEY` is not set in `.env` — check you ran `cp .env.example .env` and actually filled it in, not just the example file |
| `ModuleNotFoundError` for openai/chromadb | Re-run `pip install -r requirements.txt` inside your activated virtual environment |
| Vector search results look wrong/unrelated | Delete the `chroma_store/` folder and re-run — this forces a fresh rebuild, useful after editing `knowledge_base/` files |
| A tool call fails with a Python error | `app.py`'s loop catches this and feeds the error back to the model as the tool result, so it can try again rather than crashing — check the printed `→ calling tool:` line to see what arguments it sent |
| `AuthenticationError` from OpenAI | Your key is missing, expired, or mistyped in `.env` — check for stray spaces or quotes around the value |

---

## What's next

Day 2 picks up the AgentOps and AI Observability syllabus proper: the
anatomy of a model call, and where tool loops spend their time — using
the same reference agent pattern you just built today as the running
example every later module instruments, monitors, and operates.