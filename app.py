"""
app.py — Day 1, Step 3: wrap Steps 1 and 2 in a single agent.

01_vector_embeddings_demo.py showed what an embedding is. 02_basic_rag.py
wired retrieval and generation together by hand. This file hands that same
retrieval step to the MODEL as a tool, and lets it decide when to call it
— which is the one new idea in "agent": the model, not our code, decides
what to do next.

This is deliberately the smallest possible agent: a loop that
1. sends the conversation so far to whichever model provider is configured,
2. checks whether the model asked to call a tool,
3. if so, runs that tool and feeds the result back, and repeats,
4. otherwise prints the model's final answer and waits for the next question.

Every later module in this programme instruments, monitors, or operates a
system built on exactly this loop — so understanding these ~40 lines is
the real foundation for everything from Day 2 onward.

RUN IT:
    python app.py                 interactive chat
    python app.py --demo          runs three scripted example questions
See README.md for setup (the .env file, the two API keys, and so on).
"""


from __future__ import annotations

import argparse
import sys

from llm_provider import chat, get_provider
from tools import TOOL_FUNCTIONS, TOOL_SPECS

SYSTEM_PROMPT = (
    "You are a customer service assistant for Lloyds Financial (a training "
    "scenario, not a real bank). Answer questions about mortgages, "
    "overdrafts, fraud handling, ISAs and savings using the lookup_policy "
    "tool — never answer a policy, rate, or eligibility question from your "
    "own memory, since the tool is the only source of current, correct "
    "figures. Use calculate_monthly_repayment for any repayment or "
    "affordability arithmetic. If lookup_policy returns no good match, say "
    "so plainly rather than guessing. Keep answers short and concrete."
)

MAX_AGENT_STEPS = 6  # a safety cap — see the note in run_agent_turn() below


def run_agent_turn(messages: list[dict]) -> str:
    """
    Run the request/tool-call loop until the model gives a final text
    answer, and return that answer. `messages` is extended in place with
    every step, so the full conversation (including tool calls and their
    results) is visible afterwards for inspection.
    """
    for step in range(MAX_AGENT_STEPS):
        turn = chat(messages, tools=TOOL_SPECS, system=SYSTEM_PROMPT)

        if not turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.text})
            return turn.text

        # The model asked for one or more tools. Record its request, then
        # run each tool and feed the result straight back.
        messages.append({
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls
            ],
        })
        for tc in turn.tool_calls:
            print(f"    \u2192 calling tool: {tc.name}({tc.arguments})")
            fn = TOOL_FUNCTIONS.get(tc.name)
            if fn is None:
                result = f"Error: no such tool '{tc.name}'."
            else:
                try:
                    result = fn(**tc.arguments)
                except Exception as exc:  # a badly-formed call must not crash the loop
                    result = f"Error running {tc.name}: {exc}"
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result,
            })

    # MAX_AGENT_STEPS reached with no final answer: this is exactly the
    # "runaway loop" failure class Day 3 names precisely. Rather than loop
    # forever (and rack up token cost silently), we stop and say so.
    return (
        "(Stopped after repeated tool calls with no final answer \u2014 this is the "
        "runaway-loop pattern Day 3 covers. Try rephrasing the question.)"
    )


def run_demo() -> None:
    demo_questions = [
        "What's the maximum loan-to-value on a mortgage if I'm a first-time buyer?",
        "My card was stolen and used before I noticed \u2014 how much am I liable for?",
        "What would the monthly repayment be on a GBP 220,000 mortgage at 5.1% over 30 years?",
    ]
    for q in demo_questions:
        print(f"\n{'=' * 70}\nCustomer: {q}")
        answer = run_agent_turn([{"role": "user", "content": q}])
        print(f"\nAssistant: {answer}")


def run_repl() -> None:
    print("Lloyds Financial assistant (Day 1 training build). Type 'exit' to quit.")
    messages: list[dict] = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        answer = run_agent_turn(messages)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run three scripted example questions and exit")
    args = parser.parse_args()

    try:
        print(f"(Using provider: {get_provider()})")
    except RuntimeError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        run_demo()
    else:
        run_repl()
