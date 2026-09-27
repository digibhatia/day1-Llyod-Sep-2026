"""
tools.py — the two tools our single agent is allowed to call.

A "tool" here is just a Python function, described to the model with a
name, a plain-English description, and a JSON schema for its arguments.
The model never runs the function itself — it can only ask for it to be
run, by name, with arguments. Our job (in app.py's agent loop) is to take
that request, actually call the function, and hand the result back.

This file deliberately knows nothing about Anthropic or OpenAI. The tool
*description* below (TOOL_SPECS) is provider-neutral; llm_provider.py is
the only file that translates it into each provider's own wire format.
That separation is the whole point of the pattern this Day 1 solution
sets up: one tool, one description, callable from either provider.
"""

from __future__ import annotations

from vectorstore import format_hits, query as vector_query


def lookup_policy(query: str) -> str:
    """
    Retrieval tool: search the (synthetic) Lloyds Financial policy
    knowledge base — stored as vector embeddings in ChromaDB — for the
    passage most relevant to `query`.
    """
    hits = vector_query(query, n_results=3)
    return format_hits(hits)



def calculate_monthly_repayment(principal: float, annual_rate_percent: float, term_years: int) -> str:
    """
    Calculator tool: standard repayment-mortgage monthly payment formula.
    Kept separate from lookup_policy on purpose — a single agent commonly
    mixes a retrieval tool with an ordinary calculation tool, and the two
    have very different failure modes (retrieval can return a stale or
    irrelevant document; a calculator can be called with the wrong or
    malformed numbers). Day 3's failure taxonomy revisits both.
    """
    principal = float(principal)
    annual_rate_percent = float(annual_rate_percent)
    term_years = int(term_years)

    if principal <= 0 or term_years <= 0:
        return "Error: principal and term_years must both be greater than zero."
    if annual_rate_percent < 0:
        return "Error: annual_rate_percent cannot be negative."

    monthly_rate = (annual_rate_percent / 100) / 12
    n_payments = term_years * 12

    if monthly_rate == 0:
        payment = principal / n_payments
    else:
        payment = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** n_payments
            / ((1 + monthly_rate) ** n_payments - 1)
        )

    total_paid = payment * n_payments
    total_interest = total_paid - principal
    return (
        f"Estimated monthly repayment: GBP {payment:,.2f}\n"
        f"Over {term_years} years ({n_payments} monthly payments) at {annual_rate_percent}% per year.\n"
        f"Total repaid: GBP {total_paid:,.2f}  |  Total interest: GBP {total_interest:,.2f}\n"
        "(Illustrative figure only — ignores fees, overpayments, and rate changes after any fixed period.)"
    )


# ---------------------------------------------------------------------------
# Provider-neutral tool descriptions. Both Anthropic and OpenAI expect a
# JSON-schema "parameters"/"input_schema" object; the shape below is that
# same JSON schema, kept once and translated per provider in llm_provider.py.
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "lookup_policy",
        "description": (
            "Search the Lloyds Financial policy knowledge base (mortgages, "
            "overdrafts, fraud handling, ISAs and savings) for the passage "
            "most relevant to a customer's question. Always use this before "
            "answering any question about rates, limits, eligibility, or "
            "procedure — never answer those from memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question, or the specific fact you need to check.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_monthly_repayment",
        "description": (
            "Calculate an illustrative monthly repayment for a repayment "
            "(capital and interest) mortgage or loan, given the loan amount, "
            "the annual interest rate, and the term in years."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in GBP."},
                "annual_rate_percent": {"type": "number", "description": "Annual interest rate, e.g. 5.25 for 5.25%."},
                "term_years": {"type": "integer", "description": "Loan term in whole years."},
            },
            "required": ["principal", "annual_rate_percent", "term_years"],
        },
    },
]

# Maps a tool name to the Python function that actually performs it.
TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_monthly_repayment": calculate_monthly_repayment,
}


if __name__ == "__main__":
    print(calculate_monthly_repayment(250_000, 5.25, 25))
