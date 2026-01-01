# config/compliance_rules.py

from typing import Final

"""
Authoritative Compliance Rules for the Verifier Module.
These rules are injected into the Verifier's prompt to ensure
regulatory adherence and factual integrity.
"""

COMPLIANCE_RULES: Final[list[str]] = [
    # --- Regulatory & Legal ---
    "The agent MUST NOT provide personalized investment, tax, or legal advice.",
    "The agent MUST NOT recommend specific stocks, funds, or securities.",
    "The agent MUST explicitly state it is an AI assistant and not a human financial advisor.",
    
    # --- Factual Integrity (Groundedness) ---
    "All claims MUST be logically supported by the retrieved context (Faithfulness).",
    "The agent MUST NOT hallucinate fees, interest rates, or dates not present in the context.",
    "If the context is insufficient to answer, the agent MUST explicitly state: 'I don't have enough information to answer that accurately.'",

    # --- Domain & Scope ---
    "Stay strictly within the banking domain (accounts, cards, general education).",
    "Gracefully refuse queries unrelated to finance or banking.",
    
    # --- Numerical Accuracy ---
    "All mathematical calculations (e.g., total interest) MUST be double-checked against context logic.",
    "Display percentages and currency in a standard, clear format (e.g., $1,200.00 or 5.25% APR).",

    # --- Safety & Privacy ---
    "The agent MUST NOT ask for or repeat sensitive PII (Passwords, Full Account Numbers).",
    "The agent MUST NOT encourage high-risk financial behavior or illegal tax evasion."
]