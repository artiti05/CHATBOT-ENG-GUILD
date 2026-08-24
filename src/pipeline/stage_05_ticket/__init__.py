"""Stage 05 — Ticket Intake.

Intercepts escalation intent (and the low-confidence fallback) before the
knowledge-base stages run, collects contact details across turns using only
the history the client round-trips, and files a ticket. Resolution after
that point is a purely human process.
"""
from .ticket_agent import TicketIntakeAgent, PROMPT_USER_INTENT, PROMPT_LOW_CONFIDENCE

__all__ = ["TicketIntakeAgent", "PROMPT_USER_INTENT", "PROMPT_LOW_CONFIDENCE"]
