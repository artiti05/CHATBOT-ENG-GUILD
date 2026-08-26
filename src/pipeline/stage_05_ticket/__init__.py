"""Stage 05 — Ticket Intake.

Intercepts escalation intent (explicit request or low-confidence RAG
fallback) before the knowledge-base stages run and files a ticket
immediately, using the caller-supplied user identity. Resolution after
that point is a purely human process.
"""
from .ticket_agent import TicketIntakeAgent

__all__ = ["TicketIntakeAgent"]
