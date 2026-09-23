"""Shared domain vocabulary used by every Numi service.

Nothing in this package talks to a database, an LLM, or a network. It only
defines the types that let the Agent, Gateway, and Execution Service agree on
what a "tool call", a "target", a "risk", or a "failure" means, without any of
them trusting each other's interpretation of those types.
"""
