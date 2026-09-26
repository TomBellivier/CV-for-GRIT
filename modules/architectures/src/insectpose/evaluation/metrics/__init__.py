"""Metrics. One metric = one module + one registered name (CONVENTIONS.md §7.2).

Imposed signature: fn(bundle: EvalBundle) -> list[dict] (rows of contract 4).
No metric reads a model, a framework log or a file outside the bundle.
"""
