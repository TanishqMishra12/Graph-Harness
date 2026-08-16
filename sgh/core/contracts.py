"""
Output contract validation for SGH.

Implements §6.1 (Theorem 6.3) — the two-layer validation approach:

  Layer 1: Syntactic validation
    JSON Schema check on the node's output dict. Deterministic and fast.
    A syntactic failure is classified as a contract_violation and triggers
    Level 2 recovery (local_patch) if the retry budget is exhausted.

  Layer 2: Semantic validation (optional, advisory)
    An LLM-as-judge call that assesses whether the output is semantically
    correct given the node's intent. This layer is explicitly marked
    unreliable — the "validation gap" from Theorem 6.3 means that semantic
    validation cannot provide the same guarantees as syntactic validation.
    See docs/LIMITATIONS.md §2 for the full discussion.

The split between syntactic (deterministic) and semantic (probabilistic)
validation is a first-class design decision in the paper. This module
preserves that distinction clearly so callers can treat the two results
differently.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import jsonschema
import jsonschema.exceptions

from sgh.core.plan import OutputContract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ContractResult:
    """Result of a contract validation check."""

    __slots__ = ("passed", "layer", "message", "details")

    def __init__(
        self,
        passed: bool,
        layer: str,         # "syntactic" | "semantic" | "none"
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.passed = passed
        self.layer = layer
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"ContractResult(passed={self.passed}, layer={self.layer!r}, msg={self.message!r})"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ContractValidator:
    """
    Validates node outputs against their OutputContract (κ_v).

    Usage:
        validator = ContractValidator()
        result = validator.check(output, contract)
        if not result.passed:
            # trigger recovery
    """

    def check(
        self,
        output: dict[str, Any],
        contract: OutputContract,
        llm_client: Any | None = None,
        node_label: str = "<node>",
    ) -> ContractResult:
        """
        Run validation layers in order: syntactic first, then semantic (if enabled).

        Syntactic failure short-circuits — semantic is not attempted if syntactic fails.

        Args:
            output:     The node's output dict to validate.
            contract:   The per-node output contract κ_v.
            llm_client: An Anthropic/OpenAI client for semantic validation.
                        Required if contract.semantic_check=True. Ignored otherwise.
            node_label: Human-readable node name for log messages.

        Returns:
            ContractResult with passed=True iff all enabled checks pass.
        """
        if contract.json_schema is None and not contract.semantic_check:
            # No contract configured — pass trivially
            return ContractResult(passed=True, layer="none", message="No contract configured.")

        if contract.json_schema is not None:
            syntactic = self._syntactic_check(output, contract.json_schema, node_label)
            if not syntactic.passed:
                return syntactic

        if contract.semantic_check:
            if llm_client is None:
                logger.warning(
                    "Node %r has semantic_check=True but no llm_client was provided. "
                    "Semantic check skipped.",
                    node_label,
                )
                return ContractResult(
                    passed=True,
                    layer="semantic",
                    message="Semantic check skipped (no LLM client).",
                )
            return self._semantic_check(output, contract, llm_client, node_label)

        return ContractResult(passed=True, layer="syntactic", message="Syntactic check passed.")

    # ------------------------------------------------------------------
    # Layer 1: Syntactic
    # ------------------------------------------------------------------

    def _syntactic_check(
        self,
        output: dict[str, Any],
        schema: dict[str, Any],
        node_label: str,
    ) -> ContractResult:
        """
        Validate `output` against a JSON Schema dict.

        This is always deterministic: the same output + schema always yields
        the same result.
        """
        try:
            jsonschema.validate(instance=output, schema=schema)
            return ContractResult(
                passed=True,
                layer="syntactic",
                message="JSON Schema validation passed.",
            )
        except jsonschema.exceptions.ValidationError as exc:
            return ContractResult(
                passed=False,
                layer="syntactic",
                message=f"JSON Schema validation failed for node {node_label!r}: {exc.message}",
                details={
                    "path": list(exc.absolute_path),
                    "schema_path": list(exc.absolute_schema_path),
                    "validator": exc.validator,
                    "validator_value": exc.validator_value,
                },
            )
        except jsonschema.exceptions.SchemaError as exc:
            # The schema itself is malformed — this is a plan authoring error
            return ContractResult(
                passed=False,
                layer="syntactic",
                message=f"Malformed JSON Schema in contract for node {node_label!r}: {exc.message}",
                details={"schema_error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Layer 2: Semantic (LLM-as-judge)
    # ------------------------------------------------------------------

    def _semantic_check(
        self,
        output: dict[str, Any],
        contract: OutputContract,
        llm_client: Any,
        node_label: str,
    ) -> ContractResult:
        """
        Advisory semantic check using an LLM-as-judge.

        IMPORTANT: This is explicitly unreliable (the validation gap, Theorem 6.3).
        The LLM judge may pass bad outputs or fail good ones. Callers should treat
        this result as a soft signal, not a hard guarantee.

        The judge receives the raw output serialised as JSON and is asked to respond
        with a JSON object: {"pass": true/false, "reason": "..."}.
        """
        try:
            output_json = json.dumps(output, indent=2)
            prompt = (
                f"You are a contract validator for an AI agent node named '{node_label}'.\n\n"
                f"The node produced the following output:\n```json\n{output_json}\n```\n\n"
                "Evaluate whether this output is semantically correct and useful for the task. "
                "Respond with a JSON object with exactly two keys:\n"
                '  - "pass": true if the output is acceptable, false otherwise\n'
                '  - "reason": a one-sentence explanation\n\n'
                "Respond with ONLY the JSON object, no other text."
            )

            # Support both Anthropic and OpenAI clients via duck-typing
            if hasattr(llm_client, "messages"):
                # Anthropic
                response = llm_client.messages.create(
                    model="claude-haiku-4-5",  # Use cheapest model for judge
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
            else:
                # OpenAI-compatible
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.choices[0].message.content

            result = json.loads(raw)
            passed = bool(result.get("pass", False))
            reason = str(result.get("reason", ""))

            logger.debug(
                "Semantic check for node %r: passed=%s, reason=%r",
                node_label,
                passed,
                reason,
            )
            return ContractResult(
                passed=passed,
                layer="semantic",
                message=reason,
                details={"raw_judge_response": raw},
            )

        except Exception as exc:  # noqa: BLE001
            # Never let a semantic check exception propagate — it's advisory
            logger.warning(
                "Semantic check for node %r raised an exception (%s: %s). "
                "Treating as pass (advisory only). See LIMITATIONS.md §2.",
                node_label,
                type(exc).__name__,
                exc,
            )
            return ContractResult(
                passed=True,
                layer="semantic",
                message=f"Semantic check failed with exception (treated as pass): {exc}",
                details={"exception": str(exc)},
            )
