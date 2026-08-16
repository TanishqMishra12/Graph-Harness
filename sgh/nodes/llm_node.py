"""
LLM Node Executor for SGH.

Implements the LLM-based execution logic, strictly enforcing the context
partition (receives only ExecContext, never DiagContext).

Flow:
1. Render prompt_template with task, node_label, and upstream_outputs.
2. If the prompt contains a RECOVERY PATCH (added by local_patch), it is automatically included.
3. Call LLM API via litellm (structured output or raw text).
4. Run syntactic validation (JSON Schema) if defined in contract.
5. Return NodeOutput.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import litellm
from jsonschema import ValidationError, validate

from sgh.nodes.base import BaseNode, ExecContext, NodeOutput

logger = logging.getLogger(__name__)


class LLMNode(BaseNode):
    """
    Executes a node using an LLM.

    The client is injected, or created with default credentials if None.
    """

    def __init__(self, node: Any, kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(node)
        self.kwargs = kwargs or {}

    async def execute(self, ctx: ExecContext) -> NodeOutput:
        # 1. Prepare prompt
        template = ctx.config.prompt_template or "Solve the task: {task}"
        
        # Render prompt using safe formatting
        # We pass task, node_label, and upstream_outputs.
        # To avoid KeyError for missing vars, we do a basic replacement.
        prompt = template
        prompt = prompt.replace("{task}", str(ctx.task_description or ""))
        prompt = prompt.replace("{node_label}", ctx.node_label)
        
        # Inject upstream outputs as a formatted string
        if "{upstream_outputs}" in prompt:
            if ctx.upstream_outputs:
                upstream_str = json.dumps(ctx.upstream_outputs, indent=2)
            else:
                upstream_str = "{}"
            prompt = prompt.replace("{upstream_outputs}", upstream_str)

        # 2. Prepare API call
        # If there's a JSON schema contract, we ask the LLM to return JSON
        # and we can use Anthropic's tools or just ask for a JSON block.
        # For simplicity in this engine, if there's a schema, we ask for JSON directly.
        system_prompt = "You are a helpful assistant executing a node in a workflow DAG."
        if ctx.config.contract.json_schema:
            system_prompt += (
                "\n\nOUTPUT CONTRACT: You MUST output strictly valid JSON matching the following schema. "
                "Do NOT wrap it in markdown block quotes (```json), just output the raw JSON object.\n"
                f"{json.dumps(ctx.config.contract.json_schema, indent=2)}"
            )

        try:
            response = await litellm.acompletion(
                model=ctx.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4096,
                **self.kwargs
            )
        except Exception as e:
            # Infrastructure failure -> TRANSIENT
            return NodeOutput(
                outcome="retry",
                error_message=f"LLM API Error: {e}",
                failure_type="transient"
            )

        text_output = response.choices[0].message.content.strip()
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        # 3. Validation
        payload = {}
        if ctx.config.contract.json_schema:
            try:
                payload = json.loads(text_output)
            except json.JSONDecodeError as e:
                return NodeOutput(
                    outcome="failure",
                    error_message=f"Invalid JSON returned: {e}. Output was: {text_output[:100]}...",
                    failure_type="contract_violation",
                    token_usage=usage
                )
            
            try:
                validate(instance=payload, schema=ctx.config.contract.json_schema)
            except ValidationError as e:
                return NodeOutput(
                    outcome="failure",
                    error_message=f"JSON Schema validation failed: {e.message}",
                    failure_type="contract_violation",
                    payload=payload,  # partial/invalid payload
                    token_usage=usage
                )
        else:
            # No JSON schema -> output is just text
            payload = {"text": text_output}

        # Success
        return NodeOutput(
            outcome="success",
            payload=payload,
            token_usage=usage
        )
