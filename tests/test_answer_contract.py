"""The answer contract: in the server instructions and in the MCP prompt."""

from __future__ import annotations

import re

import pytest

from stratakb.mcp_server import _ANSWER_CONTRACT

PROMPT = "answer_from_documents"


async def _prompt_text(server, **arguments) -> str:
    result = await server.get_prompt(PROMPT, arguments)
    return result.messages[0].content.text


def test_instructions_length_and_phrases(server):
    instructions = server.instructions
    assert len(instructions) <= 4000
    assert instructions.index(_ANSWER_CONTRACT[0]) < 500
    for phrase in ("strata_verify", "receipt", "verbatim", "not in the documents", "not_found"):
        assert phrase in instructions
    for rule in _ANSWER_CONTRACT:
        assert rule in instructions


async def test_named_tools_exist(server):
    named = set(re.findall(r"strata_\w+", server.instructions))
    assert named
    assert named <= {tool.name for tool in await server.list_tools()}


async def test_prompt_listed(server):
    prompts = {prompt.name: prompt for prompt in await server.list_prompts()}
    arguments = {arg.name: arg.required for arg in prompts[PROMPT].arguments}
    assert arguments == {"question": True, "layers": False}


async def test_prompt_get_fills_arguments(server):
    text = await _prompt_text(server, question="How do we fail over?", layers="a, b,")
    assert "How do we fail over?" in text
    assert 'layers=["a", "b"]' in text
    for rule in _ANSWER_CONTRACT:
        assert rule in text


@pytest.mark.parametrize("arguments", [{}, {"layers": ""}, {"layers": " , "}])
async def test_prompt_without_layers_names_none(server, arguments):
    text = await _prompt_text(server, question="How?", **arguments)
    assert "layers=" not in text


async def test_prompt_missing_question_refused(server):
    with pytest.raises(ValueError, match="question"):
        await server.get_prompt(PROMPT, {})
