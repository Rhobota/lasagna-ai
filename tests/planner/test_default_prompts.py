import pytest

from lasagna.planner.default_prompts import (
    PlanOutput,
    subtask_input_generator,
    answer_input_prompt,
    append_tool_spec_to_prompt,
)

from lasagna.agent_util import extraction

from lasagna.types import AgentRun

from typing import List


PLAN_OBJ = PlanOutput(
    thoughts = '... hum ...',
    task_statement = 'do the thing',
    is_trivial = False,
    subtasks = ['first thing', 'next thing', 'last thing'],
)

EXTRACTION_RUN = extraction('test', [], PLAN_OBJ)


@pytest.mark.asyncio
async def test_subtask_input_generator():
    human_inputs: List[List[AgentRun]] = []
    async for human_input in subtask_input_generator(EXTRACTION_RUN):
        human_inputs.append(human_input)
    assert len(human_inputs) == 3
    for task_str, human_input in zip(PLAN_OBJ.subtasks, human_inputs):
        assert len(human_input) == 1
        single_input = human_input[0]
        assert single_input['type'] == 'messages'
        messages = single_input['messages']
        assert len(messages) == 1
        message = messages[0]
        assert message['role'] == 'human'
        assert message['text']
        assert task_str in message['text']


@pytest.mark.asyncio
async def test_answer_input_prompt():
    human_input = await answer_input_prompt(EXTRACTION_RUN)
    assert len(human_input) == 1
    single_input = human_input[0]
    assert single_input['type'] == 'messages'
    messages = single_input['messages']
    assert len(messages) == 1
    message = messages[0]
    assert message['role'] == 'human'
    assert message['text']
    assert PLAN_OBJ.task_statement in message['text']


def test_append_tool_spec_to_prompt():
    # No tools:
    new_prompt = append_tool_spec_to_prompt('do well', tools = [])
    assert new_prompt == 'do well'

    # With tools:
    def foo(a: int, b: int) -> float:
        """
        Divide b into a.
        :param: a: int: numerator
        :param: b: int: denominator
        """
        return a / b
    new_prompt = append_tool_spec_to_prompt('do well', tools = [
        foo,
    ])
    assert 'do well' in new_prompt
    assert 'foo' in new_prompt
    assert 'Divide b into a.' in new_prompt
    assert 'numerator' in new_prompt
    assert 'denominator' in new_prompt
    assert 'BLABLABLA' not in new_prompt
