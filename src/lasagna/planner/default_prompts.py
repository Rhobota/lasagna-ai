from ..types import AgentRun
from ..agent_util import flat_messages
from ..util import get_name
from ..tools_util import (
    convert_to_json_schema,
    get_tool_params,
)

from .util import extract_subtask_strings, extract_task_statement

from pydantic import BaseModel, Field

import json
from typing import List, Dict, AsyncIterable, Callable


PLANNING_AGENT_SYSTEM_PROMPT = """
You plan the execution of a task by:
- analyzing it,
- determining its complexities, and
- (if required) breaking it up into subtasks.
""".strip()


class PlanOutput(BaseModel):
    thoughts: str = Field(description="your free-form thoughts about the user's most recent prompt")
    task_statement: str = Field(description="a concisely-stated task statement, fully describing what the user wants in their most recent prompt")
    is_trivial: bool = Field(description="`true` if this task has a trivial solution (such as recalling a simple fact, or making simple tool calls); `false` if this task needs to be broken up into subtasks, especially when those subtasks are exploratory in their nature")
    subtasks: List[str] = Field(description="an ordered list of subtasks that need to be performed in order to complete the task statement (if not trivial); else an empty list (if trivial)")


async def subtask_input_generator(
    extraction_run: AgentRun,
) -> AsyncIterable[List[AgentRun]]:
    assert extraction_run['type'] == 'extraction'
    extraction_result = extraction_run['result']
    subtask_strings = extract_subtask_strings(extraction_result)
    for i, subtask_string in enumerate(subtask_strings):
        yield [
            flat_messages('subtask_input_generator', [{
                'role': 'human',
                'text': f'Now work on subtask #{i+1}: {subtask_string}',
            }]),
        ]


async def answer_input_prompt(
    extraction_run: AgentRun,
) -> List[AgentRun]:
    assert extraction_run['type'] == 'extraction'
    extraction_result = extraction_run['result']
    task_statement = extract_task_statement(extraction_result)
    return [
        flat_messages('answer_input_prompt', [{
            'role': 'human',
            'text': f"Provide your final answer to the task: {task_statement}",
        }]),
    ]


def append_tool_spec_to_prompt(
    prompt: str,
    tools: List[Callable],
) -> str:
    if not tools:
        return prompt

    tool_specs: List[Dict] = []

    for tool in tools:
        description, params = get_tool_params(tool)
        tool_specs.append({
            'name': get_name(tool),
            'description': description,
            'input_schema': convert_to_json_schema(params),
        })

    sections = [
        prompt,
        "While planning, you may refer to the following tools:",
        json.dumps(tool_specs, indent=2),
    ]

    return '\n\n'.join(sections)
