from typing import (
    Callable,
    List,
    Type,
    Any, # FIXME
)
import json

from pydantic import BaseModel

from . import (
    build_simple_agent,
    Model,
    EventCallback,
    EventPayload,
    AgentRun,
    ExtractionType,
    flat_messages,
    noop_callback,
    recursive_extract_messages,
)


def simple_text_callback_binder(simple_text_callback: Callable[[str], None]) -> Callable[[EventPayload], None]:
    async def bound_simple_text_callback(event: EventPayload):
        if event[0] == 'ai' and event[1] == 'text_event':
            await simple_text_callback(event[2])
    return bound_simple_text_callback


async def simple_ask(
    partially_bound_model: Any,
    system_prompt: str,
    human_prompt: str,
    streaming_callback: Callable[[str], None] | None = None,
    tools: List[Callable] = [],
    force_tool: bool = False,
    max_tool_iters: int = 5,
) -> str:
    agent = partially_bound_model(
        build_simple_agent(
            name = "simple_ask",
            tools = tools,
            force_tool = force_tool,
            max_tool_iters = max_tool_iters,
        )
    )
    runs: list[AgentRun] = [
        flat_messages(
            agent_name = "simple_ask",
            messages = [
                {
                    'role': 'system',
                    'text': system_prompt,
                },
                {
                    'role': 'human',
                    'text': human_prompt,
                },
            ],
        )
    ]
    response: AgentRun = await agent(
        simple_text_callback_binder(streaming_callback) if streaming_callback else noop_callback,
        runs,
    )
    last_message = response["messages"][-1]
    if last_message["role"] == "ai":
        return last_message["text"]
    elif last_message["role"] == "tool_res":
        last_message["tools"][-1]["result"]
    else:
        return None # FIXME: Should this be an error or something else?


"""
Strucuted output stuff:
"""

def extract_prompt_text_from_pydantic_model(
    extraction_type: BaseModel,
) -> str:
    """
    Returns json-indented text that's {field: (<type>) <description>}

    A schema property has (description, title, type) keys.
    """
    schema_properties = extraction_type.model_json_schema()['properties']
    for k, v in schema_properties.items():
        if "description" not in v:
            raise ValueError(f'Description not provided for field: {k}')
    return json.dumps(
        {
            k: f'({v["type"]}) {v["description"]}'
            for k, v in schema_properties.items()
        },
        indent=2,
    )


def easy_structured_output_agent_binder(
    extraction_type: Type[ExtractionType],
) -> Callable[[Model, EventCallback, List[AgentRun]], AgentRun]:
    async def easy_structured_output_agent(
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        messages = recursive_extract_messages(prev_runs, from_layered_agents=False)

        message, extraction = await model.extract(
            event_callback = event_callback,
            messages = messages,
            extraction_type = extraction_type,
        )
        assert isinstance(extraction, extraction_type)

        return {
            'agent': 'easy_structured_output_agent',
            'type': 'extraction',
            'messages': [message],
            'result': extraction,
        }
    return easy_structured_output_agent


async def simple_ask_with_structured_output(
    partially_bound_model: Any,
    system_prompt: str,
    human_prompt: str,
    extraction_type: Type[ExtractionType],
    streaming_callback: Callable[[str], None] | None = None,
) -> ExtractionType:
    agent = partially_bound_model(
        easy_structured_output_agent_binder(extraction_type),
    )

    response: AgentRun = await agent(
        simple_text_callback_binder(streaming_callback) if streaming_callback else noop_callback,
        [
            flat_messages(
                agent_name = 'easy_structured_output_agent',
                messages = [
                    {
                        'role': 'system',
                        'text': system_prompt \
                            + "\n\nOutput JSON with the following schema:\n" \
                            + extract_prompt_text_from_pydantic_model(extraction_type),
                    },
                    {
                        'role': 'human',
                        'text': human_prompt,
                    },
                ],
            ),
        ],
    )

    assert response['type'] == 'extraction'
    assert isinstance(response['result'], extraction_type)
    return response['result']

