from typing import (
    Callable,
    List,
    Any,
    Union,
    Type,
    Awaitable,
)
import json

from pydantic import BaseModel

from . import (
    build_simple_agent,
    AgentCallable,
    BoundAgentCallable,
    Model,
    EventCallback,
    EventPayload,
    AgentRun,
    flat_messages,
    noop_callback,
    recursive_extract_messages,
)


def simple_text_callback_binder(simple_text_callback: Callable[[str], Awaitable[None]]) -> EventCallback:
    async def bound_simple_text_callback(event: EventPayload) -> None:
        if event[0] == 'ai' and event[1] == 'text_event':
            await simple_text_callback(event[2])
    return bound_simple_text_callback


async def simple_ask(
    binder: Callable[[AgentCallable], BoundAgentCallable],
    system_prompt: str,
    human_prompt: str,
    streaming_callback: Union[Callable[[str], Awaitable[None]], None] = None,
    tools: List[Callable] = [],
    force_tool: bool = False,
    max_tool_iters: int = 5,
) -> Any:
    agent = binder(
        build_simple_agent(
            name = "simple_ask",
            tools = tools,
            force_tool = force_tool,
            max_tool_iters = max_tool_iters,
        )
    )
    runs: List[AgentRun] = [
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
    assert response['type'] == 'messages'
    last_message = response["messages"][-1]
    if last_message["role"] == "ai":
        return last_message["text"]
    elif last_message["role"] == "tool_res":
        last_tool = last_message["tools"][-1]
        if last_tool["type"] == "any":
            return last_tool["result"]
        elif last_tool["type"] == "layered_agent":
            raise RuntimeError("layered agent tool results are not supported in simple_ask")
        else:
            raise RuntimeError(f"unknown tool result type: {last_tool['type']}")
    else:
        raise RuntimeError(f"unexpected message role: {last_message['role']}")


"""
Strucuted output stuff:
"""

def extract_prompt_text_from_pydantic_model(
    extraction_type: Type[BaseModel],
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
    extraction_type: Type[BaseModel],
) -> AgentCallable:
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
    binder: Callable[[AgentCallable], BoundAgentCallable],
    system_prompt: str,
    human_prompt: str,
    extraction_type: Type[BaseModel],
    streaming_callback: Union[Callable[[str], Awaitable[None]], None] = None,
) -> BaseModel:
    agent = binder(
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

