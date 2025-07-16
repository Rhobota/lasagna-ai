from typing import (
    Callable,
    List,
    Union,
    Type,
    Awaitable,
    Optional,
)
import json

from pydantic import BaseModel

from . import (
    build_simple_agent,
    build_extraction_agent,
    AgentCallable,
    BoundAgentCallable,
    EventCallback,
    EventPayload,
    AgentRun,
    flat_messages,
    noop_callback,
    Message,
)
from .tools_util import (
    extract_tool_result_as_sting,
)


def _build_text_event_callback(simple_text_callback: Callable[[str], Awaitable[None]]) -> EventCallback:
    async def bound_simple_text_callback(event: EventPayload) -> None:
        if event[0] == 'ai' and event[1] == 'text_event':
            await simple_text_callback(event[2])
    return bound_simple_text_callback


async def easy_ask(
    binder: Callable[[AgentCallable], BoundAgentCallable],
    prompt: str,
    system_prompt: Optional[str] = None,
    streaming_callback: Union[Callable[[str], Awaitable[None]], None] = None,
    tools: List[Callable] = [],
    force_tool: bool = False,
    max_tool_iters: int = 5,
) -> str:
    agent = binder(
        build_simple_agent(
            name = "simple_ask",
            tools = tools,
            force_tool = force_tool,
            max_tool_iters = max_tool_iters,
        )
    )

    messages: List[Message] = []
    if system_prompt:
        messages.append({
            'role': 'system',
            'text': system_prompt,
        })
    messages.append({
        'role': 'human',
        'text': prompt,
    })
    runs: List[AgentRun] = [
        flat_messages(
            agent_name = "simple_ask",
            messages = messages,
        )
    ]
    response: AgentRun = await agent(
        _build_text_event_callback(streaming_callback) if streaming_callback else noop_callback,
        runs,
    )
    assert response['type'] == 'messages'
    last_message = response["messages"][-1]
    if last_message["role"] == "ai":
        if "text" not in last_message or not isinstance(last_message["text"], str):
            raise RuntimeError("expected ai message to have a text field")
        return last_message["text"]
    elif last_message["role"] == "tool_res":
        last_tool = last_message["tools"][-1]
        if last_tool["type"] == "any":
            return extract_tool_result_as_sting(last_tool["result"])
        elif last_tool["type"] == "layered_agent":
            raise RuntimeError("layered agent tool results are not supported in simple_ask")
        else:
            raise RuntimeError(f"unknown tool result type: {last_tool['type']}")
    else:
        raise RuntimeError(f"unexpected message role: {last_message['role']}")


async def easy_extract(
    binder: Callable[[AgentCallable], BoundAgentCallable],
    prompt: str,
    extraction_type: Type[BaseModel],
    system_prompt: Optional[str] = None,
    streaming_callback: Union[Callable[[str], Awaitable[None]], None] = None,
) -> BaseModel:
    agent = binder(
        build_extraction_agent(
            name = "easy_structured_output_agent",
            extraction_type = extraction_type,
        )
    )

    messages: List[Message] = []
    if system_prompt:
        messages.append({
            'role': 'system',
            'text': system_prompt,
        })
    messages.append({
        'role': 'human',
        'text': prompt,
    })

    response: AgentRun = await agent(
        _build_text_event_callback(streaming_callback) if streaming_callback else noop_callback,
        [
            flat_messages(
                agent_name = 'easy_structured_output_agent',
                messages = messages,
            ),
        ],
    )

    assert response['type'] == 'extraction'
    assert isinstance(response['result'], extraction_type)
    return response['result']

