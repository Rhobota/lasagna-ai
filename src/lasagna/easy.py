from typing import (
    Callable,
    List,
    Union,
    Type,
    Awaitable,
    Optional,
)

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
from .types import ExtractionType


def _build_text_event_callback(simple_text_callback: Callable[[str], Awaitable[None]]) -> EventCallback:
    async def bound_simple_text_callback(event: EventPayload) -> None:
        if event[0] == 'ai' and event[1] == 'text_event':
            await simple_text_callback(event[2])
    return bound_simple_text_callback


def _extract_message_text(message: Message) -> str:
    if message["role"] == "system" or message["role"] == "human":
        return message["text"] or ""
    elif message["role"] == "ai":
        return message["text"] or ""
    elif message["role"] == "tool_res":
        return extract_tool_result_as_sting(message["tools"][-1])
    else:
        raise RuntimeError(f"unexpected message role: {message['role']}")


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
            name = "easy_ask",
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
            agent_name = "easy_ask",
            messages = messages,
        )
    ]
    response: AgentRun = await agent(
        _build_text_event_callback(streaming_callback) if streaming_callback else noop_callback,
        runs,
    )
    assert response['type'] == 'messages'
    response_text: List[str] = []
    for message in response["messages"]:
        response_text.append(_extract_message_text(message))
    return "\n".join(response_text)


async def easy_extract(
    binder: Callable[[AgentCallable], BoundAgentCallable],
    prompt: str,
    extraction_type: Type[ExtractionType],
    system_prompt: Optional[str] = None,
    streaming_callback: Union[Callable[[str], Awaitable[None]], None] = None,
) -> ExtractionType:
    agent = binder(
        build_extraction_agent(
            name = "easy_extract",
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
                agent_name = 'easy_extract',
                messages = messages,
            ),
        ],
    )

    assert response['type'] == 'extraction'
    assert isinstance(response['result'], extraction_type)
    return response['result']
