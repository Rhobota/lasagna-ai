from .types import (
    ChatParams,
    ChatMessage,
    EventCallback,
    AgentRecord,
    ModelProviderRecord,
)

from typing import List, Dict


AGENTS: Dict[str, AgentRecord] = {}

MODEL_PROVIDERS: Dict[str, ModelProviderRecord] = {}


async def run(
    chat_params: ChatParams,
    event_callback: EventCallback,
    messages: List[ChatMessage],
) -> List[ChatMessage]:
    agent = AGENTS[chat_params['agent_name']]

    provider = MODEL_PROVIDERS[chat_params['provider']]
    factory = provider['factory']
    model_names = [m['formal_name'] for m in provider['models']]
    if chat_params['model_name'] not in model_names:
        raise RuntimeError(f"unknown model name: {chat_params['model_name']}")
    kwargs = chat_params.get('model_kwargs', None)
    if kwargs is None:
        kwargs = {}
    llm = factory(model=chat_params['model_name'], **kwargs)

    return await agent['factory'](
        llm,
        event_callback,
        messages,
    )
