from .types import (
    AgentRecord,
    AgentCallable,
    ProviderRecord,
    ModelFactory,
    ModelRecord,
)

from typing import Dict, List


AGENTS: Dict[str, AgentRecord] = {}

PROVIDERS: Dict[str, ProviderRecord] = {}


def register_agent(
    key: str,
    name: str,
    runner: AgentCallable,
) -> None:
    if key in AGENTS:
        raise RuntimeError(f"An agent with this key is already registered: {key}")
    AGENTS[key] = {
        'name': name,
        'runner': runner,
    }


def register_provider(
    key: str,
    name: str,
    factory: ModelFactory,
    models: List[ModelRecord],
) -> None:
    if key in PROVIDERS:
        raise RuntimeError(f"A model provider with this key is already registered: {key}")
    PROVIDERS[key] = {
        'name': name,
        'factory': factory,
        'models': models,
    }
