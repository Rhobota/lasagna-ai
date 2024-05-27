from .types import (
    AgentRecord,
    AgentCallable,
    ModelProviderRecord,
    ProviderFactory,
    ModelRecord,
)

from typing import Dict, List


AGENTS: Dict[str, AgentRecord] = {}

MODEL_PROVIDERS: Dict[str, ModelProviderRecord] = {}


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


def register_model_provider(
    key: str,
    name: str,
    factory: ProviderFactory,
    models: List[ModelRecord],
) -> None:
    if key in MODEL_PROVIDERS:
        raise RuntimeError(f"A model provider with this key is already registered: {key}")
    MODEL_PROVIDERS[key] = {
        'name': name,
        'factory': factory,
        'models': models,
    }
