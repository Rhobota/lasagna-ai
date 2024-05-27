import pytest

from lasagna.registrar import (
    register_agent,
    register_model_provider,
    AGENTS,
    MODEL_PROVIDERS,
)

from lasagna.types import AgentCallable, ProviderFactory, ModelRecord

from typing import List


def test_register_agent():
    AGENTS.clear()
    runner: AgentCallable = 'mock'   # type: ignore
    register_agent('myagent', 'My Agent', runner)
    assert AGENTS == {
        'myagent': {
            'name': 'My Agent',
            'runner': runner,
        },
    }
    with pytest.raises(RuntimeError):
        register_agent('myagent', 'My Agent', runner)


def test_register_model_provider():
    MODEL_PROVIDERS.clear()
    factory: ProviderFactory = 'mock'   # type: ignore
    models: List[ModelRecord] = [
        {
            'formal_name': 'mymodel',
            'display_name': 'My Model',
        }
    ]
    register_model_provider('myprovider', 'My Provider', factory, models)
    assert MODEL_PROVIDERS == {
        'myprovider': {
            'name': 'My Provider',
            'factory': factory,
            'models': models,
        },
    }
    with pytest.raises(RuntimeError):
        register_model_provider('myprovider', 'My Provider', factory, models)
