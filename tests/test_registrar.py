import pytest

from lasagna.registrar import (
    register_agent,
    register_provider,
    AGENTS,
    PROVIDERS,
)

from lasagna.types import AgentCallable, ModelFactory, ModelRecord

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
    PROVIDERS.clear()
    factory: ModelFactory = 'mock'   # type: ignore
    models: List[ModelRecord] = [
        {
            'formal_name': 'mymodel',
            'display_name': 'My Model',
        }
    ]
    register_provider('myprovider', 'My Provider', factory, models)
    assert PROVIDERS == {
        'myprovider': {
            'name': 'My Provider',
            'factory': factory,
            'models': models,
        },
    }
    with pytest.raises(RuntimeError):
        register_provider('myprovider', 'My Provider', factory, models)
