import pytest
import hashlib
import random
from typing import List

from lasagna.caching import (
    in_memory_cached_agent,
    _hash_agent_runs,
)

from lasagna.types import AgentRun, Model, EventCallback, Message

from lasagna.agent_util import bind_model

from lasagna import __version__

from lasagna.mock_provider import (
    MockProvider,
)


_PREV_RUNS: List[AgentRun] = [
    {
        'type': 'messages',
        'messages': [
            {
                'role': 'system',
                'text': 'You are a robot.',
            },
        ],
    },
]


def _make_agent():
    # We need this factory so we can make as many separately-cached agents as we want.
    @bind_model(MockProvider, 'some_model')
    @in_memory_cached_agent
    async def my_agent(
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        assert prev_runs == _PREV_RUNS
        messages: List[Message] = []
        new_messages = await model.run(event_callback, messages, [])
        return {
            'type': 'messages',
            'messages': [
                *new_messages,
                {
                    'role': 'human',
                    'text': f"Here is a random value {random.random()}.",
                },
            ],
        }

    return my_agent


@pytest.mark.asyncio
async def test_in_memory_cached_agent():
    """
    We test the *in-memory* cache just as a concrete way to test the
    underlying cache decorator.
    """
    async def callback(event):
        pass

    agent1 = _make_agent()
    result1 = await agent1(callback, _PREV_RUNS)

    agent2 = _make_agent()
    result2 = await agent2(callback, _PREV_RUNS)

    assert result1 != result2   # because of the use of random.random() in the final message, these will be different (in the *vast* likelihood)

    for _ in range(5):
        result1_again = await agent1(callback, _PREV_RUNS)
        result2_again = await agent2(callback, _PREV_RUNS)

        assert result1 == result1_again  # because it was cached
        assert result2 == result2_again  # because it was cached


@pytest.mark.asyncio
async def test_hash_agent_runs():
    model = MockProvider(model='something')

    s = ''.join([v.strip() for v in f'''
        __open_dict__
            __open_list__
                __open_list__
                    __str__model
                    __str__something
                __close_list__
                __open_list__
                    __str__model_kwargs
                    __open_dict__
                        __open_list__
                        __close_list__
                    __close_dict__
                __close_list__
                __open_list__
                    __str__provider
                    __str__mock
                __close_list__
            __close_list__
        __close_dict__
        '''.strip().splitlines()])

    model_config_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()

    got = await _hash_agent_runs(model, _PREV_RUNS)

    s = ''.join([v.strip() for v in f'''
        __str____version__{__version__}__model__{model_config_hash}
        __open_list__
            __open_dict__
                __open_list__
                    __open_list__
                        __str__messages
                        __open_list__
                            __open_dict__
                                __open_list__
                                    __open_list__
                                        __str__role
                                        __str__system
                                    __close_list__
                                    __open_list__
                                        __str__text
                                        __str__You are a robot.
                                    __close_list__
                                __close_list__
                            __close_dict__
                        __close_list__
                    __close_list__
                    __open_list__
                        __str__type
                        __str__messages
                    __close_list__
                __close_list__
            __close_dict__
        __close_list__
        '''.strip().splitlines()])

    correct = hashlib.sha256(s.encode('utf-8')).hexdigest()

    assert got == correct
