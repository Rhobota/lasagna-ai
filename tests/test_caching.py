import pytest
import hashlib
from typing import List

from lasagna.caching import (
    in_memory_cached_agent,
    _hash_agent_runs,
)

from lasagna.types import AgentRun

from lasagna import __version__

from .test_agent_runner import (
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


@pytest.mark.asyncio
async def test_in_memory_cached_agent():
    pass  # TODO


@pytest.mark.asyncio
async def test_hash_agent_runs():
    model = MockProvider(model='something')
    got = await _hash_agent_runs(model, _PREV_RUNS)
    s = ''.join([v.strip() for v in f'''
        __str____version__{__version__}__model__MockProvider
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
    correct = hashlib.md5(s.encode('utf-8')).hexdigest()
    assert got == correct
