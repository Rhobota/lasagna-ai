import pytest

from lasagna.agent_util import (
    make_model_binder,
    noop_callback,
    human_input,
    extract_last_message,
)

from lasagna.types import AgentRun

from lasagna.planner.debug_model import DebugModel

from lasagna.planner.agent import (
    build_default_planning_agent,
    _minimal_copy_lrpa_aware,
    _extract_messages_lrpa_aware,
)

import copy
from typing import List


def _count_runs(run: AgentRun) -> int:
    subruns = 0
    if run['type'] == 'chain' or run['type'] == 'parallel':
        subruns = sum(_count_runs(r) for r in run['runs'])
    return subruns + 1


def _assert_recursive_structure(run: AgentRun) -> None:
    assert run['agent'] == 'planning_agent_chain'
    assert run['type'] == 'chain'

    runs = run['runs']

    if len(runs) == 2:
        assert runs[0]['agent'] == 'extraction_run'
        assert runs[1]['agent'] == 'answer_chain'

    elif len(runs) == 3:
        assert runs[0]['agent'] == 'extraction_run'
        assert runs[1]['agent'] == 'subtask_chain_of_chains'
        assert runs[2]['agent'] == 'answer_chain'

        subtask_chain_of_chains = runs[1]
        assert subtask_chain_of_chains['type'] == 'chain'

        for subtask_chain in subtask_chain_of_chains['runs']:
            assert subtask_chain['agent'] == 'subtask_chain'
            assert subtask_chain['type'] == 'chain'
            subtask_runs = subtask_chain['runs']
            assert len(subtask_runs) == 2
            assert subtask_runs[0]['agent'] == 'subtask_input_generator'
            _assert_recursive_structure(subtask_runs[1])

    else:
        assert False, f"why is len(runs) == {len(runs)}?"


@pytest.mark.asyncio
async def test_agent():
    binder_seed_0 = make_model_binder(DebugModel, 'NA', seed = 0)
    binder_seed_1 = make_model_binder(DebugModel, 'NA', seed = 1)

    planning_agent = build_default_planning_agent(binder = binder_seed_0)
    run = await planning_agent(noop_callback, human_input('hi'))
    assert _count_runs(run) == 41         # <-- if fails, did self.rand in the DebugModel change?
    _assert_recursive_structure(run)
    assert extract_last_message(
        run,
        from_tools = False,
        from_extraction = False,
    ).get('text') == 'The answer is: 13'  # <-- also depends on self.rand being reproducible

    planning_agent = build_default_planning_agent(binder = binder_seed_1)
    run = await planning_agent(noop_callback, human_input('hi'))
    assert _count_runs(run) == 5          # <-- if fails, did self.rand in the DebugModel change?
    _assert_recursive_structure(run)
    assert extract_last_message(
        run,
        from_tools = False,
        from_extraction = False,
    ).get('text') == 'The answer is: 3'   # <-- also depends on self.rand being reproducible


def test_minimal_copy_lrpa_aware():
    orig: List[AgentRun] = human_input('hi')
    orig_copy = copy.deepcopy(orig)
    modified_copy, inner_run, depth = _minimal_copy_lrpa_aware(orig)
    assert orig == orig_copy, "ensure pure function"
    assert depth == 0
    assert modified_copy == [
        {
            'agent': 'human_input',
            'type': 'messages',
            'messages': [
                {
                    'role': 'human',
                    'text': 'hi',
                },
            ],
        },
        {
            'agent': 'planning_agent_chain',
            'type': 'chain',
            'runs': [],
        },
    ]
    assert inner_run is modified_copy[1]

    orig: List[AgentRun] = [
        {
            'agent': 'human_input',
            'type': 'messages',
            'messages': [
                {
                    'role': 'human',
                    'text': 'hi',
                },
            ],
        },
        {
            'agent': 'planning_agent_chain',
            'type': 'chain',
            'runs': [
                {
                    'agent': 'extraction_run',
                    'type': 'extraction',
                    'messages': [],
                    'result': {},
                },
                {
                    'agent': 'subtask_chain_of_chains',
                    'type': 'chain',
                    'runs': [
                        {
                            'agent': 'subtask_chain',
                            'type': 'chain',
                            'runs': [
                                {
                                    'agent': 'subtask_input_generator',
                                    'type': 'messages',
                                    'messages': [
                                        {
                                            'role': 'human',
                                            'text': '...',
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ]
    orig_copy = copy.deepcopy(orig)
    modified_copy, inner_run, depth = _minimal_copy_lrpa_aware(orig)
    assert orig == orig_copy, "ensure pure function"
    assert depth == 1
    assert modified_copy == [
        {
            'agent': 'human_input',
            'type': 'messages',
            'messages': [
                {
                    'role': 'human',
                    'text': 'hi',
                },
            ],
        },
        {
            'agent': 'planning_agent_chain',
            'type': 'chain',
            'runs': [
                {
                    'agent': 'extraction_run',
                    'type': 'extraction',
                    'messages': [],
                    'result': {},
                },
                {
                    'agent': 'subtask_chain_of_chains',
                    'type': 'chain',
                    'runs': [
                        {
                            'agent': 'subtask_chain',
                            'type': 'chain',
                            'runs': [
                                {
                                    'agent': 'subtask_input_generator',
                                    'type': 'messages',
                                    'messages': [
                                        {
                                            'role': 'human',
                                            'text': '...',
                                        },
                                    ],
                                },
                                {
                                    'agent': 'planning_agent_chain',
                                    'type': 'chain',
                                    'runs': [],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ]
    assert inner_run is modified_copy[1]['runs'][1]['runs'][0]['runs'][1]  # type: ignore


def test_extract_messages_lrpa_aware():
    runs: List[AgentRun] = [
        {
            'agent': 'human_input',
            'type': 'messages',
            'messages': [
                {
                    'role': 'human',
                    'text': 'hi',
                },
            ],
        },
        {
            'agent': 'planning_agent_chain',
            'type': 'chain',
            'runs': [
                {
                    'agent': 'extraction_run',
                    'type': 'extraction',
                    'messages': [],
                    'result': {'extraction_level': 0},
                },
                {
                    'agent': 'subtask_chain_of_chains',
                    'type': 'chain',
                    'runs': [
                        {
                            'agent': 'subtask_chain',
                            'type': 'chain',
                            'runs': [
                                {
                                    'agent': 'subtask_input_generator',
                                    'type': 'messages',
                                    'messages': [
                                        {
                                            'role': 'human',
                                            'text': '... go deeper on task 1',
                                        },
                                    ],
                                },
                                {
                                    'agent': 'planning_agent_chain',
                                    'type': 'chain',
                                    'runs': [
                                        {
                                            'agent': 'extraction_run',
                                            'type': 'extraction',
                                            'messages': [],
                                            'result': {'extraction_level': 1},
                                        },
                                        {
                                            'agent': 'answer_chain',
                                            'type': 'chain',
                                            'runs': [
                                                {
                                                    'agent': 'answer_input_prompt',
                                                    'type': 'messages',
                                                    'messages': [
                                                        {
                                                            'role': 'human',
                                                            'text': '... make your final answer',
                                                        },
                                                    ],
                                                },
                                                {
                                                    'agent': 'answer_agent',
                                                    'type': 'messages',
                                                    'messages': [
                                                        {
                                                            'role': 'ai',
                                                            'text': 'it is 47',
                                                        },
                                                        {
                                                            'role': 'human',
                                                            'text': 'really?',
                                                        },
                                                        {
                                                            'role': 'ai',
                                                            'text': 'yes',
                                                        },
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            'agent': 'subtask_chain',
                            'type': 'chain',
                            'runs': [
                                {
                                    'agent': 'subtask_input_generator',
                                    'type': 'messages',
                                    'messages': [
                                        {
                                            'role': 'human',
                                            'text': '... go deeper on task 2',
                                        },
                                    ],
                                },
                                {
                                    'agent': 'planning_agent_chain',
                                    'type': 'chain',
                                    'runs': [
                                        {
                                            'agent': 'extraction_run',
                                            'type': 'extraction',
                                            'messages': [],
                                            'result': {'extraction_level': 2},
                                        },
                                        {
                                            'agent': 'answer_chain',
                                            'type': 'chain',
                                            'runs': [
                                                {
                                                    'agent': 'answer_input_prompt',
                                                    'type': 'messages',
                                                    'messages': [
                                                        {
                                                            'role': 'human',
                                                            'text': '... make your final answer now',
                                                        },
                                                    ],
                                                },
                                                {
                                                    'agent': 'answer_agent',
                                                    'type': 'messages',
                                                    'messages': [
                                                        {
                                                            'role': 'ai',
                                                            'text': 'it is 48',
                                                        },
                                                        {
                                                            'role': 'human',
                                                            'text': 'really?',
                                                        },
                                                        {
                                                            'role': 'ai',
                                                            'text': 'yes!',
                                                        },
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            'agent': 'subtask_chain',
                            'type': 'chain',
                            'runs': [
                                {
                                    'agent': 'subtask_input_generator',
                                    'type': 'messages',
                                    'messages': [
                                        {
                                            'role': 'human',
                                            'text': '... go deeper on task 3',
                                        },
                                    ],
                                },
                                {
                                    'agent': 'planning_agent_chain',
                                    'type': 'chain',
                                    'runs': [
                                        {
                                            'agent': 'extraction_run',
                                            'type': 'extraction',
                                            'messages': [],
                                            'result': {'extraction_level': 3},
                                        },
                                        {
                                            'agent': 'answer_chain',
                                            'type': 'chain',
                                            'runs': [
                                                {
                                                    'agent': 'answer_input_prompt',
                                                    'type': 'messages',
                                                    'messages': [
                                                        {
                                                            'role': 'human',
                                                            'text': '... make your final answer now again',
                                                        },
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ]
    runs_copy = copy.deepcopy(runs)
    messages = _extract_messages_lrpa_aware(runs)
    assert runs == runs_copy, "ensure pure function"
    assert messages == [
        {
            'role': 'human',
            'text': 'hi',
        },
        {
            'role': 'ai',
            'text': '{"extraction_level": 0}',
        },
        {
            'role': 'human',
            'text': '... go deeper on task 1',
        },
        {
            'role': 'ai',
            'text': 'it is 47',
        },
        {
            'role': 'ai',
            'text': 'yes',
        },
        {
            'role': 'human',
            'text': '... go deeper on task 2',
        },
        {
            'role': 'ai',
            'text': 'it is 48',
        },
        {
            'role': 'ai',
            'text': 'yes!',
        },
        {
            'role': 'human',
            'text': '... go deeper on task 3',
        },
        {
            'role': 'ai',
            'text': '{"extraction_level": 3}',
        },
        {
            'role': 'human',
            'text': '... make your final answer now again',
        },
    ]

    # Now, if we append the answer to the top-level planning_agent_chain,
    # then all the stuff inside will collapse.
    top_level_planner_run = runs[1]
    assert top_level_planner_run['agent'] == 'planning_agent_chain'
    assert top_level_planner_run['type'] == 'chain'
    top_level_planner_run['runs'].append({
        'agent': 'answer_chain',
        'type': 'chain',
        'runs': [
            {
                'agent': 'answer_input_prompt',
                'type': 'messages',
                'messages': [
                    {
                        'role': 'human',
                        'text': 'FINAL ANSWER?',
                    },
                ],
            },
            {
                'agent': 'answer_agent',
                'type': 'messages',
                'messages': [
                    {
                        'role': 'ai',
                        'text': 'answer is 7',
                    },
                    {
                        'role': 'human',
                        'text': 'no?',
                    },
                    {
                        'role': 'ai',
                        'text': 'sure is',
                    },
                ],
            },
        ],
    })
    runs_copy = copy.deepcopy(runs)
    messages = _extract_messages_lrpa_aware(runs)
    assert runs == runs_copy, "ensure pure function"
    assert messages == [
        {
            'role': 'human',
            'text': 'hi',
        },
        {
            'role': 'ai',
            'text': 'answer is 7',
        },
        {
            'role': 'ai',
            'text': 'sure is',
        },
    ]
