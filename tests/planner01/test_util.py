import pytest

from lasagna.planner01.util import (
    NoResult,
    _extract_last_result_from_run,
    extract_is_trivial,
    extract_subtask_strings,
    extract_task_statement,
)

from lasagna.types import AgentRun
from lasagna.planner01.default_prompts import PlanOutput

import copy


PLAN_OBJ = PlanOutput(
    thoughts = '... hum ...',
    task_statement = 'do the thing',
    is_trivial = False,
    subtasks = ['first thing', 'next thing', 'last thing'],
)

PLAN_DCT = PLAN_OBJ.model_dump()
assert isinstance(PLAN_DCT, dict)


NO_RESULT_RUN: AgentRun = {
    'agent': 'a',
    'type': 'parallel',
    'runs': [
        {
            'agent': 'b',
            'type': 'chain',
            'runs': [
                {
                    'agent': 'c',
                    'type': 'messages',
                    'messages': [
                        {
                            'role': 'system',
                            'text': 'do good',
                        },
                    ],
                },
            ],
        },
        {
            'agent': 'c',
            'type': 'messages',
            'messages': [
                {
                    'role': 'system',
                    'text': 'do better',
                },
            ],
        },
    ],
}


def _insert_result(result) -> AgentRun:
    run = copy.deepcopy(NO_RESULT_RUN)
    assert run['type'] == 'parallel'
    run['runs'].append({
        'agent': 'd',
        'type': 'extraction',
        'messages': [],
        'result': copy.deepcopy(result),
    })
    return run


def test_extract_last_result_from_run():
    with pytest.raises(NoResult):
        _extract_last_result_from_run(NO_RESULT_RUN)

    run = _insert_result(PLAN_OBJ)
    assert _extract_last_result_from_run(run) == PLAN_OBJ

    run = _insert_result(PLAN_DCT)
    assert _extract_last_result_from_run(run) == PLAN_DCT


def test_extract_is_trivial():
    assert extract_is_trivial(_insert_result(PLAN_OBJ)) == False
    assert extract_is_trivial(_insert_result(PLAN_DCT)) == False


def test_extract_subtask_strings():
    assert extract_subtask_strings(_insert_result(PLAN_OBJ)) == ['first thing', 'next thing', 'last thing']
    assert extract_subtask_strings(_insert_result(PLAN_DCT)) == ['first thing', 'next thing', 'last thing']


def test_extract_task_statement():
    assert extract_task_statement(_insert_result(PLAN_OBJ)) == 'do the thing'
    assert extract_task_statement(_insert_result(PLAN_DCT)) == 'do the thing'
