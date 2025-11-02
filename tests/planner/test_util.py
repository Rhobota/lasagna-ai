from lasagna.planner.util import (
    extract_is_trivial,
    extract_subtask_strings,
    extract_task_statement,
)

from lasagna.planner.default_prompts import PlanOutput


PLAN_OBJ = PlanOutput(
    thoughts = '... hum ...',
    task_statement = 'do the thing',
    is_trivial = False,
    subtasks = ['first thing', 'next thing', 'last thing'],
)

PLAN_DCT = PLAN_OBJ.model_dump()
assert isinstance(PLAN_DCT, dict)


def test_extract_is_trivial():
    assert extract_is_trivial(PLAN_OBJ) == False
    assert extract_is_trivial(PLAN_DCT) == False


def test_extract_subtask_strings():
    assert extract_subtask_strings(PLAN_OBJ) == ['first thing', 'next thing', 'last thing']
    assert extract_subtask_strings(PLAN_DCT) == ['first thing', 'next thing', 'last thing']


def test_extract_task_statement():
    assert extract_task_statement(PLAN_OBJ) == 'do the thing'
    assert extract_task_statement(PLAN_DCT) == 'do the thing'
