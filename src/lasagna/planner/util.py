from typing import List, Any


def extract_is_trivial(
    result: Any,
) -> bool:
    extraction_type = type(result)

    if hasattr(result, 'is_trivial'):
        is_trivial = result.is_trivial
    elif isinstance(result, dict) and ('is_trivial' in result):
        is_trivial = result['is_trivial']
    else:
        raise ValueError(f"{extraction_type} must have an `is_trivial` property or key to work in this planner")

    assert isinstance(is_trivial, bool), f"{extraction_type}.is_trivial must be a boolean; instead it is a {type(is_trivial)}"
    return is_trivial


def extract_subtask_strings(
    result: Any,
) -> List[str]:
    extraction_type = type(result)

    if hasattr(result, 'subtasks'):
        subtasks = result.subtasks
    elif isinstance(result, dict) and ('subtasks' in result):
        subtasks = result['subtasks']
    else:
        raise ValueError(f"{extraction_type} must have an `subtasks` property or key to work in this planner")

    assert isinstance(subtasks, list), f"{extraction_type}.subtasks must be a list; instead it is a {type(subtasks)}"
    for i, s in enumerate(subtasks):
        assert isinstance(s, str), f"subtasks[{i}] must be a str; instead it is a {type(s)}"
    return subtasks


def extract_task_statement(
    result: Any,
) -> str:
    extraction_type = type(result)

    if hasattr(result, 'task_statement'):
        task_statement = result.task_statement
    elif isinstance(result, dict) and ('task_statement' in result):
        task_statement = result['task_statement']
    else:
        raise ValueError(f"{extraction_type} must have an `task_statement` property or key to work in this planner")

    assert isinstance(task_statement, str), f"{extraction_type}.task_statement must be a str; instead it is a {type(task_statement)}"
    return task_statement
