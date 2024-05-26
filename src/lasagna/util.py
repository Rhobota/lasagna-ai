import re

from typing import Tuple, List, Literal, Union, TypeVar, Callable

from .types import ToolParam


T = TypeVar('T')


def parse_docstring(docstring: str) -> Tuple[str, List[ToolParam]]:
    lines = docstring.splitlines()
    indent_amounts = [(len(line) - len(line.lstrip())) for line in lines]
    first_indent = max(ia for ia in indent_amounts[:2])
    lines_stripped = []
    for ia, l in zip(indent_amounts, lines):
        to_strip = min(ia, first_indent, len(l))
        lines_stripped.append(l[to_strip:].rstrip())
    dedented_docs = "\n".join(lines_stripped).strip()
    description_match = re.search(r"^(.*?)(?=:param:)", dedented_docs, re.DOTALL)
    if description_match is None:
        raise ValueError("no description found")
    description = ' '.join(description_match[1].strip().splitlines())
    if not description:
        raise ValueError("no description found")
    params_found = re.findall(r":param:\s+(\w+):\s+([\w ]+):\s+(.+)", dedented_docs)
    params: List[ToolParam] = [
        {
            'name': p[0].strip(),
            'type': p[1].strip(),
            'description': p[2].strip(),
        }
        for p in params_found
    ]
    for p in params:
        if not p['name']:
            raise ValueError("no parameter name found")
        if not p['type'].startswith('enum ') and p['type'] not in ['str', 'float', 'int', 'bool']:
            raise ValueError(f"invalid type found: {p['type']}")
        if not p['description']:
            raise ValueError("no parameter name found")
    return description, params


def combine_pairs(
    lst: List[T],
    decision_func: Callable[[T, T], Union[Literal[False], Tuple[Literal[True], T]]],
) -> List[T]:
    # This implementation is ugly but I can't find a nice way to do it.
    res: List[T] = []
    i: int = 0
    while i < len(lst) - 1:
        j = i + 1
        flag = decision_func(lst[i], lst[j])
        if flag is False:
            res.append(lst[i])
        else:
            new_val = flag[1]
            res.append(new_val)
            i += 1  # <-- This is where the ugliness comes from! We need to skip the next iteration
        i += 1
    if i < len(lst):
        # If the last pair was *not* combined, then we need to add the last element.
        res.append(lst[i])
    return res
