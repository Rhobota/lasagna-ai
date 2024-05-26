import re

from typing import Tuple, List

from .types import ToolParam


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
