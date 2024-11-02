![Lasagna AI Logo](https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/lasagna-ai.png)

# Lasagna AI

[![PyPI - Version](https://img.shields.io/pypi/v/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
![Test Status](https://github.com/Rhobota/lasagna-ai/actions/workflows/test.yml/badge.svg?branch=main)
[![Downloads](https://static.pepy.tech/badge/lasagna-ai)](https://pepy.tech/project/lasagna-ai)

- 🥞  **Layered agents!**
  - Agents for your agents!
  - Tool-use, structured output ("extraction"), and layering FTW 💪
  - Ever wanted a _recursive_ agent? Now you can have one! 🤯
  - _Parallel_ tool-calling by default.
  - Fully asyncio.
  - 100% Python type hints.
  - Functional-style 😎
  - (optional) Easy & pluggable caching! 🏦

- 🚣  **Streamable!**
  - Event streams for _everything_.
  - Asyncio generators are awesome.

- 🗃️ **Easy database integration!**
  - Don't rage when trying to store raw messages and token counts. 😡 🤬
  - Yes, you _can_ have _both_ streaming and easy database storage.

- ↔️ **Provider/model agnostic and interoperable!**
  - Native support for [OpenAI](https://platform.openai.com/docs/models), [Anthropic](https://docs.anthropic.com/en/docs/welcome), [NVIDIA NIM/NGC](https://build.nvidia.com/explore/reasoning) (+ more to come).
  - Message representations are canonized. 😇
  - Supports vision!
  - Easily build committees!
  - Swap providers or models mid-conversation.
  - Delegate tasks among model providers or model sizes.
  - Parallelize all the things.

-----

## Table of Contents

- [Installation](#installation)
- [Used By](#used-by)
- [Quickstart](#quickstart)
- [Debug Logging](#debug-logging)
- [Special Thanks](#special-thanks)
- [License](#license)

## Installation

```console
pip install -U lasagna-ai[openai,anthropic]
```

If you want to easily run all the [./examples](./examples), then you can
install the extra dependencies used by those examples:

```console
pip install -U lasagna-ai[openai,anthropic,example-deps]
```

## Used By

Lasagna is used in production by:

[![AutoAuto](https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/autoauto.png)](https://www.autoauto.ai/)

## Quickstart

Here is the _most simple_ agent (it doesn't add *anything* to the underlying model).
More complex agents would add tools and/or use layers of agents, but not this one!
Anyway, run it in your terminal and you can chat interactively with the model. 🤩

(taken from [./examples/quickstart.py](./examples/quickstart.py))

```python
from lasagna import (
    known_models,
    build_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable

import asyncio

from dotenv import load_dotenv; load_dotenv()


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


async def main() -> None:
    system_prompt = "You are grumpy."
    tools: List[Callable] = []
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
```

**Want to add your first tool?** LLMs can't natively do arithmetic
(beyond simple arithmetic with small numbers), so let's give our
model a tool for doing arithmetic! 😎

(full example at [./examples/quickstart_with_math_tool.py](./examples/quickstart_with_math_tool.py))

```python
import sympy as sp

...

def evaluate_math_expression(expression: str) -> float:
    """
    This tool evaluates a math expression and returns the result.
    Pass math expression as a string, for example:
     - "3 * 6 + 1"
     - "cos(2 * pi / 3) + log(8)"
     - "(4.5/2) + (6.3/1.2)"
     - ... etc
    :param: expression: str: the math expression to evaluate
    """
    expr = sp.sympify(expression)
    result = float(expr.evalf())
    return result

...

    ...
    tools: List[Callable] = [
        evaluate_math_expression,
    ]
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    ...

...
```

**Simple RAG:** Everyone's favorite tool: _Retrieval Augmented Generation_ (RAG). Let's GO! 📚💨  
See: [./examples/demo_rag.py](./examples/demo_rag.py)

## Debug Logging

This library logs using Python's builtin `logging` module. It logs mostly to `INFO`, so here's a snippet of code you can put in _your_ app to see those traces:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# ... now use Lasagna as you normally would, but you'll see extra log traces!
```

## Special Thanks

Special thanks to those who inspired this library:
- Numa Dhamani (buy her book: [Introduction to Generative AI](https://a.co/d/03dHnRmX))
- Dave DeCaprio's [voice-stream library](https://github.com/DaveDeCaprio/voice-stream)

## License

`lasagna-ai` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Joke Acronym

Layered Agents with toolS And aGeNts and Ai
