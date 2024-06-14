![Lasagna AI Logo](https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/lasagna-ai.png)

# Lasagna AI

[![PyPI - Version](https://img.shields.io/pypi/v/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
![Test Status](https://github.com/Rhobota/lasagna-ai/actions/workflows/test.yml/badge.svg?branch=main)

- 🥞  **Layered agents!**
  - Agents for your agents!
  - Tool-use and layering FTW 💪
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
  - Native support for OpenAI, Anthropic, MistralAI (+ more to come).
  - Message representations are canonized. 😇
  - Supports vision!
  - Easily build committees!
  - Swap providers or models mid-conversation.
  - Delegate tasks among model providers or model sizes.
  - Parallelize all the things.

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install -U lasagna-ai[openai,anthropic]
```

## Quickstart

TODO

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

## License

`lasagna-ai` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Joke Acronym

Layered Agents with toolS And aGeNts and Ai

