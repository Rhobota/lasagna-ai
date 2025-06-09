![Lasagna AI Logo](https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/lasagna-ai.png)

# Lasagna AI

[![PyPI - Version](https://img.shields.io/pypi/v/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
![Test Status](https://github.com/Rhobota/lasagna-ai/actions/workflows/test.yml/badge.svg?branch=main)
[![Downloads](https://static.pepy.tech/badge/lasagna-ai)](https://pepy.tech/project/lasagna-ai)

- 📚 **Official Docs:** [https://lasagna-ai.rhobota.com/](https://lasagna-ai.rhobota.com/)

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
  - Core support for [OpenAI](https://platform.openai.com/docs/models), [Anthropic](https://docs.anthropic.com/en/docs/welcome), and [AWS Bedrock](https://docs.aws.amazon.com/bedrock/).
  - Experimental support for [Ollama](https://ollama.com/search) and [NVIDIA NIM/NGC](https://build.nvidia.com/explore/reasoning).
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
- [Debug Logging](#debug-logging)
- [Special Thanks](#special-thanks)
- [License](#license)

## Installation

```console
pip install -U lasagna-ai[openai,anthropic,bedrock]
```

If you want to easily run all the [./examples](./examples), then you can
install the extra dependencies used by those examples:

```console
pip install -U lasagna-ai[openai,anthropic,bedrock,example-deps]
```

## Used By

Lasagna AI is used in production by:

[![AutoAuto](https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/autoauto.png)](https://www.autoauto.ai/)

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
