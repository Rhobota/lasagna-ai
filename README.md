# Lasagna AI

[![PyPI - Version](https://img.shields.io/pypi/v/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lasagna-ai.svg)](https://pypi.org/project/lasagna-ai)
![Test Status](https://github.com/Rhobota/lasagna-ai/workflows/Test%20Matrix/badge.svg)

TODO: More badges:
 - Linux, MacOS, Windows
 - Code coverage %

**Agents for your agents!**

-----

## Highlights

- Canonizes OpenAI, Anthropic, MistralAI (and more in the future)
   - easily swap between them
   - build comities
   - etc
- Everything is streamable.
- Also easy to integrate with a database.
- Lasagna has layers. Now your agents can too!
   - Build hierarchical (or even recursive) agents! Lasagna has layers!
- Fully asyncio (with threadpools where appropriate)
   - Parallel tool calling (your tools can by either sync or async, no prob!)
- Supports _vision_!
- Strongly typed
- 90% test coverage

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install lasagna-ai[openai,anthropic]
```

## License

`lasagna-ai` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
