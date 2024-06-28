from lasagna import (
    bind_model,
    build_most_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable, cast

import asyncio

from dotenv import load_dotenv; load_dotenv()

import sympy as sp


MODEL_BINDER = bind_model('openai', 'gpt-3.5-turbo-0125')


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
    result = cast(float, expr.evalf())
    return result


async def main() -> None:
    system_prompt = "You are my math assistant."
    tools: List[Callable] = [
        evaluate_math_expression,
    ]
    my_agent = build_most_simple_agent(tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
