from lasagna import (
    known_models,
    build_most_simple_agent,
    build_layered_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import cast

import asyncio

from dotenv import load_dotenv; load_dotenv()

import sympy as sp    # type: ignore


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
    math_agent = build_layered_agent(
        tools = [
            evaluate_math_expression,
        ],
        system_prompt = "You are a math assistant.",
        doc = "Use this tool if the user asks a math question.",
    )
    health_agent = known_models.BIND_ANTHROPIC_claude_35_sonnet()(
        build_layered_agent(
            tools = [],
            system_prompt = "You are a health coach who motivates through fear.",
            doc = "Use this tool if the user asks a health question.",
        ),
    )
    my_bound_agent = known_models.BIND_OPENAI_gpt_4o_mini()(
        build_most_simple_agent(
            tools = [
                math_agent,
                health_agent,
            ],
        ),
    )
    await tui_input_loop(
        my_bound_agent,
        system_prompt = "You interact with the user and delegate to your tools as needed.",
    )


if __name__ == '__main__':
    asyncio.run(main())
