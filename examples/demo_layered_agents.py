from lasagna import (
    known_models,
    build_simple_agent,
    build_standard_message_extractor,
)

from lasagna.tui import (
    tui_input_loop,
)

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
    result = float(expr.evalf())
    return result


async def main() -> None:
    math_agent = build_simple_agent(
        name = 'math_agent',
        tools = [
            evaluate_math_expression,
        ],
        message_extractor = build_standard_message_extractor(
            strip_tool_messages = True,
            system_prompt_override = "You are a math assistant.",
        ),
        doc = "Use this tool if the user asks a math question.",
    )
    health_agent = known_models.anthropic_claude_sonnet_4_5_binder(
        build_simple_agent(
            name = 'health_agent',
            tools = [],
            message_extractor = build_standard_message_extractor(
                strip_tool_messages = True,
                system_prompt_override = "You are a health coach who motivates through fear.",
            ),
            doc = "Use this tool if the user asks a health question.",
        ),
    )
    my_bound_agent = known_models.openai_gpt_5_mini_binder(
        build_simple_agent(
            name = 'root_agent',
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
