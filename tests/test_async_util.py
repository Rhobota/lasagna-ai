import pytest

import asyncio
import time

from typing import List, Tuple

from lasagna.async_util import async_throttle

from lasagna.util import get_name
from lasagna.tools_util import get_tool_params


@pytest.mark.asyncio
async def test_async_throttle_max_concurrent():
    shared_decorator = async_throttle(max_concurrent=2)

    @shared_decorator
    async def foo_1() -> Tuple[str, float]:
        print('foo_1')
        call_time = time.time()
        await asyncio.sleep(0.05)
        return 'foo_1', call_time

    @shared_decorator
    async def foo_2() -> Tuple[str, float]:
        print('foo_2')
        call_time = time.time()
        await asyncio.sleep(0.05)
        return 'foo_2', call_time

    def _assert_results(expected_name: str, res: List[Tuple[str, float]]) -> None:
        res = sorted(res)  # likely already sorted, based on how the even loop schedules tasks, but that's not guaranteed, so we sort here
        assert len(res) == 10

        for name, _ in res:
            assert name == expected_name

        pairs: list[float] = []
        for i in range(0, len(res), 2):
            _, a = res[i]
            _, b = res[i+1]
            assert 0.0 <= b - a <= 0.01
            avg = (a + b) / 2
            pairs.append(avg)

        for a, b in zip(pairs, pairs[1:]):
            diff = abs(a - b)
            assert 0.04 < diff < 0.06

    tasks = [
        f()
        for _ in range(10)
        for f in [foo_1, foo_2]
    ]

    res = await asyncio.gather(*tasks)
    assert len(res) == 20

    foo_1_res = res[0::2]
    foo_2_res = res[1::2]

    _assert_results('foo_1', foo_1_res)
    _assert_results('foo_2', foo_2_res)


@pytest.mark.asyncio
async def test_async_throttle_max_per_second():
    shared_decorator = async_throttle(max_per_second=10)
    period = 1 / 10

    @shared_decorator
    async def foo_1() -> Tuple[str, float]:
        print('foo_1')
        call_time = time.time()
        await asyncio.sleep(3 * period)  # <-- sleeping *longer* than the period, so that calls overlap
        return 'foo_1', call_time

    @shared_decorator
    async def foo_2() -> Tuple[str, float]:
        print('foo_2')
        call_time = time.time()
        await asyncio.sleep(3 * period)  # <-- sleeping *longer* than the period, so that calls overlap
        return 'foo_2', call_time

    def _assert_results(expected_name: str, res: List[Tuple[str, float]]) -> None:
        res = sorted(res)  # likely already sorted, based on how the even loop schedules tasks, but that's not guaranteed, so we sort here
        assert len(res) == 10

        for name, _ in res:
            assert name == expected_name

        for (_, a), (_, b) in zip(res, res[1:]):
            diff = b - a
            assert (0.9 * period) < diff < (1.1 * period)

    tasks = [
        f()
        for _ in range(10)
        for f in [foo_1, foo_2]
    ]

    res = await asyncio.gather(*tasks)
    assert len(res) == 20

    foo_1_res = res[0::2]
    foo_2_res = res[1::2]

    _assert_results('foo_1', foo_1_res)
    _assert_results('foo_2', foo_2_res)


@pytest.mark.asyncio
async def test_async_throttle_max_per_second__immediate_cases():
    @async_throttle(max_per_second=10)
    async def foo() -> float:
        call_time = time.time()
        return call_time

    # Ensure *first* call is immediate:
    t1 = time.time()
    t2 = await foo()
    assert 0.0 <= t2 - t1 < 0.01

    # Ensure *second* call is delayed:
    t1 = time.time()
    t2 = await foo()
    assert 0.09 <= t2 - t1 < 0.11

    # Ensure call is immediate if *enough time* has passed already:
    await asyncio.sleep(0.3)
    t1 = time.time()
    t2 = await foo()
    assert 0.0 <= t2 - t1 < 0.01

    # Ensure we delay *again*, if called without delay:
    t1 = time.time()
    t2 = await foo()
    assert 0.09 <= t2 - t1 < 0.11


def test_tool_param_passthrough():
    async def foo(a: int, b: str) -> float:
        """
        a func foo
        :param: a: int: the a param is an int
        :param: b: str: the b param is a str
        """
        return 3.14

    correct = (
        'a func foo',
        [
            {
                'name': 'a',
                'type': 'int',
                'description': 'the a param is an int',
            },
            {
                'name': 'b',
                'type': 'str',
                'description': 'the b param is a str',
            },
        ],
    )

    assert get_name(foo) == 'foo'
    assert get_tool_params(foo) == correct

    d1 = async_throttle(max_concurrent=2)
    d2 = async_throttle(max_per_second=10)
    d3 = async_throttle(max_concurrent=2, max_per_second=10)

    for d in [d1, d2, d3]:
        assert get_name(d(foo)) == 'foo'
        assert get_tool_params(d(foo)) == correct
