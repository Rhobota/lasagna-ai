import asyncio

from typing import AsyncIterator, Iterator, TypeVar, Tuple, List, Union, Literal, cast


T = TypeVar('T')  # <-- for generic programming


async def fake_async(source: Iterator[T]) -> AsyncIterator[T]:
    for v in source:
        yield v


async def aenumerate(source: AsyncIterator[T]) -> AsyncIterator[Tuple[int, T]]:
    i = 0
    async for v in source:
        yield i, v
        i += 1


async def apeek(source: AsyncIterator[T], n: int = 1) -> Tuple[List[T], AsyncIterator[T]]:
    if not isinstance(n, int):
        raise ValueError(f"invalid type for `n`: {type(n)}")
    if n < 0:
        raise ValueError(f"invalid value for `n`: {n}")
    if n == 0:
        return [], source

    try:
        first = []
        for _ in range(n):
            v = await source.__anext__()
            first.append(v)
    except StopAsyncIteration:
        raise ValueError(f"this generate had less than {n} items")

    async def generator() -> AsyncIterator[T]:
        for v in first:
            yield v
        async for v in source:
            yield v

    return first, generator()


def adup(source: AsyncIterator[T]) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    q1: asyncio.Queue[Union[Tuple[Literal[True], T], Tuple[Literal[False], Union[None, Exception]]]] = asyncio.Queue()
    q2: asyncio.Queue[Union[Tuple[Literal[True], T], Tuple[Literal[False], Union[None, Exception]]]] = asyncio.Queue()
    async def pull() -> None:
        try:
            async for v in source:
                await q1.put((True, v))
                await q2.put((True, v))
            await q1.put((False, None))
            await q2.put((False, None))
        except Exception as e:
            await q1.put((False, e))
            await q2.put((False, e))
    _ = asyncio.create_task(pull())
    async def gen(q: asyncio.Queue[Union[Tuple[Literal[True], T], Tuple[Literal[False], Union[None, Exception]]]]) -> AsyncIterator[T]:
        while True:
            success, v = await q.get()
            if success:
                v = cast(T, v)   # <-- mypy should *know* this, but it doesn't for some reason
                yield v
            else:
                v = cast(Union[None, Exception], v)   # <-- mypy should *know* this, but it doesn't for some reason
                await q.put((False, v))  # <-- put it back so that we find it again if we try to re-itereate
                if v is not None:
                    raise v
                else:
                    break
    return gen(q1), gen(q2)


async def prefix_stream(prefix_list: List[T], source: AsyncIterator[T]) -> AsyncIterator[T]:
    for v in prefix_list:
        yield v
    async for v in source:
        yield v