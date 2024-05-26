import pytest

from lasagna.stream import (
    fake_async,
    aenumerate,
    apeek,
    adup,
)

from typing import List


EMPTY: List[int] = []


@pytest.mark.asyncio
async def test_fake_async():
    stream = fake_async(EMPTY)
    vals = [v async for v in stream]
    assert vals == []

    stream = fake_async(['a'])
    vals = [v async for v in stream]
    assert vals == ['a']

    stream = fake_async([1, 5, 'ryan', None, 8])
    vals = [v async for v in stream]
    assert vals == [1, 5, 'ryan', None, 8]


@pytest.mark.asyncio
async def test_aenumerate():
    stream = fake_async(EMPTY)
    vals = [v async for v in aenumerate(stream)]
    assert vals == []

    stream = fake_async([4, 6, 9])
    vals = [v async for v in aenumerate(stream)]
    assert vals == [
        (0, 4),
        (1, 6),
        (2, 9),
    ]


@pytest.mark.asyncio
async def test_apeek():
    stream = fake_async(EMPTY)
    with pytest.raises(ValueError):
        first, stream = await apeek(stream, n=1.0)  # type: ignore

    stream = fake_async(EMPTY)
    with pytest.raises(ValueError):
        first, stream = await apeek(stream, n=-1)

    stream = fake_async(EMPTY)
    first, stream = await apeek(stream, n=0)
    assert first == []
    vals = [v async for v in stream]
    assert vals == []

    stream = fake_async([1, 3, 8])
    first, stream = await apeek(stream, n=0)
    assert first == []
    vals = [v async for v in stream]
    assert vals == [1, 3, 8]

    stream = fake_async(EMPTY)
    with pytest.raises(ValueError):
        first, stream = await apeek(stream, n=1)

    stream = fake_async([8])
    first, stream = await apeek(stream, n=1)
    assert first == [8]
    vals = [v async for v in stream]
    assert vals == [8]

    stream = fake_async([8, 99])
    first, stream = await apeek(stream, n=1)
    assert first == [8]
    vals = [v async for v in stream]
    assert vals == [8, 99]

    stream = fake_async(EMPTY)
    with pytest.raises(ValueError):
        first, stream = await apeek(stream, n=2)

    stream = fake_async([8])
    with pytest.raises(ValueError):
        first, stream = await apeek(stream, n=2)

    stream = fake_async([8, 99])
    first, stream = await apeek(stream, n=2)
    assert first == [8, 99]
    vals = [v async for v in stream]
    assert vals == [8, 99]

    stream = fake_async([8, 99, 100])
    first, stream = await apeek(stream, n=2)
    assert first == [8, 99]
    vals = [v async for v in stream]
    assert vals == [8, 99, 100]


@pytest.mark.asyncio
async def test_adup():
    async def _test(truth):
        stream = fake_async(truth)
        a, b = adup(stream)
        assert [v async for v in a] == truth
        assert [v async for v in b] == truth
        assert [v async for v in a] == []  # <-- should be empty now
        assert [v async for v in b] == []  # <-- should be empty now
    await _test([])
    await _test(['hi'])
    await _test(['hi', 8])
    await _test([None, 'hi', 8])
    await _test([5, None, 'hi', 8])

    async def gen_with_error():
        yield 'hi'
        yield 99
        yield 4 / 0   # <-- will raise ZeroDivisionError
    async def check(q):
        stuff = []
        with pytest.raises(ZeroDivisionError):
            async for v in q:
                stuff.append(v)
        assert stuff == ['hi', 99]
        async for v in q:
            assert False  # <-- q should be empty, thus this isn't hit
    a, b = adup(gen_with_error())
    await check(a)
    await check(b)
    a, b = adup(gen_with_error())
    await check(b)
    await check(a)
