import asyncio
import functools
import time

from typing import (
    Optional, Union,
    TypeVar, Callable, Awaitable,
)

from typing_extensions import ParamSpec


P = ParamSpec('P')  # <-- for generic programming
T = TypeVar('T')    # <-- for generic programming


def async_throttle(
    max_concurrent: Optional[int] = None,
    max_per_second: Optional[float] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions to throttle their invocations to stay
    under the specified `max_concurrent` and/or `max_per_second`.
    """
    if max_concurrent is None and max_per_second is None:
        raise ValueError(f"you must specify *at least one* of `max_concurrent` or `max_per_second`")

    elif max_concurrent is not None and max_per_second is not None:
        def decorator(wrapped: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
            tc = _async_throttle_concurrent(max_concurrent)
            tr = _async_throttle_rate(max_per_second)
            return tc(tr(wrapped))  # <-- this is the correct order to apply the decorators
        return decorator

    elif max_concurrent is not None:
        return _async_throttle_concurrent(max_concurrent)

    elif max_per_second is not None:
        return _async_throttle_rate(max_per_second)

    else:
        raise RuntimeError("unreachable")


def _async_throttle_concurrent(
    max_concurrent: int,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(wrapped: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        sem = asyncio.Semaphore(max_concurrent)

        @functools.wraps(wrapped)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with sem:
                return await wrapped(*args, **kwargs)

        return wrapper

    return decorator


def _async_throttle_rate(
    max_per_second: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(wrapped: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        period = 1.0 / max_per_second
        last_call: Union[float, None] = None

        @functools.wraps(wrapped)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal last_call
            now = time.time()
            if last_call is None:
                last_call = now
                delay = 0.0
            else:
                this_call = last_call + period
                delay = max(0.0, this_call - now)
                last_call = max(this_call, now)
            if delay > 0.0:
                await asyncio.sleep(delay)
            return await wrapped(*args, **kwargs)

        return wrapper

    return decorator
