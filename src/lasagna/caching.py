import asyncio
import time
import functools

from typing import Callable, Union, List, Awaitable, Dict

from .types import (
    AgentCallable,
    AgentRun,
    Model,
    EventCallback,
    EventPayload,
    CacheKey,
    CacheRecord,
    CacheEventPayload,
)

from .util import get_name, recursive_hash

from . import __version__

import logging

_LOG = logging.getLogger(__name__)


async def _hash_agent_runs(model: Model, runs: List[AgentRun]) -> CacheKey:
    model_config_hash = model.config_hash()
    seed = f"__version__{__version__}__model__{model_config_hash}"
    def _do() -> CacheKey:
        return recursive_hash(seed, runs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)


def cached_agent(
    query_record: Callable[[CacheKey], Awaitable[Union[CacheRecord, None]]],
    should_cache_record: Callable[[Model, List[AgentRun], AgentRun], Awaitable[bool]],
    save_record: Callable[[CacheKey, CacheRecord], Awaitable[None]],
    simulate_time: bool,
    hash_function: Callable[[Model, List[AgentRun]], Awaitable[CacheKey]] = _hash_agent_runs,
) -> Callable[[AgentCallable], AgentCallable]:
    def decorator(agent: AgentCallable) -> AgentCallable:
        @functools.wraps(agent, assigned=['__module__', '__qualname__', '__doc__'])
        async def new_agent(model: Model, event_callback: EventCallback, prev_runs: List[AgentRun]) -> AgentRun:
            hash = await hash_function(model, prev_runs)
            old_cached_record = await query_record(hash)
            start_time = time.time()

            if old_cached_record is not None:
                _LOG.info(f"Cache **HIT**: {hash}")
                for event in old_cached_record['events']:
                    if simulate_time:
                        to_wait = event['delta_time'] - (time.time() - start_time)
                        if to_wait > 0.0:
                            await asyncio.sleep(to_wait)
                    await event_callback(event['event'])
                return old_cached_record['run']

            _LOG.info(f"Cache miss: {hash}")

            captured_events: List[CacheEventPayload] = []
            async def event_callback_wrapper(event: EventPayload) -> None:
                captured_events.append({
                    'delta_time': time.time() - start_time,
                    'event': event,
                })
                await event_callback(event)

            run = await agent(model, event_callback_wrapper, prev_runs)

            if await should_cache_record(model, prev_runs, run):
                _LOG.info(f"Will store in cache: {hash}")
                new_cached_record: CacheRecord = {
                    'events': captured_events,
                    'run': run,
                }
                await save_record(hash, new_cached_record)
            else:
                _LOG.info(f"Will *not* store in cache: {hash}")

            return run

        new_agent.__name__ = get_name(agent)

        return new_agent

    return decorator


def in_memory_cached_agent(agent: AgentCallable) -> AgentCallable:
    """
    This is mostly just for demo to show how to use the @cached_agent
    decorator.
    A real system would use an external database, and would probably
    do some sort of LRU thing in the database so you don't cache too much.
    Also consider only caching *short* runs (because long runs are unlikely
    to every see a cache-hit, so it's not worth it).
    """
    cache: Dict[CacheKey, CacheRecord] = {}

    async def in_memory_query_record(key: CacheKey) -> Union[CacheRecord, None]:
        return cache.get(key)

    async def in_memory_should_cache_record(model: Model, prev_runs: List[AgentRun], this_run: AgentRun) -> bool:
        return len(cache) < 1000   # !!! just a demo, don't read into this

    async def in_memory_save_record(key: CacheKey, record: CacheRecord) -> None:
        cache[key] = record

    decorator = cached_agent(
        query_record = in_memory_query_record,
        should_cache_record = in_memory_should_cache_record,
        save_record = in_memory_save_record,
        simulate_time=True,
    )

    return decorator(agent)
