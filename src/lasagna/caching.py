import asyncio
import time
import functools

from typing import Callable, Union, List, Awaitable

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


def cached_agent(
    query_record: Callable[[CacheKey], Awaitable[Union[CacheRecord, None]]],
    should_cache_record: Callable[[Model, List[AgentRun], AgentRun], Awaitable[bool]],
    save_record: Callable[[CacheKey, CacheRecord], Awaitable[None]],
    simulate_time: bool,
) -> Callable[[AgentCallable], AgentCallable]:
    def decorator(agent: AgentCallable) -> AgentCallable:
        @functools.wraps(agent, assigned=['__module__', '__name__', '__qualname__', '__doc__'])
        async def new_agent(model: Model, event_callback: EventCallback, prev_runs: List[AgentRun]) -> AgentRun:
            hash = _hash_agent_runs(model, prev_runs)
            old_cached_record = await query_record(hash)
            start_time = time.time()

            if old_cached_record is not None:
                for event in old_cached_record['events']:
                    if simulate_time:
                        to_wait = event['delta_time'] - (time.time() - start_time)
                        if to_wait > 0.0:
                            await asyncio.sleep(to_wait)
                    await event_callback(event['event'])
                return old_cached_record['run']

            captured_events: List[CacheEventPayload] = []
            async def event_callback_wrapper(event: EventPayload) -> None:
                captured_events.append({
                    'delta_time': time.time() - start_time,
                    'event': event,
                })
                await event_callback(event)

            run = await agent(model, event_callback_wrapper, prev_runs)

            if await should_cache_record(model, prev_runs, run):
                new_cached_record: CacheRecord = {
                    'events': captured_events,
                    'run': run,
                }
                await save_record(hash, new_cached_record)

            return run

        return new_agent

    return decorator


def _hash_agent_runs(model: Model, runs: List[AgentRun]) -> CacheKey:
    return ''  # TODO (also LOG)
