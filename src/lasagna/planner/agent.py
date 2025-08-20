"""
The Lasagna Recursive Planning Agent (LRPA)

This planning agent ("planner") recursively breaks down a task until
each subtask (or subsubtask, or subsubsubtask, etc, as deep as needed)
becomes "trivial" (that is the recursion base case). It then aggregates
answers back up the recursion tree using the "answer agent" at each level
to build up higher-level answers until the final (original) task is
complete. It performs a DFS traversal, injecting answers to subtasks
into the history to inform future subtasks (at the same level) and
higher-level subtasks (when unwinding).

It uses the Lasagna "layered agent" ideas (both the hierarchical AgentRun
data structure as well as the agents-calling-agents idea). Recursive
planning agents were a core idea I had when I began this library, which
is why its design is the way it is, so it's great to get this LRPA
implementation started!

I still have some ideas left to implement, like:

1. Backtracking: What if the task list turns out to be bad? Sometimes
   you don't find that out until you are half-way through the task.
   A backtracking feature would allow the agent to bail on a plan
   and start over with a new plan (using what it learned to inform
   the new plan).

2. Loopback: Unlike backtracking, loopback keeps the _same_ plan but
   just returns back to a previous step that it found out needs revision.
   It then does the revision, and recomputes the later steps with those
   new/fixed results.

3. Retries: Sometimes you just need to try again. Same plan, same approach,
   just give it another try.

4. History Exclusion: Some subtasks don't need the full history, thus those
   subtasks should be launched _without_ the full history. This will help
   in three ways: (1) fewer tokens to pay for, (2) faster responses,
   (3) better answers (it's good not to include context when it's not needed).
   The planner will identify subtasks that can stand on their own,
   and kick those off without the full history. Another benefit: We can
   execute those in parallel! A more _generalized_ idea here is to generate
   subtask dependencies, which would enable optimal parallelization and
   precise history inclusion.

5. Watchdogs: After each generation (be it a subtask list, an answer, etc)
   we could have watchdog models judge those output and accept/reject them.
   Accepted outputs would move through the process as normal. Rejected outputs
   would be regenerated with injected feedback from the watchdog.

6. Timebound: Part of the generation should be governed by how much _time_
   is left to complete the original task. As time approaches the deadline,
   the generation should do less diving and more unwinding. A deadline is
   optional, but it may be helpful for open-ended tasks that could go on
   forever. Similar a John Cleese's point - creativity comes from reveling
   in the unknown for as long as you can, but also meeting deadlines.
   (PS: This feature may be implemented as a watchdog.)

7. Pins: "Put a pin in that." Since the planner is DFS, there's a danger
   in diving forever into the first subtask. The planner should be able
   to "pin" something and come back to it later (if times allows), so that
   it can begin unwinding a task and move on.

8. Standard Tooling:
   - web search,
   - memory (maybe via a file-system API),
      - namespaced to the task and its parents
   - ability to execute Python code,
   - etc.
"""

from ..types import (
    Model,
    EventCallback,
    AgentRun,
    Message,
    AgentCallable,
    BoundAgentCallable,
    ExtractionType,
)
from ..agent_util import (
    override_system_prompt,
    build_simple_agent,
    recursive_extract_messages,
    extraction,
    MessageExtractor,
)
from ..agent_util import chained_runs
from ..pydantic_util import result_to_string

from .util import extract_is_trivial
from . import default_prompts

import copy
from typing import List, Tuple, Callable, Type, AsyncIterable, Awaitable


def build_default_planning_agent(
    *,
    binder: Callable[[AgentCallable], BoundAgentCallable],
    tools: List[Callable] = [],
    max_tool_iters: int = 5,
    max_depth: int = 4,
) -> BoundAgentCallable:
    return build_planning_agent(
        binder = binder,
        answer_agent = binder(build_simple_agent(
            'answer_agent',
            tools = tools,
            max_tool_iters = max_tool_iters,
            message_extractor = _extract_messages_lrpa_aware,
        )),
        system_prompt = default_prompts.append_tool_spec_to_prompt(
            default_prompts.PLANNING_AGENT_SYSTEM_PROMPT,
            tools,
        ),
        message_extractor = _extract_messages_lrpa_aware,
        subtask_input_generator = default_prompts.subtask_input_generator,
        answer_input_prompt = default_prompts.answer_input_prompt,
        extraction_type = default_prompts.PlanOutput,
        max_depth = max_depth,
    )


def build_planning_agent(
    *,
    binder: Callable[[AgentCallable], BoundAgentCallable],
    answer_agent: BoundAgentCallable,
    system_prompt: str,
    message_extractor: MessageExtractor,
    subtask_input_generator: Callable[[AgentRun], AsyncIterable[List[AgentRun]]],
    answer_input_prompt: Callable[[AgentRun], Awaitable[List[AgentRun]]],
    extraction_type: Type[ExtractionType],
    max_depth: int,
) -> BoundAgentCallable:
    @binder
    async def planning_agent(
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        # We want to be **pure**, but also not copy more than we need to.
        # So, `prev_runs` will be minimally copied, and `planning_agent_chain`
        # will be inserted into the copied `prev_runs` in the correct location.
        # That is: Modifying `planning_agent_chain` will modify the *copied*
        #          `prev_runs`, which is fine since it's a copy. The code below
        #          will append its output into `planning_agent_chain`.
        prev_runs, planning_agent_chain, depth = _minimal_copy_lrpa_aware(prev_runs)

        # All Lasagna agents return a *new* AgentRun representing
        # the new content generated by this agent. We'll keep track
        # of all the content generated by this agent in the `output_runs`
        # list below.
        assert planning_agent_chain['type'] == 'chain'
        output_runs = planning_agent_chain['runs']

        messages_orig = message_extractor(prev_runs)

        messages_new_system_prompt = override_system_prompt(
            messages_orig,
            system_prompt = system_prompt,
        )

        extraction_message, extraction_result = await model.extract(
            event_callback,
            messages = messages_new_system_prompt,
            extraction_type = extraction_type,
        )
        extraction_run = extraction('extraction_run', [extraction_message], extraction_result)
        output_runs.append(extraction_run)

        is_trivial = extract_is_trivial(extraction_result)

        if not is_trivial and depth < max_depth:
            subtask_chain_of_chains = chained_runs('subtask_chain_of_chains', [])
            output_runs.append(subtask_chain_of_chains)
            async for subtask_input in subtask_input_generator(extraction_run):
                subtask_chain = chained_runs('subtask_chain', subtask_input)
                subtask_chain_of_chains['runs'].append(subtask_chain)
                subtask_run = await planning_agent(
                    event_callback,
                    prev_runs,
                )
                subtask_chain['runs'].append(subtask_run)

        answer_chain = chained_runs('answer_chain', await answer_input_prompt(extraction_run))
        output_runs.append(answer_chain)
        answer_run = await answer_agent(
            event_callback,
            prev_runs,
        )
        answer_chain['runs'].append(answer_run)

        return planning_agent_chain

    return planning_agent


_PLANNER_RECURSION_AGENTS = ['planning_agent_chain', 'subtask_chain_of_chains', 'subtask_chain']


def _minimal_copy_lrpa_aware(
    prev_runs: List[AgentRun],
    depth: int = 0,
) -> Tuple[List[AgentRun], AgentRun, int]:
    """
    Minimally copy `prev_runs` and add a 'planning_agent_chain'
    to the end of it at the deepest depth. This implementation
    assumes the structure created by `planning_agent`, where the
    new output always goes at the end of the deepest planning chain.
    """
    if not prev_runs:
        raise ValueError("empty `prev_runs`")

    last_run = prev_runs[-1]

    if last_run['agent'] in _PLANNER_RECURSION_AGENTS:
        assert last_run['type'] == 'chain'
        last_run_copy = copy.copy(last_run)     # shallow copy
        (
            last_run_copy['runs'],
            deepest_planning_agent_chain,
            deepest_depth,
        ) = _minimal_copy_lrpa_aware(
            last_run_copy['runs'],
            depth + 1 if last_run['agent'] == 'planning_agent_chain' else depth
        )
        prev_runs_copy = copy.copy(prev_runs)   # shallow copy
        prev_runs_copy[-1] = last_run_copy
        return prev_runs_copy, deepest_planning_agent_chain, deepest_depth

    # This is the _first_ call to the planning agent (no recursion yet).
    # Or it's the _deepest_ call. Either way, we're at the base case.
    # So merely append the root of a new planning agent output to `prev_runs`.
    planning_agent_chain = chained_runs('planning_agent_chain', [])
    prev_runs_copy = copy.copy(prev_runs)  # shallow copy
    prev_runs_copy.append(planning_agent_chain)
    return prev_runs_copy, planning_agent_chain, depth


def _extract_messages_lrpa_aware(
    prev_runs: List[AgentRun],
    depth: int = 0,
) -> List[Message]:
    """
    This function extracts only _relevant_ messages. If a subtask has an
    answer, the subtasks below it are removed (since the answer is what
    matters in this case). If a subtask does not yet have an answer, its
    full surrounding context is preserved so that an answer can be reached.
    Stated another way: The subtask tree is pruned to keep only the messages
    relevant to obtain the next answer.
    NOTE: The implementation of this function is tied to and assumes this
    is called from the `planning_agent`. That is, it assumes the hierarchical/
    recursive `prev_runs` in the input match that built by `planning_agent`.
    """
    messages: List[Message] = []

    for run in prev_runs:
        if run['agent'] == 'planning_agent_chain':
            # Special case! We want to only recurse when the answer hasn't been found.
            assert run['type'] == 'chain'
            pa_runs = run['runs']
            if len(pa_runs) > 0 and pa_runs[-1]['agent'] == 'answer_chain':
                answer_chain = pa_runs[-1]
                assert answer_chain['type'] == 'chain'
                answer_ai_messages = [
                    m
                    for m in recursive_extract_messages(
                        answer_chain,
                        from_tools=False,
                        from_extraction=False,
                    )
                    if m['role'] == 'ai'
                ]
                if answer_ai_messages:
                    messages.extend(answer_ai_messages)
                    continue

        if run['type'] == 'messages':
            messages.extend(
                run['messages'],
            )

        elif run['type'] == 'chain' or run['type'] == 'parallel':
            messages.extend(
                _extract_messages_lrpa_aware(run['runs'], depth = depth + 1),
            )

        elif run['type'] == 'extraction':
            messages.append({
                'role': 'ai',
                'text': result_to_string(run['result']),
            })

        else:
            raise RuntimeError(f"unknown type: {run['type']}")

    #if depth == 0:
    #    from ..agent_util import to_str
    #    print(to_str(prev_runs))
    #    import json
    #    print(json.dumps(messages, indent=2))
    #    print('-' * 60)

    return messages
