from typing import List

from .types import ModelRecord

from .agent_util import make_model_binder


OPENAI_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'gpt-5.1-2025-11-13',
        'display_name': 'GPT-5.1',
    },
    {
        'formal_name': 'gpt-5-mini-2025-08-07',
        'display_name': 'GPT-5 mini',
    },
    {
        'formal_name': 'gpt-5-nano-2025-08-07',
        'display_name': 'GPT-5 nano',
    },
    {
        'formal_name': 'gpt-4.1-2025-04-14',
        'display_name': 'GPT-4.1',
    },
    {
        'formal_name': 'gpt-5-2025-08-07',
        'display_name': 'GPT-5',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4o-mini-2024-07-18',
        'display_name': 'GPT-4o mini',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4o-2024-11-20',
        'display_name': 'GPT-4o',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4o-2024-08-06',
        'display_name': 'GPT-4o',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4o-2024-05-13',
        'display_name': 'GPT-4o',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4-turbo-2024-04-09',
        'display_name': 'GPT-4',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-3.5-turbo-0125',
        'display_name': 'GPT-3.5',
        'outdated': True,
    },
]

openai_gpt_5_1_binder      = make_model_binder('openai', 'gpt-5.1-2025-11-13')
openai_gpt_5_mini_binder   = make_model_binder('openai', 'gpt-5-mini-2025-08-07')
openai_gpt_5_nano_binder   = make_model_binder('openai', 'gpt-5-nano-2025-08-07')
openai_gpt_4_1_binder      = make_model_binder('openai', 'gpt-4.1-2025-04-14')


ANTHROPIC_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'claude-opus-4-5-20251101',
        'display_name': 'Claude Opus 4.5',
    },
    {
        'formal_name': 'claude-sonnet-4-5-20250929',
        'display_name': 'Claude Sonnet 4.5',
    },
    {
        'formal_name': 'claude-haiku-4-5-20251001',
        'display_name': 'Claude Haiku 4.5',
    },
    {
        'formal_name': 'claude-opus-4-1-20250805',
        'display_name': 'Claude Opus 4.1',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-5-haiku-20241022',
        'display_name': 'Claude Haiku 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'claude-opus-4-20250514',
        'display_name': 'Claude Opus 4',
        'outdated': True,
    },
    {
        'formal_name': 'claude-sonnet-4-20250514',
        'display_name': 'Claude Sonnet 4',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-7-sonnet-20250219',
        'display_name': 'Claude Sonnet 3.7',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-5-sonnet-20240620',
        'display_name': 'Claude Sonnet 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-opus-20240229',
        'display_name': 'Claude Opus 3',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-sonnet-20240229',
        'display_name': 'Claude Sonnet 3',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-haiku-20240307',
        'display_name': 'Claude Haiku 3',
        'outdated': True,
    },
]

anthropic_claude_opus_4_5_binder  = make_model_binder('anthropic', 'claude-opus-4-5-20251101')
anthropic_claude_sonnet_4_5_binder  = make_model_binder('anthropic', 'claude-sonnet-4-5-20250929', strict_tools = True)
anthropic_claude_haiku_4_5_binder = make_model_binder('anthropic', 'claude-haiku-4-5-20251001')


OLLAMA_KNOWN_MODELS: List[ModelRecord] = [
    # We'll only list models here that (1) support tool-calling because
    # that's kind of the whole point of lasagna, and (2) we've tested
    # ourselves and pass our "vibe" test. Users are of course more than
    # welcome to bind *other* Ollama models to their agents, as they see fit.
    {
        'formal_name': 'gpt-oss:120b',
        'display_name': 'OpenAI gpt-oss-120b',
    },
    {
        'formal_name': 'gpt-oss:20b',
        'display_name': 'OpenAI gpt-oss-20b',
    },
    {
        'formal_name': 'llama3.2',
        'display_name': 'Meta Llama 3.2',
    },
    {
        'formal_name': 'mistral-small',
        'display_name': 'Mistral Small',
    },
    {
        'formal_name': 'mistral-large',
        'display_name': 'Mistral Large',
    },
]

ollama_openai_gpt_oss_120b  = make_model_binder('ollama', 'gpt-oss:120b',  num_ctx = 131072)
ollama_openai_gpt_oss_20b   = make_model_binder('ollama', 'gpt-oss:20b',   num_ctx = 131072)
ollama_llama3_2_binder      = make_model_binder('ollama', 'llama3.2',      num_ctx = 131072)
ollama_mistral_small_binder = make_model_binder('ollama', 'mistral-small', num_ctx = 32768)
ollama_mistral_large_binder = make_model_binder('ollama', 'mistral-large', num_ctx = 131072)


BEDROCK_KNOWN_MODELS: List[ModelRecord] = [
    #{
    #    'formal_name': '',  # No US cross-region ID, yet?
    #    'display_name': 'Claude Opus 4.5',
    #},
    {
        'formal_name': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'display_name': 'Claude Sonnet 4.5',
    },
    {
        'formal_name': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'display_name': 'Claude Haiku 4.5',
    },
    {
        'formal_name': 'us.anthropic.claude-opus-4-1-20250805-v1:0',
        'display_name': 'Claude Opus 4.1',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
        'display_name': 'Claude Sonnet 4',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'display_name': 'Claude Haiku 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-opus-4-20250514-v1:0',
        'display_name': 'Claude Opus 4',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'display_name': 'Claude Sonnet 3.7',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'display_name': 'Claude Sonnet 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'display_name': 'Claude Sonnet 3.5 (v2)',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-opus-20240229-v1:0',
        'display_name': 'Claude Opus 3',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-sonnet-20240229-v1:0',
        'display_name': 'Claude Sonnet 3',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-haiku-20240307-v1:0',
        'display_name': 'Claude Haiku 3',
        'outdated': True,
    },
]

bedrock_claude_sonnet_4_5_binder = make_model_binder('bedrock', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
bedrock_claude_haiku_4_5_binder = make_model_binder('bedrock', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')


abstract_binder = make_model_binder('__abstract__', '')
