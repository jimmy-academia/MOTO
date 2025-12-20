# trace/myopto/utils/__init__.py
"""
LLM utilities for Trace.

Quick Start:
    from myopto.utils import set_role_models, get_llm
    
    # Configure models
    set_role_models(executor="gpt-4o-mini", optimizer="gpt-4o")
    
    # Or use local SLMs
    set_role_models(
        executor="Qwen/Qwen2-0.5B-Instruct",
        backend="LocalSLM"
    )
    
    # Get LLM and call
    llm = get_llm("executor")
    response = llm(messages=[{"role": "user", "content": "Hello"}])
    
    # For JSON output, use response_format directly:
    response = llm(
        messages=[{"role": "user", "content": "Extract name and age from: John is 30"}],
        response_format={"type": "json_schema", "json_schema": {...}}
    )

Backends:
    - LiteLLM: Cloud providers (OpenAI, Anthropic, etc.) - supports json_schema
    - CustomLLM: OpenAI-compatible endpoints - supports json_schema
    - LocalSLM: Local models via transformers/MLX
    - AutoGen: AutoGen wrapper
"""
from myopto.utils.llm_router import (
    # Router API
    set_role_models,
    set_role_config,
    clear_role_config,
    get_role_config,
    get_llm,
    # LLM call helpers
    llm_text,
    extract_text,
    try_parse_json,
)

from myopto.utils.usage import (
    configure_usage,
    reset_usage,
    get_total_cost,
    get_cost_by_role,
    get_tokens_by_role,
    get_usage_summary,
    print_usage_summary,
    track_response,
    UsageRecord,
)

from myopto.utils.llm import (
    LLM,
    LiteLLM,
    CustomLLM,
    AutoGenLLM,
    AbstractModel,
)

# LocalSLM is optional (requires transformers or mlx)
try:
    from myopto.utils.llm import LocalSLM
except ImportError:
    LocalSLM = None

__all__ = [
    # Router
    "set_role_models",
    "set_role_config",
    "clear_role_config",
    "get_role_config",
    "get_llm",
    # LLM call helpers
    "llm_text",
    "extract_text",
    "try_parse_json",
    # Usage tracking
    "configure_usage",
    "reset_usage",
    "get_total_cost",
    "get_cost_by_role",
    "get_tokens_by_role",
    "get_usage_summary",
    "print_usage_summary",
    "track_response",
    "UsageRecord",
    # Backend classes
    "LLM",
    "LiteLLM",
    "CustomLLM",
    "AutoGenLLM",
    "LocalSLM",
    "AbstractModel",
]