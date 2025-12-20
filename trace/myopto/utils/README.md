# Trace LLM Utilities

This module provides flexible LLM backends for the Trace framework, supporting both cloud APIs and local small language models (SLMs).

## Quick Start

```python
from myopto.utils import set_role_models, get_llm, llm_json

# Configure models for each role
set_role_models(
    executor="gpt-4o-mini",
    optimizer="gpt-4o",
    metaoptimizer="gpt-4o",
)

# Get LLM instance and call
llm = get_llm("executor")
response = llm(messages=[{"role": "user", "content": "Hello!"}])
```

## Backends

### LiteLLM (Default)

Supports all cloud providers via [LiteLLM](https://github.com/BerriAI/litellm).

```python
from myopto.utils import set_role_models

# OpenAI
set_role_models(executor="gpt-4o-mini")

# Anthropic
set_role_models(optimizer="claude-3-5-sonnet-latest")

# Azure
set_role_models(executor="azure/gpt-4o")
```

### LocalSLM (Local Models)

Run small language models locally using transformers or MLX (Apple Silicon optimized).

**Recommended Models for MacBook Air M1:**
| Model | Size | Speed | Memory | Best For |
|-------|------|-------|--------|----------|
| Qwen/Qwen2-0.5B-Instruct | 500M | 20-40 tok/s | 1-2 GB | General tasks |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 15-30 tok/s | 2-3 GB | Conversational |
| microsoft/phi-1_5 | 1.3B | 15-25 tok/s | 2-3 GB | Technical/Math |

```python
from myopto.utils import set_role_models, set_role_config

# Basic local SLM setup
set_role_models(
    executor="Qwen/Qwen2-0.5B-Instruct",
    optimizer="Qwen/Qwen2-0.5B-Instruct",
    backend="LocalSLM",
)

# Fine-grained control
set_role_config(
    "executor",
    backend="LocalSLM",
    model="Qwen/Qwen2-0.5B-Instruct",
    device="mps",  # mps (Apple), cuda, or cpu
    max_new_tokens=200,
    torch_dtype="float16",
)

# MLX backend (faster on Apple Silicon)
set_role_config(
    "executor",
    backend="LocalSLM",
    model="mlx-community/Qwen2-0.5B-Instruct",
    use_mlx=True,
)
```

**Installation for LocalSLM:**
```bash
# Standard transformers
pip install torch transformers accelerate

# MLX (Apple Silicon only, faster)
pip install mlx mlx-lm
```

### CustomLLM (OpenAI-Compatible Endpoints)

For local servers, proxies, or any OpenAI-compatible API.

```python
from myopto.utils import set_role_config

# LiteLLM proxy server
set_role_config(
    "executor",
    backend="CustomLLM",
    model="gpt-4o",
    base_url="http://localhost:4000",
    api_key="sk-xxx",
)

# vLLM server
set_role_config(
    "executor",
    backend="CustomLLM",
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1",
)
```

## JSON Schema Output

Use `llm_json()` for structured outputs:

```python
from myopto.utils import llm_json

result = llm_json(
    "Extract the person's name and age from: John is 30 years old.",
    json_schema={
        "name": "person_info",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        },
        "strict": True
    }
)
# result = {"name": "John", "age": 30}
```

### Backend-Specific Behavior

| Backend | JSON Schema Handling |
|---------|---------------------|
| **LiteLLM** | Native `response_format` with `json_schema` |
| **CustomLLM** | Native `response_format` with `json_schema` |
| **LocalSLM** | JSON instruction injected into prompt (no `response_format`) |

This design ensures:
- OpenAI/ChatGPT models use their native structured output capability
- SLMs don't receive unsupported `response_format` parameters

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRACE_DEFAULT_LLM_BACKEND` | Default backend | `LiteLLM` |
| `TRACE_LITELLM_MODEL` | Default LiteLLM model | `gpt-4o` |
| `TRACE_LLM_MODEL_<ROLE>` | Role-specific model | - |
| `TRACE_LOCALSLM_MODEL` | Default local model | `Qwen/Qwen2-0.5B-Instruct` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |

## Migration from Previous Version

The new system is backward compatible. Existing code using `LLM()` or environment variables will continue to work.

**New recommended approach:**
```python
# Before (still works)
from myopto.utils.llm import LLM
llm = LLM(role="executor")

# After (recommended)
from myopto.utils import set_role_models, get_llm
set_role_models(executor="gpt-4o-mini")
llm = get_llm("executor")
```

**Using local SLMs:**
```python
# Set backend to LocalSLM
import os
os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LocalSLM"
os.environ["TRACE_LOCALSLM_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"

# Or use the router API (cleaner)
from myopto.utils import set_role_models
set_role_models(
    executor="Qwen/Qwen2-0.5B-Instruct",
    optimizer="Qwen/Qwen2-0.5B-Instruct",
    backend="LocalSLM",
)
```

## API Reference

### Router Functions

- `set_role_models(executor=None, optimizer=None, metaoptimizer=None, backend=None, key=None)` - Configure models by role
- `set_role_config(role, backend=None, model=None, ...)` - Fine-grained role config
- `get_llm(role="executor", **kwargs)` - Get LLM instance for role
- `clear_role_config(role=None)` - Clear config (all if role=None)

### Call Helpers

- `llm_json(prompt, json_schema, role="executor", ...)` - Get JSON output
- `llm_text(prompt, role="executor", ...)` - Get text output
- `extract_text(response)` - Extract text from any response format
- `try_parse_json(text)` - Parse JSON from text with fence stripping

### Usage Tracking

- `configure_usage(enabled=True)` - Enable/disable tracking
- `reset_usage()` - Reset counters
- `get_total_cost()` - Get total USD cost
- `get_cost_by_role()` - Get costs by role
- `get_tokens_by_role()` - Get tokens by role
- `get_usage_summary()` - Get full summary dict
- `print_usage_summary()` - Print formatted summary