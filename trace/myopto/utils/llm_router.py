# trace/myopto/utils/llm_router.py
"""
LLM Router

Routes LLM calls to appropriate backends based on role configuration.
Supports OpenAI, Anthropic, and LocalSLM backends.
"""

from typing import Any, Callable, Dict, Optional
from .localslm import LocalSLM
from .usage import MODEL_REGISTRY, PRICING_PER_TOKEN, resolve_model_name


# --------------------------------------------------
# Role Configuration Store
# --------------------------------------------------

_role_configs: Dict[str, Dict[str, Any]] = {}
_llm_instances: Dict[str, Any] = {}


# --------------------------------------------------
# Configuration API
# --------------------------------------------------

def set_role_models(**kwargs):
    """
    Configure models for arbitrary roles.
    
    Usage:
        set_role_models(executor="qwen2-0.5b", optimizer="gpt-4o-mini", judge="tinyllama")
    
    Args:
        **kwargs: role_name=model_name pairs (any string as role name)
    """
    for role, model in kwargs.items():
        _set_role(role, model)


def _set_role(role: str, model: str):
    """Set up a role with the given model."""
    # Resolve short name to full name
    full_model = resolve_model_name(model)
    
    # Determine backend from model name
    if model in PRICING_PER_TOKEN and PRICING_PER_TOKEN[model]["prompt"] == 0.0:
        backend = "LocalSLM"
    elif "gpt" in model or "gpt" in full_model:
        backend = "OpenAI"
    elif "claude" in model or "claude" in full_model:
        backend = "Anthropic"
    else:
        backend = "LocalSLM"  # Default to local for unknown models
    
    _role_configs[role] = {
        "backend": backend,
        "model": full_model,
        "short_name": model,
        "max_tokens": 1000,
    }
    
    # Clear cached instance
    if role in _llm_instances:
        del _llm_instances[role]


def set_role_config(role: str, **kwargs):
    """
    Configure an LLM role with detailed options.
    
    Args:
        role: Role name (any string)
        **kwargs: Configuration options
    """
    if role not in _role_configs:
        _role_configs[role] = {}
    _role_configs[role].update(kwargs)
    if role in _llm_instances:
        del _llm_instances[role]


def get_role_config(role: str) -> Dict[str, Any]:
    """Get configuration for a role."""
    return _role_configs.get(role, {}).copy()


# --------------------------------------------------
# LLM Instance Management
# --------------------------------------------------

def get_llm(role: str = "executor") -> Callable:
    """
    Get an LLM callable for a role.
    
    Args:
        role: Role name
        
    Returns:
        Callable that takes prompt and returns response
    """
    if role in _llm_instances:
        return _llm_instances[role]
    
    config = _role_configs.get(role, _role_configs["executor"])
    backend = config.get("backend", "OpenAI")
    
    if backend == "LocalSLM":
        instance = _create_local_slm(config)
    elif backend == "OpenAI":
        instance = _create_openai_llm(config)
    elif backend == "Anthropic":
        instance = _create_anthropic_llm(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    _llm_instances[role] = instance
    return instance


def _create_local_slm(config: Dict[str, Any]) -> Callable:
    """Create LocalSLM instance."""
    slm = LocalSLM(
        model_name=config.get("model", "Qwen/Qwen2-0.5B-Instruct"),
        device=config.get("device", "mps"),
        use_mlx=config.get("use_mlx", True),  # Default True for Apple Silicon
        torch_dtype=config.get("torch_dtype", "float16"),
    )
    
    max_tokens = config.get("max_new_tokens", config.get("max_tokens", 150))
    
    def call_slm(prompt: str, **kwargs) -> str:
        return slm.generate(
            prompt, 
            max_new_tokens=kwargs.get("max_new_tokens", max_tokens)
        )
    
    return call_slm


def _create_openai_llm(config: Dict[str, Any]) -> Callable:
    """Create OpenAI LLM callable."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    model = config.get("model", "gpt-4o-mini")
    max_tokens = config.get("max_tokens", 1000)
    
    client = openai.OpenAI()
    
    def call_openai(prompt: str, **kwargs) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", max_tokens),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.choices[0].message.content
    
    return call_openai


def _create_anthropic_llm(config: Dict[str, Any]) -> Callable:
    """Create Anthropic LLM callable."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    model = config.get("model", "claude-3-haiku-20240307")
    max_tokens = config.get("max_tokens", 1000)
    
    client = anthropic.Anthropic()
    
    def call_anthropic(prompt: str, **kwargs) -> str:
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", max_tokens),
        )
        return response.content[0].text
    
    return call_anthropic


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def reset_router():
    """Reset router state."""
    global _llm_instances
    _llm_instances = {}


def list_backends() -> list:
    """List available backends."""
    return ["OpenAI", "Anthropic", "LocalSLM"]