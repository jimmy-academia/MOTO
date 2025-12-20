# trace/myopto/utils/llm_router.py
"""
LLM Router

Routes LLM calls to appropriate backends based on role configuration.
Supports OpenAI, Anthropic, and LocalSLM backends.

Returns OpenAI-compatible response objects for all backends.
Accepts both `messages` (List[Dict]) and `prompt` (str) for flexibility.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from .localslm import LocalSLM
    from .usage import MODEL_REGISTRY, PRICING_PER_TOKEN, resolve_model_name
except ImportError:
    LocalSLM = None
    MODEL_REGISTRY = {}
    PRICING_PER_TOKEN = {}
    def resolve_model_name(m): return m


# ----------------------------------------------------------------------------------------------------
# Response Types (OpenAI-compatible)
# ----------------------------------------------------------------------------------------------------

@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str = "stop"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = 0
    model: str = "local"
    choices: List[Choice] = None
    usage: Usage = None
    
    def __post_init__(self):
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = Usage()


# ----------------------------------------------------------------------------------------------------
# Role Configuration Store
# ----------------------------------------------------------------------------------------------------

_role_configs: Dict[str, Dict[str, Any]] = {}
_llm_instances: Dict[str, Any] = {}


# ----------------------------------------------------------------------------------------------------
# Configuration API
# ----------------------------------------------------------------------------------------------------

def set_role_models(**kwargs):
    """
    Configure models for arbitrary roles.
    
    Usage:
        set_role_models(executor="qwen2-0.5b", optimizer="gpt-4o-mini", judge="tinyllama")
    """
    for role, model in kwargs.items():
        _set_role(role, model)


def _set_role(role: str, model: str):
    """Set up a role with the given model."""
    full_model = resolve_model_name(model)
    
    if model in PRICING_PER_TOKEN and PRICING_PER_TOKEN[model]["prompt"] == 0.0:
        backend = "LocalSLM"
    elif "gpt" in model or "gpt" in full_model:
        backend = "OpenAI"
    elif "claude" in model or "claude" in full_model:
        backend = "Anthropic"
    else:
        backend = "LocalSLM"
    
    _role_configs[role] = {
        "backend": backend,
        "model": full_model,
        "short_name": model,
        "max_tokens": 1000,
    }
    
    if role in _llm_instances:
        del _llm_instances[role]


def set_role_config(role: str, **kwargs):
    """Configure an LLM role with detailed options."""
    if role not in _role_configs:
        _role_configs[role] = {}
    _role_configs[role].update(kwargs)
    if role in _llm_instances:
        del _llm_instances[role]


def get_role_config(role: str) -> Dict[str, Any]:
    """Get configuration for a role."""
    return _role_configs.get(role, {}).copy()


# ----------------------------------------------------------------------------------------------------
# LLM Instance Management
# ----------------------------------------------------------------------------------------------------

def get_llm(role: str = "executor") -> Callable:
    """
    Get an LLM callable for a role.
    
    Args:
        role: Role name
        
    Returns:
        Callable with dual interface:
          - llm(messages=[...], **kwargs) -> ChatCompletion
          - llm(prompt="...", **kwargs) -> ChatCompletion
    """
    if role in _llm_instances:
        return _llm_instances[role]
    
    config = _role_configs.get(role)
    if config is None:
        config = _role_configs.get("executor", {"backend": "OpenAI", "model": "gpt-4o-mini"})
    
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


# ----------------------------------------------------------------------------------------------------
# Input Normalization
# ----------------------------------------------------------------------------------------------------

def _normalize_input(
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Normalize input to messages format.
    
    Accepts either:
      - messages: List[Dict] with role/content
      - prompt: str (converted to single user message)
    """
    if messages is not None:
        return messages
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    return []


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert messages list to a single prompt string for local models."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------------------------------
# Backend Factories
# ----------------------------------------------------------------------------------------------------

def _create_local_slm(config: Dict[str, Any]) -> Callable:
    """Create LocalSLM callable with OpenAI-compatible interface."""
    if LocalSLM is None:
        raise ImportError("LocalSLM not available")
    
    slm = LocalSLM(
        model_name=config.get("model", "Qwen/Qwen2-0.5B-Instruct"),
        device=config.get("device", "mps"),
        use_mlx=config.get("use_mlx", True),
        torch_dtype=config.get("torch_dtype", "float16"),
    )
    model_name = config.get("model", "local")
    default_max_tokens = config.get("max_new_tokens", config.get("max_tokens", 150))
    
    def call_slm(
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        *,
        max_tokens: int = None,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ChatCompletion:
        msgs = _normalize_input(messages, prompt)
        prompt_str = _messages_to_prompt(msgs)
        
        tokens = max_new_tokens or max_tokens or default_max_tokens
        content = slm.generate(prompt_str, max_new_tokens=tokens, temperature=temperature)
        
        return ChatCompletion(
            model=model_name,
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            usage=Usage()
        )
    
    return call_slm


def _create_openai_llm(config: Dict[str, Any]) -> Callable:
    """Create OpenAI LLM callable - returns raw OpenAI response."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    model = config.get("model", "gpt-4o-mini")
    default_max_tokens = config.get("max_tokens", 1000)
    
    client = openai.OpenAI()
    
    def call_openai(
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        *,
        max_tokens: int = None,
        temperature: float = 0.7,
        response_format: Dict = None,
        n: int = 1,
        **kwargs
    ) -> Any:
        msgs = _normalize_input(messages, prompt)
        
        call_kwargs = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens or default_max_tokens,
            "temperature": temperature,
            "n": n,
        }
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        
        return client.chat.completions.create(**call_kwargs)
    
    return call_openai


def _create_anthropic_llm(config: Dict[str, Any]) -> Callable:
    """Create Anthropic LLM callable - returns OpenAI-compatible wrapper."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    model = config.get("model", "claude-3-haiku-20240307")
    default_max_tokens = config.get("max_tokens", 1000)
    
    client = anthropic.Anthropic()
    
    def call_anthropic(
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        *,
        max_tokens: int = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ChatCompletion:
        msgs = _normalize_input(messages, prompt)
        
        # Extract system message if present
        system_content = None
        user_messages = []
        for msg in msgs:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append(msg)
        
        call_kwargs = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens or default_max_tokens,
        }
        if system_content:
            call_kwargs["system"] = system_content
        
        response = client.messages.create(**call_kwargs)
        content = response.content[0].text if response.content else ""
        
        usage = Usage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        
        return ChatCompletion(
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            usage=usage
        )
    
    return call_anthropic


# ----------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------

def extract_text(response: Any) -> str:
    """Extract text content from various response formats."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    
    # OpenAI-style response object
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content or ""
        if hasattr(choice, "text"):
            return choice.text or ""
    
    # Dict-style response
    if isinstance(response, dict):
        if "choices" in response:
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        if "content" in response:
            return response["content"]
        if "text" in response:
            return response["text"]
    
    return str(response)


def try_parse_json(text: str) -> dict:
    """Parse JSON from text, stripping markdown fences if present."""
    import json
    import re
    
    text = text.strip()
    
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


# ----------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------

def reset_router():
    """Reset router state."""
    global _llm_instances
    _llm_instances = {}


def clear_role_config(role: str = None):
    """Clear config for a role (or all if role=None)."""
    global _role_configs, _llm_instances
    if role is None:
        _role_configs = {}
        _llm_instances = {}
    else:
        _role_configs.pop(role, None)
        _llm_instances.pop(role, None)


def list_backends() -> list:
    """List available backends."""
    return ["OpenAI", "Anthropic", "LocalSLM"]


__all__ = [
    "set_role_models",
    "set_role_config",
    "get_role_config",
    "get_llm",
    "reset_router",
    "clear_role_config",
    "list_backends",
    "extract_text",
    "try_parse_json",
    "ChatCompletion",
    "Choice",
    "Message",
    "Usage",
]