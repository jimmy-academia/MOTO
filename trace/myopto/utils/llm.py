# trace/myopto/utils/llm.py
"""
LLM backends for Trace. Supports:
- LiteLLM (cloud providers)
- CustomLLM (OpenAI-compatible endpoints)
- AutoGen (autogen wrapper)
- LocalSLM (local small models via transformers/MLX)
"""
from __future__ import annotations

from typing import List, Dict, Any, Callable, Union, Optional
import os
import time
import json
import warnings

import litellm
import openai

try:
    import autogen
except ImportError:
    autogen = None

# Optional: transformers for local SLM
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

# Optional: MLX for Apple Silicon
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# --------------------------------------------------
# Router Config Helpers
# --------------------------------------------------
def _get_router_model_for_role(role: Optional[str]) -> Optional[str]:
    """
    Check llm_router._ROLE_CFG for a model configured via set_role_models().
    Returns None if no config is found (caller should fall back to env vars).
    """
    if role is None:
        return None
    try:
        from myopto.utils import llm_router
        r = llm_router._norm_role(role)
        with llm_router._ROLE_CFG_LOCK:
            cfg = llm_router._ROLE_CFG.get(r)
            if cfg is not None and cfg.model is not None:
                return cfg.model
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _get_router_config_for_role(role: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get the full config dict from llm_router for a given role."""
    if role is None:
        return None
    try:
        from myopto.utils import llm_router
        r = llm_router._norm_role(role)
        with llm_router._ROLE_CFG_LOCK:
            cfg = llm_router._ROLE_CFG.get(r)
            if cfg is not None:
                return {
                    "model": cfg.model,
                    "base_url": cfg.base_url,
                    "api_key": cfg.api_key,
                    "reset_freq": cfg.reset_freq,
                    "cache": cfg.cache,
                    "backend": cfg.backend,
                }
    except ImportError:
        pass
    except Exception:
        pass
    return None


# --------------------------------------------------
# Abstract Model Base
# --------------------------------------------------
class AbstractModel:
    """
    Base class for LLM backends. Handles model refresh and usage tracking.
    
    Subclasses should override the `model` property and optionally `create()`.
    """

    def __init__(
        self,
        factory: Callable,
        reset_freq: Union[int, None] = None,
        role: Union[str, None] = None,
    ) -> None:
        """
        Args:
            factory: Function that returns a callable model.
            reset_freq: Seconds before refreshing model (None = never).
            role: Role label for accounting (executor/optimizer/metaoptimizer).
        """
        self.factory = factory
        self._model = self.factory()
        self.reset_freq = reset_freq
        self._init_time = time.time()
        self.role = role

    @property
    def model(self):
        """Override in subclasses. Returns callable for completions."""
        return self._model

    def __call__(self, *args, **kwargs) -> Any:
        """Main API. Handles refresh and usage tracking."""
        if self.reset_freq is not None and time.time() - self._init_time > self.reset_freq:
            self._model = self.factory()
            self._init_time = time.time()

        resp = self.model(*args, **kwargs)

        # Usage tracking (opt-in)
        try:
            from myopto.utils.usage import track_response
            track_response(
                resp,
                model=getattr(self, "model_name", None),
                role=getattr(self, "role", None),
                backend=self.__class__.__name__,
            )
        except Exception:
            pass

        return resp

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = self.factory()


# --------------------------------------------------
# LiteLLM Backend
# --------------------------------------------------
class LiteLLM(AbstractModel):
    """
    LLM backend using the LiteLLM library.
    Supports all providers LiteLLM supports (OpenAI, Anthropic, etc).
    
    Model resolution order:
        1. Explicit `model` argument
        2. llm_router config (set_role_models())
        3. Role-specific env: TRACE_LLM_MODEL_<ROLE>
        4. Global env: TRACE_LITELLM_MODEL
        5. Default: 'gpt-4o'
    """

    def __init__(
        self,
        model: Union[str, None] = None,
        reset_freq: Union[int, None] = None,
        cache: bool = True,
        role: Union[str, None] = None,
    ) -> None:
        # Priority 1: Explicit model
        # Priority 2: Router config
        if model is None:
            model = _get_router_model_for_role(role)

        # Priority 3: Role-specific env
        if model is None and role is not None:
            suffix = str(role).upper()
            model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or \
                    os.environ.get(f"TRACE_LITELLM_MODEL_{suffix}")

        # Priority 4: Global env
        if model is None:
            model = os.environ.get('TRACE_LITELLM_MODEL')

        # Priority 5: Deprecated env (with warning)
        if model is None:
            deprecated = os.environ.get('DEFAULT_LITELLM_MODEL')
            if deprecated:
                warnings.warn(
                    "DEFAULT_LITELLM_MODEL is deprecated. Use TRACE_LITELLM_MODEL instead."
                )
                model = deprecated

        # Priority 6: Default
        if model is None:
            model = 'gpt-4o'

        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(self.model_name)
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, model_name: str):
        if model_name.startswith('azure/'):
            scope = os.environ.get('AZURE_TOKEN_PROVIDER_SCOPE')
            if scope:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = get_bearer_token_provider(DefaultAzureCredential(), scope)
                return lambda *args, **kwargs: litellm.completion(
                    model_name, *args, azure_ad_token_provider=credential, **kwargs
                )
        return lambda *args, **kwargs: litellm.completion(model_name, *args, **kwargs)

    @property
    def model(self):
        return lambda *args, **kwargs: self._model(*args, **kwargs)


# --------------------------------------------------
# CustomLLM Backend (OpenAI-compatible endpoints)
# --------------------------------------------------
class CustomLLM(AbstractModel):
    """
    Backend for OpenAI-compatible API endpoints.
    Useful for: LiteLLM proxy servers, vLLM, local servers, etc.
    
    Supports response_format with JSON schema for structured outputs.
    """

    def __init__(
        self,
        model: Union[str, None] = None,
        reset_freq: Union[int, None] = None,
        cache: bool = True,
        role: Union[str, None] = None,
        base_url: Union[str, None] = None,
        api_key: Union[str, None] = None,
    ) -> None:
        router_cfg = _get_router_config_for_role(role) if (model is None or base_url is None) else None
        suffix = str(role).upper() if role else None

        # Model resolution
        if model is None:
            if router_cfg and router_cfg.get("model"):
                model = router_cfg["model"]
            elif suffix:
                model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or \
                        os.environ.get(f"TRACE_CUSTOMLLM_MODEL_{suffix}")
            if model is None:
                model = os.environ.get('TRACE_CUSTOMLLM_MODEL', 'gpt-4o')

        # Base URL resolution
        if base_url is None:
            if router_cfg and router_cfg.get("base_url"):
                base_url = router_cfg["base_url"]
            elif suffix:
                base_url = os.environ.get(f"TRACE_CUSTOMLLM_URL_{suffix}") or \
                           os.environ.get("TRACE_CUSTOMLLM_URL")
            if base_url is None:
                base_url = os.environ.get('TRACE_CUSTOMLLM_URL', 'http://localhost:4000')

        # API key resolution
        if api_key is None:
            if router_cfg and router_cfg.get("api_key"):
                api_key = router_cfg["api_key"]
            elif suffix:
                api_key = os.environ.get(f"TRACE_CUSTOMLLM_API_KEY_{suffix}") or \
                          os.environ.get("TRACE_CUSTOMLLM_API_KEY")
            if api_key is None:
                api_key = os.environ.get('TRACE_CUSTOMLLM_API_KEY', 'sk-placeholder')

        self.model_name = model
        self.cache = cache
        self.base_url = base_url
        self.api_key = api_key

        factory = lambda: self._factory(base_url, api_key)
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, base_url: str, api_key: str) -> openai.OpenAI:
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    @property
    def model(self):
        return lambda *args, **kwargs: self.create(*args, **kwargs)

    def create(self, **config: Any):
        """Create completion. Supports response_format for JSON schema."""
        if 'model' not in config:
            config['model'] = self.model_name
        return self._model.chat.completions.create(**config)


# --------------------------------------------------
# LocalSLM Backend (transformers / MLX)
# --------------------------------------------------
# Recommended models for M1 Mac (<1B params):
#   - Qwen/Qwen2-0.5B-Instruct (best balance)
#   - TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   - microsoft/phi-1_5

class LocalSLM(AbstractModel):
    """
    Backend for running small local models via transformers or MLX.
    
    Optimized for Apple Silicon (M1/M2) with MPS acceleration.
    Falls back to CPU if MPS unavailable.
    
    Recommended models:
        - Qwen/Qwen2-0.5B-Instruct (best for M1, ~20-40 tok/s)
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0
        - microsoft/phi-1_5
    
    Config resolution:
        1. Explicit `model` argument
        2. llm_router config
        3. Env: TRACE_LOCALSLM_MODEL_<ROLE> or TRACE_LOCALSLM_MODEL
        4. Default: 'Qwen/Qwen2-0.5B-Instruct'
    """

    # Class-level cache for loaded models
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model: Union[str, None] = None,
        reset_freq: Union[int, None] = None,
        role: Union[str, None] = None,
        use_mlx: bool = False,
        device: Union[str, None] = None,
        max_new_tokens: int = 150,
        torch_dtype: Optional[str] = "float16",
    ) -> None:
        if not HAS_TRANSFORMERS and not HAS_MLX:
            raise ImportError(
                "LocalSLM requires either 'transformers' or 'mlx-lm'.\n"
                "Install with: pip install torch transformers\n"
                "Or for MLX: pip install mlx mlx-lm"
            )

        router_cfg = _get_router_config_for_role(role)
        suffix = str(role).upper() if role else None

        # Model resolution
        if model is None:
            if router_cfg and router_cfg.get("model"):
                model = router_cfg["model"]
            elif suffix:
                model = os.environ.get(f"TRACE_LOCALSLM_MODEL_{suffix}")
            if model is None:
                model = os.environ.get('TRACE_LOCALSLM_MODEL', 'Qwen/Qwen2-0.5B-Instruct')

        self.model_name = model
        self.use_mlx = use_mlx and HAS_MLX
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype

        # Device selection for transformers
        if device is None:
            if self.use_mlx:
                device = "mlx"
            elif HAS_TRANSFORMERS and torch.backends.mps.is_available():
                device = "mps"
            elif HAS_TRANSFORMERS and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        factory = lambda: self._load_model()
        super().__init__(factory, reset_freq, role=role)

    def _load_model(self):
        """Load model with caching for efficiency."""
        cache_key = f"{self.model_name}:{self.device}:{self.use_mlx}"
        
        if cache_key in LocalSLM._model_cache:
            return LocalSLM._model_cache[cache_key]

        if self.use_mlx and HAS_MLX:
            model, tokenizer = mlx_load(self.model_name)
            LocalSLM._model_cache[cache_key] = (model, tokenizer, "mlx")
            return (model, tokenizer, "mlx")

        # Transformers path
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device if self.device != "mps" else None,
        )
        
        if self.device == "mps":
            model = model.to("mps")

        LocalSLM._model_cache[cache_key] = (model, tokenizer, "transformers")
        LocalSLM._tokenizer_cache[cache_key] = tokenizer
        return (model, tokenizer, "transformers")

    @property
    def model(self):
        return lambda *args, **kwargs: self.create(*args, **kwargs)

    def create(self, **config: Any) -> Dict[str, Any]:
        """
        Generate completion in OpenAI-compatible format.
        
        Accepts:
            messages: List[Dict] - Chat messages
            max_tokens / max_new_tokens: int
            temperature: float
            response_format: Optional[Dict] - For JSON mode (best-effort)
        """
        model_data, tokenizer, backend = self._model
        messages = config.get("messages", [])
        max_tokens = config.get("max_tokens") or config.get("max_new_tokens") or self.max_new_tokens
        temperature = config.get("temperature", 0.7)
        do_sample = temperature > 0
        response_format = config.get("response_format")

        # Build prompt
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                else:
                    parts.append(f"Assistant: {content}")
            parts.append("Assistant:")
            text = "\n".join(parts)

        # Add JSON instruction if response_format requests it
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            text = text.rstrip()
            schema = response_format.get("json_schema", {}).get("schema", {})
            if schema:
                text += f"\n\nRespond ONLY with valid JSON matching this schema: {json.dumps(schema)}"
            else:
                text += "\n\nRespond ONLY with valid JSON."

        # Generate
        if backend == "mlx":
            response_text = mlx_generate(
                model_data, tokenizer, prompt=text, max_tokens=max_tokens
            )
        else:
            # Transformers
            inputs = tokenizer(text, return_tensors="pt")
            if self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            elif self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model_data.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response
            if text in response_text:
                response_text = response_text[len(text):].strip()

        # Return OpenAI-compatible format
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": self.model_name,
            "usage": {
                "prompt_tokens": len(text.split()),  # Approximate
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(text.split()) + len(response_text.split()),
            },
        }

    @classmethod
    def clear_cache(cls):
        """Clear loaded models from memory."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        if HAS_TRANSFORMERS:
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------
# AutoGen Backend
# --------------------------------------------------
class AutoGenLLM(AbstractModel):
    """Wrapper around autogen's OpenAIWrapper."""

    def __init__(
        self,
        config_list: List = None,
        filter_dict: Dict = None,
        reset_freq: Union[int, None] = None,
        model: Union[str, None] = None,
        role: Union[str, None] = None,
    ) -> None:
        if autogen is None:
            raise ImportError("autogen is not installed. pip install pyautogen")

        # Priority 1: Router config
        if model is None:
            model = _get_router_model_for_role(role)

        # Priority 2: Role-specific env
        if model is None and role is not None:
            suffix = str(role).upper()
            model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or \
                    os.environ.get(f"TRACE_AUTOGEN_MODEL_{suffix}")

        # Priority 3: Global env
        if model is None:
            model = os.environ.get("TRACE_AUTOGEN_MODEL")

        # Apply model filter
        if model is not None:
            if filter_dict is None:
                filter_dict = {"model": model}
            elif isinstance(filter_dict, dict) and "model" not in filter_dict:
                filter_dict = dict(filter_dict)
                filter_dict["model"] = model

        if config_list is None:
            try:
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
            except Exception:
                config_list = auto_construct_oai_config_list_from_env()
                if config_list:
                    os.environ["OAI_CONFIG_LIST"] = json.dumps(config_list)
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

        if filter_dict is not None:
            config_list = autogen.filter_config(config_list, filter_dict)

        self.model_name = model
        factory = lambda: self._factory(config_list)
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, config_list):
        return autogen.OpenAIWrapper(config_list=config_list)

    @property
    def model(self):
        return lambda *args, **kwargs: self.create(*args, **kwargs)

    def create(self, **config: Any):
        return self._model.create(**config)


def auto_construct_oai_config_list_from_env() -> List:
    """Build config list from environment API keys."""
    config_list = []
    if os.environ.get("OPENAI_API_KEY"):
        config_list.append({
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"]
        })
    if os.environ.get("ANTHROPIC_API_KEY"):
        config_list.append({
            "model": "claude-3-5-sonnet-latest",
            "api_key": os.environ["ANTHROPIC_API_KEY"]
        })
    return config_list


# --------------------------------------------------
# Default Backend Selection
# --------------------------------------------------
TRACE_DEFAULT_LLM_BACKEND = os.getenv('TRACE_DEFAULT_LLM_BACKEND', 'LiteLLM')

if TRACE_DEFAULT_LLM_BACKEND == 'AutoGen':
    print("Using AutoGen as the default LLM backend.")
    LLM = AutoGenLLM
elif TRACE_DEFAULT_LLM_BACKEND == 'CustomLLM':
    print("Using CustomLLM as the default LLM backend.")
    LLM = CustomLLM
elif TRACE_DEFAULT_LLM_BACKEND == 'LocalSLM':
    print("Using LocalSLM as the default LLM backend.")
    LLM = LocalSLM
elif TRACE_DEFAULT_LLM_BACKEND == 'LiteLLM':
    print("Using LiteLLM as the default LLM backend.")
    LLM = LiteLLM
else:
    raise ValueError(f"Unknown LLM backend: {TRACE_DEFAULT_LLM_BACKEND}")