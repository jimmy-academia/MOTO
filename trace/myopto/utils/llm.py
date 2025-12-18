# trace/myopto/utils/llm.py
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import os
import time
import json
import litellm
import openai
import warnings

try:
    import autogen  # We import autogen here to avoid the need of installing autogen
except ImportError:
    autogen = None


# ---------------------------------------------------------------------------
# Helper to resolve model from llm_router config (lazy import to avoid circular)
# ---------------------------------------------------------------------------
def _get_router_model_for_role(role: Optional[str]) -> Optional[str]:
    """
    Check llm_router._ROLE_CFG for a model configured via set_role_models().
    Returns None if no config is found (caller should fall back to env vars).
    Uses lazy import to avoid circular dependency.
    """
    if role is None:
        return None
    try:
        # Lazy import to avoid circular dependency (llm_router imports from this file)
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
    """
    Get the full config dict from llm_router for a given role.
    Returns None if no config found.
    """
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
                }
    except ImportError:
        pass
    except Exception:
        pass
    return None


class AbstractModel:
    """
    A minimal abstraction of a model api that refreshes the model every
    reset_freq seconds (this is useful for long-running models that may require
    refreshing certificates or memory management).
    """

    def __init__(
        self,
        factory: Callable,
        reset_freq: Union[int, None] = None,
        role: Union[str, None] = None,
    ) -> None:
        """
        Args:
            factory: A function that takes no arguments and returns a model that is callable.
            reset_freq: The number of seconds after which the model should be
                refreshed. If None, the model is never refreshed.
            role: Optional role label for accounting/routing (e.g., executor/optimizer/metaoptimizer).
        """
        self.factory = factory
        self._model = self.factory()
        self.reset_freq = reset_freq
        self._init_time = time.time()
        self.role = role

    # Overwrite this `model` property when subclassing.
    @property
    def model(self):
        """ When self.model is called, text responses should always be available at ['choices'][0].['message']['content'] """
        return self._model

    # This is the main API
    def __call__(self, *args, **kwargs) -> Any:
        """ The call function handles refreshing the model if needed. """
        if self.reset_freq is not None and time.time() - self._init_time > self.reset_freq:
            self._model = self.factory()
            self._init_time = time.time()

        resp = self.model(*args, **kwargs)

        # Best-effort usage/cost tracking (opt-in via myopto.utils.usage.configure_usage()).
        try:
            from myopto.utils.usage import track_response

            track_response(
                resp,
                model=getattr(self, "model_name", None),
                role=getattr(self, "role", None),
                backend=self.__class__.__name__,
            )
        except Exception:
            # Never break the main call path due to accounting.
            pass

        return resp

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = self.factory()


class AutoGenLLM(AbstractModel):
    """This is the main class Trace uses to interact with the model. It is a wrapper around autogen's OpenAIWrapper. For using models not supported by autogen, subclass AutoGenLLM and override the `_factory` and  `create` method. Users can pass instances of this class to optimizers' llm argument."""

    def __init__(
        self,
        config_list: List = None,
        filter_dict: Dict = None,
        reset_freq: Union[int, None] = None,
        model: Union[str, None] = None,
        role: Union[str, None] = None,
    ) -> None:
        if autogen is None:
            raise ImportError("autogen is not installed but AutoGenLLM was requested.")

        # PRIORITY 1: Check llm_router config (set via set_role_models())
        if model is None:
            model = _get_router_model_for_role(role)

        # PRIORITY 2: role/env-driven model selection
        if model is None and role is not None:
            suffix = str(role).upper()
            model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or os.environ.get(f"TRACE_AUTOGEN_MODEL_{suffix}")
        
        # PRIORITY 3: global env var
        if model is None:
            model = os.environ.get("TRACE_AUTOGEN_MODEL")

        # If a model is provided, it can be applied via filter_dict.
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
                if len(config_list) > 0:
                    os.environ.update({"OAI_CONFIG_LIST": json.dumps(config_list)})
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

        if filter_dict is not None:
            config_list = autogen.filter_config(config_list, filter_dict)

        # for tracking purposes (best-effort)
        self.model_name = model

        factory = lambda *args, **kwargs: self._factory(config_list)
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, config_list):
        return autogen.OpenAIWrapper(config_list=config_list)

    @property
    def model(self):
        return lambda *args, **kwargs: self.create(*args, **kwargs)

    # This is main API. We use the API of autogen's OpenAIWrapper
    def create(self, **config: Any):
        """Make a completion for a given config using available clients.
        Besides the kwargs allowed in openai's [or other] client, we allow the following additional kwargs.
        The config in each client will be overridden by the config.

        Args:
            - context (Dict | None): The context to instantiate the prompt or messages. Default to None.
            - cache (AbstractCache | None): A Cache object to use for response cache. Default to None.
            - agent (AbstractAgent | None): The object responsible for creating a completion if an agent.
            - (Legacy) cache_seed (int | None) for using the DiskCache. Default to 41.
            - filter_func (Callable | None): A function that takes in the context and the response
            - allow_format_str_template (bool | None): Whether to allow format string template in the config. Default to false.
            - api_version (str | None): The api version. Default to None. E.g., "2024-02-01".
        """
        return self._model.create(**config)


def auto_construct_oai_config_list_from_env() -> List:
    """
    Collect various API keys saved in the environment and return a format like:
    [{"model": "gpt-4", "api_key": xxx}, {"model": "claude-3.5-sonnet", "api_key": xxx}]

    Note this is a lazy function that defaults to gpt-4o and claude-3.5-sonnet.
    If you want to specify your own model, please provide an OAI_CONFIG_LIST in the environment or as a file
    """
    config_list = []
    if os.environ.get("OPENAI_API_KEY") is not None:
        config_list.append(
            {"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}
        )
    if os.environ.get("ANTHROPIC_API_KEY") is not None:
        config_list.append(
            {
                "model": "claude-3-5-sonnet-latest",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
        )
    return config_list


class LiteLLM(AbstractModel):
    """
    This is an LLM backend supported by LiteLLM library.

    Model resolution order:
        1. Explicit `model` argument passed to __init__
        2. llm_router config (set via set_role_models())
        3. Role-specific env var: TRACE_LLM_MODEL_<ROLE> or TRACE_LITELLM_MODEL_<ROLE>
        4. Global env var: TRACE_LITELLM_MODEL
        5. Deprecated env var: DEFAULT_LITELLM_MODEL
        6. Default: 'gpt-4o'
    """

    def __init__(
        self,
        model: Union[str, None] = None,
        reset_freq: Union[int, None] = None,
        cache=True,
        role: Union[str, None] = None,
    ) -> None:
        # PRIORITY 1: Explicit model argument (already set, skip if provided)
        
        # PRIORITY 2: Check llm_router config (set via set_role_models())
        if model is None:
            model = _get_router_model_for_role(role)

        # PRIORITY 3: Role-specific environment variables
        if model is None and role is not None:
            suffix = str(role).upper()
            model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or os.environ.get(f"TRACE_LITELLM_MODEL_{suffix}")

        # PRIORITY 4: Global TRACE_LITELLM_MODEL env var
        if model is None:
            model = os.environ.get('TRACE_LITELLM_MODEL')
        
        # PRIORITY 5: Deprecated DEFAULT_LITELLM_MODEL (with warning)
        if model is None:
            deprecated_model = os.environ.get('DEFAULT_LITELLM_MODEL')
            if deprecated_model is not None:
                warnings.warn(
                    "DEFAULT_LITELLM_MODEL environment variable is deprecated. "
                    "Please use set_role_models() or TRACE_LITELLM_MODEL instead."
                )
                model = deprecated_model
        
        # PRIORITY 6: Default fallback
        if model is None:
            model = 'gpt-4o'

        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(self.model_name)  # an LLM instance uses a fixed model
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, model_name: str):
        if model_name.startswith('azure/'):  # azure model
            azure_token_provider_scope = os.environ.get('AZURE_TOKEN_PROVIDER_SCOPE', None)
            if azure_token_provider_scope is not None:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = get_bearer_token_provider(DefaultAzureCredential(), azure_token_provider_scope)
                return lambda *args, **kwargs: litellm.completion(
                    model_name, *args, azure_ad_token_provider=credential, **kwargs
                )
        return lambda *args, **kwargs: litellm.completion(model_name, *args, **kwargs)

    @property
    def model(self):
        return lambda *args, **kwargs: self._model(*args, **kwargs)


class CustomLLM(AbstractModel):
    """
    This is for Custom server's API endpoints that are OpenAI Compatible.
    Such server includes LiteLLM proxy server.
    
    Config resolution order:
        1. Explicit arguments passed to __init__
        2. llm_router config (set via set_role_models() / set_role_config())
        3. Role-specific env vars
        4. Global env vars
        5. Defaults
    """

    def __init__(
        self,
        model: Union[str, None] = None,
        reset_freq: Union[int, None] = None,
        cache=True,
        role: Union[str, None] = None,
        base_url: Union[str, None] = None,
        api_key: Union[str, None] = None,
    ) -> None:
        # PRIORITY 1: Check llm_router config for all settings
        router_cfg = _get_router_config_for_role(role) if (model is None or base_url is None or api_key is None) else None
        
        # Role suffix for env var lookups
        suffix = str(role).upper() if role is not None else None

        # Model resolution
        if model is None:
            if router_cfg and router_cfg.get("model"):
                model = router_cfg["model"]
            elif suffix:
                model = os.environ.get(f"TRACE_LLM_MODEL_{suffix}") or os.environ.get(f"TRACE_CUSTOMLLM_MODEL_{suffix}")
            if model is None:
                model = os.environ.get('TRACE_CUSTOMLLM_MODEL', 'gpt-4o')

        # Base URL resolution
        if base_url is None:
            if router_cfg and router_cfg.get("base_url"):
                base_url = router_cfg["base_url"]
            elif suffix:
                base_url = os.environ.get(f"TRACE_CUSTOMLLM_URL_{suffix}") or os.environ.get("TRACE_CUSTOMLLM_URL")
            if base_url is None:
                base_url = os.environ.get('TRACE_CUSTOMLLM_URL', 'http://localhost:4000')

        # API key resolution
        if api_key is None:
            if router_cfg and router_cfg.get("api_key"):
                api_key = router_cfg["api_key"]
            elif suffix:
                api_key = os.environ.get(f"TRACE_CUSTOMLLM_API_KEY_{suffix}") or os.environ.get("TRACE_CUSTOMLLM_API_KEY")
            if api_key is None:
                api_key = os.environ.get('TRACE_CUSTOMLLM_API_KEY', 'sk-placeholder')

        self.model_name = model
        self.cache = cache
        self.base_url = base_url
        self.api_key = api_key

        factory = lambda: self._factory(base_url, api_key)  # an LLM instance uses a fixed endpoint
        super().__init__(factory, reset_freq, role=role)

    @classmethod
    def _factory(cls, base_url: str, server_api_key: str) -> openai.OpenAI:
        return openai.OpenAI(base_url=base_url, api_key=server_api_key)

    @property
    def model(self):
        return lambda *args, **kwargs: self.create(*args, **kwargs)

    def create(self, **config: Any):
        if 'model' not in config:
            config['model'] = self.model_name
        return self._model.chat.completions.create(**config)

# --------------------------------------------------
# Default LLM backend selection
# --------------------------------------------------
TRACE_DEFAULT_LLM_BACKEND = os.getenv('TRACE_DEFAULT_LLM_BACKEND', 'LiteLLM')
if TRACE_DEFAULT_LLM_BACKEND == 'AutoGen':
    print("Using AutoGen as the default LLM backend.")
    LLM = AutoGenLLM
elif TRACE_DEFAULT_LLM_BACKEND == 'CustomLLM':
    print("Using CustomLLM as the default LLM backend.")
    LLM = CustomLLM
elif TRACE_DEFAULT_LLM_BACKEND == 'LiteLLM':
    print("Using LiteLLM as the default LLM backend.")
    LLM = LiteLLM
else:
    raise ValueError(f"Unknown LLM backend: {TRACE_DEFAULT_LLM_BACKEND}")