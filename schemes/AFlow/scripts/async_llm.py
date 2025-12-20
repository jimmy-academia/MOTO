# schemes/AFlow/scripts/async_llm.py
"""
Async LLM Interface for AFlow - Adapted for llm_router integration.

This module provides AFlow-compatible LLM interfaces that use the centralized
llm_router instead of the original config file-based approach.

Key changes from original:
- LLMsConfig.default() replaced with llm_router integration
- create_llm_instance() now wraps get_llm()
- AsyncLLM adapted to work with LiteLLM backend
"""

from typing import Dict, Optional, Any, Union
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost

try:
    from schemes.AFlow.scripts.formatter import BaseFormatter, FormatError
except ImportError:
    # Fallback if formatter not available
    BaseFormatter = None
    FormatError = Exception


# ----------------------------------------------------------------------------------------------------
# LLM Configuration Classes (Compatibility Layer)
# ----------------------------------------------------------------------------------------------------

class LLMConfig:
    """
    Compatibility class for AFlow's LLMConfig.
    
    This maintains the same interface as the original LLMConfig but
    delegates to llm_router for actual model configuration.
    """
    
    def __init__(self, config: dict = None, role: str = "executor"):
        self.role = role
        config = config or {}
        self.model = config.get("model", None)
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 1.0)
        # These are kept for compatibility but not used
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", None)


class LLMsConfig:
    """
    Compatibility class for AFlow's LLMsConfig.
    
    Instead of loading from YAML files, this integrates with llm_router
    to use the centralized model configuration.
    """
    
    _instance = None
    
    def __init__(self):
        pass
    
    @classmethod
    def default(cls) -> 'LLMsConfig':
        """Get the default configuration instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, model_name: str) -> Optional[LLMConfig]:
        """
        Get LLM config for a model name.
        
        In the adapted version, this returns a config that will
        use llm_router. The model_name can be:
        - A role name like "optimizer" or "executor"
        - An actual model name (passed through)
        """
        # Map common role patterns
        role = "executor"
        if "optim" in model_name.lower() or "claude" in model_name.lower():
            role = "optimizer"
        
        return LLMConfig({"model": model_name}, role=role)


# ----------------------------------------------------------------------------------------------------
# Token Usage Tracker
# ----------------------------------------------------------------------------------------------------

class TokenUsageTracker:
    """Tracks token usage and costs for LLM calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.usage_history = []
    
    def track(self, response: Any) -> Dict[str, Any]:
        """Track usage from a response."""
        usage_record = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
        
        # Extract usage from response
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                usage_record["input_tokens"] = usage.prompt_tokens
                self.total_input_tokens += usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                usage_record["output_tokens"] = usage.completion_tokens
                self.total_output_tokens += usage.completion_tokens
        
        self.usage_history.append(usage_record)
        return usage_record
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": get_total_cost(),
            "call_count": len(self.usage_history),
            "history": self.usage_history,
        }


# ----------------------------------------------------------------------------------------------------
# AsyncLLM - Main Interface
# ----------------------------------------------------------------------------------------------------

class AsyncLLM:
    """
    Async LLM interface compatible with AFlow's original interface.
    
    This class wraps llm_router's LLM interface to provide the same
    API that AFlow's code expects, including async calls and formatting.
    """
    
    def __init__(
        self,
        config: Union[LLMConfig, str, Dict, Any] = None,
        role: str = "executor",
        system_msg: str = None
    ):
        """
        Initialize AsyncLLM.
        
        Args:
            config: LLMConfig, model name string, dict config, or LLMAdapter
            role: Role for llm_router (executor, optimizer)
            system_msg: Optional system message for all calls
        """
        # Handle different config types
        if isinstance(config, str):
            # Model name string
            if "optim" in config.lower() or "claude" in config.lower():
                role = "optimizer"
            self.model_name = config
        elif isinstance(config, LLMConfig):
            self.model_name = config.model
            role = config.role
        elif isinstance(config, dict):
            self.model_name = config.get("model")
            role = config.get("role", role)
        elif hasattr(config, "role"):
            # LLMAdapter or similar
            role = config.role
            self.model_name = None
        else:
            self.model_name = None
        
        self.role = role
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of the underlying LLM."""
        if self._llm is None:
            self._llm = get_llm(role=self.role)
        return self._llm
    
    async def __call__(self, prompt: str) -> str:
        """
        Async call to generate text.
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated text response
        """
        messages = []
        
        if self.sys_msg:
            messages.append({"role": "system", "content": self.sys_msg})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call the LLM
        response = self.llm(messages=messages)
        
        # Track usage
        self.usage_tracker.track(response)
        
        # Extract text
        return self._extract_text(response)
    
    async def call_with_format(
        self,
        prompt: str,
        formatter: Any,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Call LLM with structured output formatting.
        
        Args:
            prompt: The prompt to send
            formatter: Formatter instance for parsing response
            max_retries: Maximum retry attempts on parse failure
            
        Returns:
            Parsed response as dictionary
        """
        for attempt in range(max_retries):
            try:
                raw_response = await self(prompt)
                
                if formatter is not None and hasattr(formatter, "parse"):
                    return formatter.parse(raw_response)
                
                # Fallback: return as-is
                return {"response": raw_response}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                continue
        
        return {"response": await self(prompt)}
    
    def _extract_text(self, response: Any) -> str:
        """Extract text content from response."""
        if isinstance(response, str):
            return response
        
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return choice.message.content
            if hasattr(choice, "text"):
                return choice.text
        
        if isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            if "content" in response:
                return response["content"]
            if "text" in response:
                return response["text"]
        
        return str(response)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_tracker.get_summary()


# ----------------------------------------------------------------------------------------------------
# Factory Function
# ----------------------------------------------------------------------------------------------------

def create_llm_instance(
    llm_config: Union[LLMConfig, str, Dict, Any] = None,
    role: str = "executor"
) -> AsyncLLM:
    """
    Create an AsyncLLM instance using the provided configuration.
    
    This is the main factory function used by AFlow's code to create
    LLM instances. It now routes through llm_router.
    
    Args:
        llm_config: Configuration for the LLM. Can be:
            - LLMConfig instance
            - String model name
            - Dictionary with config values
            - LLMAdapter from the scheme
        role: Role for llm_router (executor, optimizer)
        
    Returns:
        AsyncLLM instance ready for use
    """
    # Handle LLMAdapter passthrough
    if hasattr(llm_config, "role") and hasattr(llm_config, "llm"):
        # This is our LLMAdapter - wrap it
        return AsyncLLM(config=llm_config, role=llm_config.role)
    
    # Handle string (model name)
    if isinstance(llm_config, str):
        return AsyncLLM(config=llm_config, role=role)
    
    # Handle LLMConfig
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(config=llm_config, role=llm_config.role)
    
    # Handle dict
    if isinstance(llm_config, dict):
        return AsyncLLM(config=llm_config, role=role)
    
    # Default
    return AsyncLLM(role=role)


# ----------------------------------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------------------------------

__all__ = [
    "LLMConfig",
    "LLMsConfig",
    "AsyncLLM",
    "TokenUsageTracker",
    "create_llm_instance",
]