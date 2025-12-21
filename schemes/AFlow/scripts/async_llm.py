# schemes/AFlow/scripts/async_llm.py
"""
Async LLM interface for AFlow - bridges to llm_router.
"""

from typing import Dict, Any, Union

from pydantic import BaseModel, Field

from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost
from utils.logs import logger


# ----------------------------------------------------------------------------------------------------
# Configuration Models
# ----------------------------------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """Configuration for LLM instances."""
    model: str = Field(default="gpt-4o-mini")
    role: str = Field(default="executor")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)


class LLMsConfig:
    """Container for multiple LLM configurations."""
    
    def __init__(self, configs: Dict[str, LLMConfig] = None):
        self.configs = configs or {}
    
    def get(self, name: str) -> LLMConfig:
        return self.configs.get(name, LLMConfig())
    
    @classmethod
    def default(cls) -> "LLMsConfig":
        return cls({
            "gpt-4o": LLMConfig(model="gpt-4o", role="optimizer"),
            "gpt-4o-mini": LLMConfig(model="gpt-4o-mini", role="executor"),
            "claude-3-5-sonnet": LLMConfig(model="claude-3-5-sonnet-20241022", role="optimizer"),
        })


# ----------------------------------------------------------------------------------------------------
# Token Usage Tracking
# ----------------------------------------------------------------------------------------------------

class TokenUsageTracker:
    """Tracks token usage across LLM calls."""
    
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
        if isinstance(config, str):
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
        
        response = self.llm(messages=messages)
        
        self.usage_tracker.track(response)
        
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
        # Prepare prompt with format instructions
        if formatter is not None and hasattr(formatter, "prepare_prompt"):
            formatted_prompt = formatter.prepare_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                raw_response = await self(formatted_prompt)
                
                if formatter is not None and hasattr(formatter, "parse"):
                    result = formatter.parse(raw_response)
                    # Ensure all expected fields exist (with defaults if needed)
                    if hasattr(formatter, "_get_field_names"):
                        for field in formatter._get_field_names():
                            if field not in result:
                                result[field] = ""
                    return result
                
                return {"response": raw_response}
                
            except Exception as e:
                last_error = e
                logger.debug(f"[AsyncLLM] Format attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                continue
        
        # Final fallback
        raw_response = await self(formatted_prompt)
        return {"response": raw_response}
    
    def _extract_text(self, response: Any) -> str:
        """Extract text content from response."""
        if isinstance(response, str):
            return response
        
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return choice.message.content or ""
            if hasattr(choice, "text"):
                return choice.text or ""
        
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
        llm_config: Configuration for the LLM.
        role: Default role if not specified in config.
        
    Returns:
        Configured AsyncLLM instance.
    """
    if llm_config is None:
        return AsyncLLM(role=role)
    
    if isinstance(llm_config, AsyncLLM):
        return llm_config
    
    return AsyncLLM(config=llm_config, role=role)