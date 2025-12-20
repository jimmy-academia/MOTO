# trace/myopto/utils/localslm.py
"""
LocalSLM: Local Small Language Model

Wrapper for running small language models locally using transformers or MLX.
"""

from typing import Optional
import os


class LocalSLM:
    """
    Local Small Language Model wrapper.
    
    Supports loading and running small models like Qwen2-0.5B, Phi-3-mini,
    or similar models locally using HuggingFace transformers or MLX.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        device: str = "mps",
        use_mlx: bool = True,  # Default True for Apple Silicon
        torch_dtype: str = "float16",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize LocalSLM.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (mps, cuda, cpu)
            use_mlx: Use MLX backend (Apple Silicon only)
            torch_dtype: Torch dtype (float16, bfloat16, float32)
            cache_dir: Cache directory for model weights
        """
        self.model_name = model_name
        self.device = device
        self.use_mlx = use_mlx
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def _load_model(self):
        """Load model and tokenizer."""
        if self._loaded:
            return
        
        if self.use_mlx:
            self._load_mlx()
        else:
            self._load_transformers()
        
        self._loaded = True

    # --------------------------------------------------
    # Transformers Backend
    # --------------------------------------------------

    def _load_transformers(self):
        """Load model using transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch"
            )
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.float16)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=dtype,
            device_map=self.device if self.device != "mps" else None,
            trust_remote_code=True,
        )
        
        # Move to device if MPS
        if self.device == "mps":
            self.model = self.model.to("mps")
        
        self.model.eval()

    # --------------------------------------------------
    # MLX Backend (Apple Silicon)
    # --------------------------------------------------

    def _load_mlx(self):
        """Load model using MLX."""
        try:
            from mlx_lm import load, generate
            self._mlx_generate = generate
        except ImportError:
            raise ImportError(
                "mlx-lm required for MLX backend. Install with: "
                "pip install mlx-lm"
            )
        
        self.model, self.tokenizer = load(self.model_name)

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            
        Returns:
            Generated text
        """
        self._load_model()
        
        if self.use_mlx:
            return self._generate_mlx(prompt, max_new_tokens, temperature)
        else:
            return self._generate_transformers(
                prompt, max_new_tokens, temperature, top_p, do_sample
            )

    def _generate_transformers(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Generate using transformers."""
        import torch
        
        # Format as chat if model supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode (only new tokens)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()

    def _generate_mlx(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using MLX."""
        # Format as chat
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        
        response = self._mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_new_tokens,
            temp=temperature,
        )
        
        return response.strip()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> str:
        """Make instance callable."""
        return self.generate(prompt, **kwargs)

    def unload(self):
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_info(self) -> dict:
        """Get model info."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_mlx": self.use_mlx,
            "torch_dtype": self.torch_dtype,
            "loaded": self._loaded,
        }