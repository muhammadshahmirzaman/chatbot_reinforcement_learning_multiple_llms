"""
Transformers adapter for local HuggingFace model inference.

This adapter loads and runs HuggingFace Transformers models locally
for inference without requiring an external server.
"""

import time
import logging
from typing import Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..base import OfflineAdapter, GenerationConfig, ModelResponse, MemoryInfo, ExecutionMode

logger = logging.getLogger(__name__)


class TransformersAdapter(OfflineAdapter):
    """
    Adapter for local HuggingFace Transformers models.
    
    This adapter loads models directly into GPU/CPU memory for inference,
    supporting various model architectures available on HuggingFace Hub.
    """
    
    def __init__(
        self,
        name: str,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        estimated_memory_mb: float = 4000,
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ):
        """
        Initialize the Transformers adapter.
        
        Args:
            name: Unique name for this adapter.
            model_name_or_path: HuggingFace model ID or local path.
            device: Target device ("cuda", "cuda:0", "cpu").
            torch_dtype: Data type ("float16", "bfloat16", "float32").
            estimated_memory_mb: Estimated GPU memory usage in MB.
            trust_remote_code: Whether to trust remote code for models.
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes).
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes).
        """
        self._name = name
        self.model_path = model_name_or_path
        self.device = device
        self.torch_dtype = self._get_torch_dtype(torch_dtype)
        self._estimated_memory_mb = estimated_memory_mb
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        self.model = None
        self.tokenizer = None
    
    @staticmethod
    def _get_torch_dtype(dtype_str: str):
        """Convert string dtype to torch.dtype."""
        if not TORCH_AVAILABLE or torch is None:
            return dtype_str  # Return string if torch not available
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    @property
    def name(self) -> str:
        return self._name
    
    def load(self) -> None:
        """Load the model to the configured device."""
        self.load_to_device(self.device)
    
    def load_to_device(self, device: str) -> None:
        """
        Load model to the specified device.
        
        Args:
            device: Target device (e.g., "cuda:0", "cpu").
        """
        logger.info(f"Loading {self._name} from {self.model_path} to {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        # Prepare quantization config
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            if not TORCH_AVAILABLE:
                logger.warning("Torch not available, skipping quantization config")
            else:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=self.load_in_4bit,
                        load_in_8bit=self.load_in_8bit,
                        bnb_4bit_compute_dtype=torch.float16 if self.load_in_4bit else None
                    )
                except ImportError:
                    logger.warning("Could not import BitsAndBytesConfig")

        # Handle device_map for GPU vs CPU
        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code
            ).cpu()
        else:
            # Arguments for from_pretrained
            kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": device,
                "trust_remote_code": self.trust_remote_code,
            }
            
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **kwargs
            )
        
        self.model.eval()
        self.device = device
        logger.info(f"Successfully loaded {self._name}")
    
    def offload_to_cpu(self) -> None:
        """Offload model from GPU to CPU."""
        if self.model is not None:
            logger.info(f"Offloading {self._name} to CPU")
            self.model = self.model.cpu()
            self.device = "cpu"
            torch.cuda.empty_cache()
    
    def unload(self) -> None:
        """Completely unload the model to free memory."""
        logger.info(f"Unloading {self._name}")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
    
    def get_memory_usage(self) -> MemoryInfo:
        """Return estimated memory usage."""
        gpu_mem = self._estimated_memory_mb if "cuda" in str(self.device) else 0
        cpu_mem = self._estimated_memory_mb if self.device == "cpu" else 0
        return MemoryInfo(
            gpu_memory_mb=gpu_mem,
            cpu_memory_mb=cpu_mem,
            is_loaded=self.is_loaded()
        )
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> ModelResponse:
        """
        Generate a response using local inference.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration.
            
        Returns:
            ModelResponse with generated text and metadata.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Model {self._name} is not loaded")
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ModelResponse(
            text=text,
            model_name=self._name,
            latency_ms=latency_ms,
            tokens_used=len(outputs[0]),
            execution_mode=ExecutionMode.OFFLINE
        )
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "unloaded"
        return f"TransformersAdapter(name={self._name}, model={self.model_path}, status={status})"
