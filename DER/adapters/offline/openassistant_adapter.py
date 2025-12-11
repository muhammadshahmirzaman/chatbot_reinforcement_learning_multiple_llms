"""
OpenAssistant adapter for local model inference.

This adapter integrates with the OpenAssistant model training framework,
supporting SFT, RL, and reward model variants.
"""

import sys
import time
import logging
from typing import Optional
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..base import OfflineAdapter, GenerationConfig, ModelResponse, MemoryInfo, ExecutionMode

logger = logging.getLogger(__name__)


class OpenAssistantAdapter(OfflineAdapter):
    """
    Adapter for OpenAssistant models with SFT/RL/Reward model support.
    
    This adapter integrates with the OpenAssistant model training framework
    located in models/LAION-AI-Open-Assistant-e1769c1/model.
    """
    
    def __init__(
        self,
        name: str,
        model_path: str,
        device: str = "cuda",
        model_type: str = "sft",
        use_reward_head: bool = False,
        estimated_memory_mb: float = 8000,
        **kwargs
    ):
        """
        Initialize the OpenAssistant adapter.
        
        Args:
            name: Unique name for this adapter.
            model_path: Path to the OpenAssistant model directory.
            device: Target device ("cuda", "cuda:0", "cpu").
            model_type: Type of model ("sft", "rl", "reward", "seq2seq").
            use_reward_head: Whether to load without the LM head (for reward models).
            estimated_memory_mb: Estimated GPU memory usage in MB.
        """
        self._name = name
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.use_reward_head = use_reward_head
        self._estimated_memory_mb = estimated_memory_mb
        
        self.model = None
        self.tokenizer = None
        self._oa_module_added = False
    
    @property
    def name(self) -> str:
        return self._name
    
    def _ensure_oa_import(self):
        """Ensure the OpenAssistant model training module is importable."""
        if self._oa_module_added:
            return
        
        # Add OpenAssistant model path to sys.path
        oa_model_path = Path(self.model_path).resolve()
        if oa_model_path.exists():
            # Try to find the model_training directory
            model_training = oa_model_path / "model_training"
            if model_training.exists():
                parent = oa_model_path
            else:
                # Maybe model_path points to model_training directly
                parent = oa_model_path.parent
            
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
                logger.debug(f"Added {parent} to sys.path")
        
        self._oa_module_added = True
    
    def load(self) -> None:
        """Load the model to the configured device."""
        self.load_to_device(self.device)
    
    def load_to_device(self, device: str) -> None:
        """
        Load model to the specified device.
        
        Args:
            device: Target device (e.g., "cuda:0", "cpu").
        """
        logger.info(f"Loading OpenAssistant model {self._name} to {device}")
        
        self._ensure_oa_import()
        
        # Try to use OpenAssistant's model loading utilities
        try:
            from model_training.models import get_specific_model
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = get_specific_model(
                self.model_path,
                seq2seqmodel=(self.model_type == "seq2seq"),
                without_head=self.use_reward_head
            )
            self.model = self.model.to(device)
            self.model.eval()
            
        except ImportError:
            # Fallback to standard transformers loading
            logger.warning("Could not import OpenAssistant utilities, using standard loading")
            from transformers import AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            ).to(device)
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
        return f"OpenAssistantAdapter(name={self._name}, type={self.model_type}, status={status})"
