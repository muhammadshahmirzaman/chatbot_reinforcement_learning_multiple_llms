"""
DER Environment for LLM ensemble reasoning.

This module provides the reinforcement learning environment for the DER
(Dynamic Ensemble Reasoning) system, supporting both online and offline models
through the adapter registry.
"""

from typing import Optional, List
from transformers import AutoTokenizer

from adapters.registry import ModelRegistry
from adapters.base import GenerationConfig


class Environment:
    """
    RL environment for dynamic LLM selection.
    
    The environment takes actions (model selections) and returns states
    (encoded question-answer pairs) and rewards (answer quality scores).
    """
    
    # Knowledge transfer prompt template
    KNOWLEDGE_TRANSFER_PROMPT = (
        "{} \nThese are the answers to the question from other student: \n{} \n"
        "Using other student's answers as additional advice, you need to give "
        "a more satisfactory answer directly. DO NOT mention other students."
    )
    
    # State encoding template
    STATE_TEMPLATE = "Question:{} Answer:{}"
    
    def __init__(
        self,
        actor,
        reward_calculator,
        model_registry: ModelRegistry,
        state_device: str = "cpu",
        actor_device: str = "cpu",
        stop_threshold: float = 0.73
    ):
        """
        Initialize the environment.
        
        Args:
            actor: The actor network for action selection.
            reward_calculator: Calculator for reward computation.
            model_registry: Registry containing model adapters.
            state_device: Device for state tensors.
            actor_device: Device for the actor network.
            stop_threshold: Score threshold for early stopping.
        """
        self.target = None
        self.state_s = None  # String state (accumulated answer)
        self.attention_masks_e = None
        self.actor = actor.to(actor_device)
        self.state = None  # Tensor state
        self.action_sequence = []
        self.ifinit = True
        self.question = None
        
        # Model registry for accessing LLMs
        self.model_registry = model_registry
        self.model_names = model_registry.get_available_models()
        
        # Configuration
        self.state_device = state_device
        self.stop_threshold = stop_threshold
        
        # Tokenizer for state encoding
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
        
        # Reward calculator
        self.reward_calculator = reward_calculator
        
        # Generation config
        self.generation_config = GenerationConfig()
    
    @property
    def num_models(self) -> int:
        """Return the number of available models."""
        return len(self.model_names)
    
    def get_model_costs(self) -> List[float]:
        """Return list of model costs for reward shaping."""
        return [self.model_registry.get_model_cost(name) for name in self.model_names]
    
    def reset(self, initial_prompt, attention_masks, sources, target):
        """
        Reset the environment for a new episode.
        
        Args:
            initial_prompt: Initial tokenized prompt.
            attention_masks: Attention masks for the prompt.
            sources: The question text.
            target: The reference answer for reward calculation.
        """
        self.question = sources
        self.state_s = " "
        self.state = initial_prompt
        self.attention_masks_e = attention_masks
        self.target = target
        self.action_sequence = []
        self.ifinit = True
    
    def step(self, actions):
        """
        Take an action in the environment.
        
        Args:
            actions: Index of the model to use (0 to num_models-1).
            
        Returns:
            Tuple of (answer_text, score, stop_flag).
        """
        # Get model name from action index
        model_name = self.model_names[actions]
        
        # Prepare input prompt
        if self.ifinit:
            inputs = self.question
        else:
            # Use KTP feedback/state if available, otherwise standard template
            feedback = getattr(self, 'last_ktp_feedback', self.state_s)
            inputs = self.KNOWLEDGE_TRANSFER_PROMPT.format(self.question, feedback)
        
        # Generate response using the selected model adapter
        adapter = self.model_registry.get(model_name)
        if adapter is None:
            raise RuntimeError(f"Model '{model_name}' not found in registry")
        
        response = adapter.generate(inputs, self.generation_config)
        self.state_s = response.text
        
        # Track action sequence
        self.action_sequence.append(actions)
        self.ifinit = False
        
        # Use KTP for Reward and Next State Generation
        # KTP inputs: Question, Answer, Ground Truth
        # KTP returns: Reward, Next State Info (feedback/hint)
        reward, next_state_info = self.reward_calculator.process(
            self.question, 
            self.state_s, 
            self.target
        )
        
        # Store feedback for next iteration
        self.last_ktp_feedback = next_state_info
        
        # Update state encoding for Agent (Actor/Critic)
        # We use the KTP feedback as part of the state representation
        new_inputs = self.STATE_TEMPLATE.format(self.question, next_state_info)
        new_inputs = self.preprocess_text_data(new_inputs)
        self.state = new_inputs['input_ids'].to(self.state_device)
        self.attention_masks_e = new_inputs['attention_mask'].to(self.state_device)
        
        # Check stopping condition
        score = [reward] # Maintain list format for compatibility
        
        if score[0] >= self.stop_threshold:
            stop = 1
        else:
            stop = 0
        
        return self.state_s, score, stop
    
    def preprocess_text_data(self, data):
        """
        Tokenize text data for state encoding.
        
        Args:
            data: Text to tokenize.
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        inputs = self.tokenizer(
            data,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }


# Backward compatibility: Legacy environment that uses the old LLMs function
class LegacyEnvironment:
    """
    Legacy environment using the original LLM_models.py.
    
    This class maintains backward compatibility with the original DER
    implementation while the new adapter-based system is being tested.
    """
    
    def __init__(self, actor, reward_calculator):
        """Initialize with original behavior."""
        from utils.LLM_models import LLMs
        
        self.target = None
        self.state_s = None
        self.attention_masks_e = None
        try:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        self.actor = actor.to(device)
        self.state = None
        self.action_sequence = []
        self.ifinit = True
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
        self.reward_calculator = reward_calculator
        self._llms_func = LLMs
        
        # Original hardcoded model list
        self.model_names = [
            "koala-7B-HF", "Vicuna-13B", "alpaca-13B", "dolly-12B",
            "baize-13B", "stablelm-7B", "mpt-7B", "OpenAssistant-12B",
            "t5-xxl", "moss", "chatglm-6B"
        ]
    
    @property
    def num_models(self) -> int:
        return len(self.model_names)
    
    def reset(self, initial_prompt, attention_masks, sources, target):
        self.question = sources
        self.state_s = " "
        self.state = initial_prompt
        self.attention_masks_e = attention_masks
        self.target = target
        self.action_sequence = []
        self.ifinit = True
    
    def step(self, actions):
        model_name = self.model_names
        PROMPT = "{} \nThese are the answers to the question from other student: \n{} \nUsing other student's answers as additional advice, you need to give a more satisfactory answer directly. DO NOT mention other students."
        State = "Question:{} Answer:{}"
        
        if self.ifinit:
            inputs = self.question
        else:
            inputs = PROMPT.format(self.question, self.state_s)
        
        self.state_s = self._llms_func(model_name[actions], inputs)
        
        new_inputs = State.format(self.question, self.state_s)
        new_inputs = self.preprocess_text_data(new_inputs)
        self.state = new_inputs['input_ids'].to('cuda:3')
        self.attention_masks_e = new_inputs['attention_mask'].to('cuda:3')
        
        self.action_sequence.append(actions)
        self.ifinit = False
        
        score = self.reward_calculator.reward_calc([self.state_s], [self.target])
        if score[0] >= 0.73:
            stop = 1
        else:
            stop = 0
        return self.state_s, score, stop
    
    def preprocess_text_data(self, data):
        inputs = self.tokenizer(
            data,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }
