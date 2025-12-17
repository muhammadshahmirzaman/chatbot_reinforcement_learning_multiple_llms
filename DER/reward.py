
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# BERTScore imports guarded to avoid hard failures during import.
try:
    import bert_score
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except Exception as e:  # broad to catch missing deps
    bert_score = None
    BERTScorer = None
    BERTSCORE_AVAILABLE = False

from utils.bart_score import BARTScorer
try:
    from bleurt import score
    BLEURT_AVAILABLE = True
except ImportError:
    score = None
    BLEURT_AVAILABLE = False

class Bert_calculate:
    def __init__(self, device=None):
        self.tokenizer = None
        self.scorer = None
        self.max_tokens = 512
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def load_weight(self):
        """Load BERTScore weights with safe CPU-first defaults."""
        if not BERTSCORE_AVAILABLE:
            print("BERTScore is not installed; rewards will be zero.")
            self.tokenizer = None
            self.scorer = None
            return

        print(f"loading pretrained bertscore on {self.device}...")

        preferred_model = "bert-base-multilingual-cased"
        fallback_model = "bert-base-uncased"

        def _model_exists_locally(model_name: str) -> bool:
            return os.path.isdir(model_name) and bool(os.listdir(model_name))

        # Prefer HF download unless a local directory truly exists with files.
        model_candidates = [preferred_model, fallback_model]

        for model_name in model_candidates:
            try:
                # Only treat as local if a non-empty directory is present.
                force_local = _model_exists_locally(model_name)
                if force_local:
                    print(f"Using local BERTScore model at {model_name}")
                self.tokenizer = bert_score.utils.get_tokenizer(model_type=model_name)
                self.scorer = BERTScorer(
                    lang="en",
                    device=self.device,
                    model_type=model_name # ,
                    # use_fast=True
                )
                return
            except Exception as e:
                print(f"BERTScore load failed for {model_name}: {e}")
                continue

        # Final fallback: disable scorer gracefully
        print("BERTScore could not be loaded; rewards will be zero.")
        self.tokenizer = None
        self.scorer = None

    def bertscore_calculate(self, candidates, references, batchsize):
        if self.scorer is None:
            return [0.0] * len(candidates)
            
        for i in range(len(candidates)):
            candidate = candidates[i]
            reference = references[i]
            # Handle empty strings
            if not candidate or not reference:
                candidates[i] = candidate if candidate else ""
                references[i] = reference if reference else ""
                continue
                
            candidate_tokens = self.tokenizer.tokenize(
                self.tokenizer.decode(self.tokenizer.encode(candidate, add_special_tokens=True)))
            reference_tokens = self.tokenizer.tokenize(
                self.tokenizer.decode(self.tokenizer.encode(reference, add_special_tokens=True)))
            
            if len(candidate_tokens) > self.max_tokens or len(reference_tokens) > self.max_tokens:
                candidate_tokens = candidate_tokens[:self.max_tokens]
                reference_tokens = reference_tokens[:self.max_tokens]
                candidate = self.tokenizer.convert_tokens_to_string(candidate_tokens)
                reference = self.tokenizer.convert_tokens_to_string(reference_tokens)
                min_len = min(len(candidate), len(reference))
                candidates[i] = candidate[:min_len]
                references[i] = reference[:min_len]
                
        P, R, F1 = self.scorer.score(candidates, references, verbose=False, batch_size=batchsize)
        return F1.numpy().tolist()

class Bart_calculate:
    def __init__(self):
        self.bart_scorer = None

    def load_weight(self):
        # Keep BartScore disabled in CPU-only flows to avoid GPU dependency.
        print("BartScore loading skipped on CPU-only setup.")
        self.bart_scorer = None
        
    def calculate_bart_score(self, candidate, reference):
        torch.cuda.empty_cache()
        results = []
        for i in range(len(candidate)):
            result = self.bart_scorer.score([candidate[i]], [reference[i]], batch_size=1)
            results.append(result[0])
        return results

class Bleurt_calculate:
    def __init__(self):
        self.scorer = None

    def load_weight(self):
        if not BLEURT_AVAILABLE:
            print("BLEURT not available, skipping load.")
            return
        print("loading pretrained blerut...")
        checkpoint = "./bleurt/bleurt-base-128"
        self.scorer = score.BleurtScorer(checkpoint)

    def calculate_bleurt_score(self, candidate, reference):
        if not self.scorer:
            return [0.0] * len(candidate)
            
        results = []
        for i in range(len(candidate)):
            scores = self.scorer.score(references=[reference[i]], candidates=[candidate[i]])
            assert isinstance(scores, list) and len(scores) == 1
            results.append(scores[0])
        return results

class Reward_calculate:
    def __init__(self, device=None):
        self.bert_calc = Bert_calculate(device=device)

    def load_checkpoint(self):
        self.bert_calc.load_weight()

    def reward_calc(self, candidate, reference):
        bert = self.bert_calc.bertscore_calculate(candidate, reference, batchsize=1)
        reward = bert
        return reward

class KTP_calculate:
    """
    Knowledge Transfer Prompt (KTP) Calculator.
    
    This class implements the KTP mechanism which:
    1. Inputs Q, A, and Ground Truth
    2. Uses LLM + Transformer (Reward_calculate) for evaluation
    3. Calculates Reward
    4. Prepares Q&A for the next state
    """
    def __init__(self, device='cpu'):
        self.reward_engine = Reward_calculate(device=device)
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load_checkpoint(self):
        # Load reward engine
        self.reward_engine.load_checkpoint()
        
        # Load KTP LLM (using opt-125m as it's available locally)
        print("loading KTP model (opt-125m)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
            self.model = AutoModelForCausalLM.from_pretrained('./opt-125m')
            # Handle device placement safely
            if self.device != 'cpu' and torch.cuda.is_available():
                self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load KTP model: {e}")
            self.model = None
            
    def process(self, question, generated_answer, ground_truth):
        """
        Process the turn using KTP.
        
        Args:
            question: The original question
            generated_answer: The answer generated by the selected model
            ground_truth: The reference answer
            
        Returns:
            reward: Calculated reward score
            next_state_info: Information/Text for the next state
        """
        # 1. Calculate Reward (Transformer part)
        # Using existing Reward_calculate (BERTScore)
        reward = self.reward_engine.reward_calc([generated_answer], [ground_truth])[0]
        
        # 2. Prepare Next State (LLM part)
        # We generate a critique or feedback based on the answer and ground truth
        
        if self.model and self.tokenizer:
            prompt = (f"Question: {question}\n"
                     f"Student Answer: {generated_answer}\n"
                     f"Reference Answer: {ground_truth}\n"
                     f"Compare the student answer to the reference. "
                     f"Provide a brief hint or follow-up question to help improve the answer:\n")
            
            inputs = self.tokenizer(prompt, return_tensors='pt')
            if self.device != 'cpu' and torch.cuda.is_available():
                inputs = inputs.to(self.device)
                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
                
            feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the generated part (simple heuristic)
            feedback = feedback[len(prompt):].strip()
            
            # The next state text will be the current answer + the KTP feedback
            next_state_info = f"{generated_answer}\n[KTP Hint]: {feedback}"
        else:
            # Fallback if KTP model not loaded
            next_state_info = generated_answer
        
        return reward, next_state_info

    def reward_calc(self, candidate, reference):
        """Backward compatibility wrapper for LegacyEnvironment."""
        return self.reward_engine.reward_calc(candidate, reference)

if __name__ == '__main__':
    pass