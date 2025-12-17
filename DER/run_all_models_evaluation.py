#!/usr/bin/env python
"""
Comprehensive Model Evaluation Script for DER

This script evaluates all available models (from config and local directories)
on a test dataset, collecting metrics and producing a detailed dashboard.
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import traceback

# Set Windows-friendly environment variables
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import DER infrastructure
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adapters.registry import ModelRegistry
    from adapters.base import GenerationConfig
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import adapters.registry: {e}. Will use fallback model discovery.")
    REGISTRY_AVAILABLE = False


def get_system_stats():
    """Collect system statistics (reused from train_production.py)."""
    stats = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gpu': {},
        'cpu': {},
        'memory': {}
    }
    
    # GPU stats
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats['gpu']['available'] = True
            stats['gpu']['device_count'] = torch.cuda.device_count()
            stats['gpu']['current_device'] = torch.cuda.current_device()
            stats['gpu']['device_name'] = torch.cuda.get_device_name(0)
            stats['gpu']['memory_allocated_gb'] = torch.cuda.memory_allocated(0) / 1024**3
            stats['gpu']['memory_reserved_gb'] = torch.cuda.memory_reserved(0) / 1024**3
            stats['gpu']['max_memory_allocated_gb'] = torch.cuda.max_memory_allocated(0) / 1024**3
        else:
            stats['gpu']['available'] = False
    except:
        stats['gpu']['available'] = False
    
    # CPU stats
    try:
        import psutil
        stats['cpu']['percent'] = psutil.cpu_percent(interval=1)
        stats['cpu']['count'] = psutil.cpu_count()
        mem = psutil.virtual_memory()
        stats['memory']['total_gb'] = mem.total / 1024**3
        stats['memory']['used_gb'] = mem.used / 1024**3
        stats['memory']['percent'] = mem.percent
    except ImportError:
        stats['cpu']['percent'] = 'N/A (install psutil)'
    
    return stats


def discover_models_from_config(config_path: str, device: str) -> Dict[str, Any]:
    """Discover models from YAML config using ModelRegistry."""
    models = {}
    
    if not REGISTRY_AVAILABLE:
        logger.warning("ModelRegistry not available, skipping config-based discovery")
        return models
    
    try:
        registry = ModelRegistry.from_config(config_path)
        model_names = registry.list_models()
        
        for model_name in model_names:
            try:
                adapter = registry.get(model_name)
                models[model_name] = {
                    'type': 'registry',
                    'adapter': adapter,
                    'source': 'config',
                    'path': config_path
                }
                logger.info(f"Discovered model from config: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to get adapter for {model_name}: {e}")
                models[model_name] = {
                    'type': 'registry',
                    'adapter': None,
                    'source': 'config',
                    'error': str(e)
                }
    except Exception as e:
        logger.error(f"Failed to load model registry from {config_path}: {e}")
    
    return models


def discover_models_from_directory(models_dir: str, device: str) -> Dict[str, Any]:
    """Scan local directory for model files/folders."""
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.debug(f"Models directory not found: {models_dir}")
        return models
    
    # Look for HuggingFace-style model directories
    for item in models_path.iterdir():
        if not item.is_dir():
            continue
        
        # Check for common HF model files
        config_file = item / "config.json"
        has_model_files = any(item.glob("*.bin")) or any(item.glob("*.safetensors")) or (item / "pytorch_model.bin").exists()
        
        if config_file.exists() or has_model_files:
            model_name = item.name
            models[model_name] = {
                'type': 'transformers',
                'path': str(item),
                'source': 'local_directory',
                'adapter': None
            }
            logger.info(f"Discovered local model directory: {model_name}")
    
    # Also check for .pt/.bin files in root
    for item in models_path.iterdir():
        if item.is_file() and item.suffix in ['.pt', '.bin', '.pth']:
            model_name = item.stem
            if model_name not in models:  # Don't overwrite directory-based entries
                models[model_name] = {
                    'type': 'torch',
                    'path': str(item),
                    'source': 'local_file',
                    'adapter': None
                }
                logger.info(f"Discovered local model file: {model_name}")
    
    return models


def load_model_safely(model_name: str, model_info: Dict, device: str, timeout: int = 300) -> Tuple[Optional[Any], Optional[str], str]:
    """
    Attempt to load a model safely with timeout.
    
    Returns: (model_instance, tokenizer_instance, error_message)
    """
    start_time = time.time()
    error_msg = None
    model = None
    tokenizer = None
    
    try:
        # Use adapter if available
        if model_info.get('adapter') is not None:
            adapter = model_info['adapter']
            try:
                # Ensure adapter is loaded
                if hasattr(adapter, 'load'):
                    adapter.load()
                elif hasattr(adapter, 'load_to_device'):
                    adapter.load_to_device(device)
                
                # For adapters, we return the adapter itself
                return adapter, None, ""
            except Exception as e:
                error_msg = f"Adapter load failed: {str(e)}"
                logger.error(f"Failed to load {model_name} via adapter: {e}")
                return None, None, error_msg
        
        # Fallback: Direct transformers loading
        model_path = model_info.get('path')
        if not model_path:
            error_msg = "No model path specified"
            return None, None, error_msg
        
        if not TRANSFORMERS_AVAILABLE:
            error_msg = "transformers library not available"
            return None, None, error_msg
        
        # Check timeout
        if time.time() - start_time > timeout:
            error_msg = f"Load timeout after {timeout}s"
            return None, None, error_msg
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            error_msg = f"Tokenizer load failed: {str(e)}"
            return None, None, error_msg
        
        # Load model
        try:
            # CPU-safe loading
            if device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32 if TORCH_AVAILABLE else None,
                    trust_remote_code=True
                ).cpu()
            else:
                # GPU loading - skip quantization on CPU
                if model_info.get('load_in_4bit') or model_info.get('load_in_8bit'):
                    # Skip quantization for CPU
                    if device == "cpu":
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32 if TORCH_AVAILABLE else None,
                            trust_remote_code=True
                        ).cpu()
                    else:
                        try:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=model_info.get('load_in_4bit', False),
                                load_in_8bit=model_info.get('load_in_8bit', False)
                            )
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                quantization_config=quantization_config,
                                device_map=device,
                                trust_remote_code=True
                            )
                        except ImportError:
                            logger.warning(f"bitsandbytes not available, loading {model_name} without quantization")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16 if TORCH_AVAILABLE else None,
                                device_map=device,
                                trust_remote_code=True
                            )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if device != "cpu" and TORCH_AVAILABLE else torch.float32,
                        device_map=device if device != "cpu" else None,
                        trust_remote_code=True
                    )
                    if device == "cpu":
                        model = model.cpu()
            
            model.eval()
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer, ""
            
        except Exception as e:
            error_msg = f"Model load failed: {str(e)}"
            logger.error(f"Failed to load model {model_name}: {e}")
            return None, None, error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error loading {model_name}: {e}")
        return None, None, error_msg


def simple_similarity(text1: str, text2: str) -> float:
    """Calculate simple token overlap similarity."""
    if not text1 or not text2:
        return 0.0
    
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def exact_match(pred: str, ref: str) -> bool:
    """Check exact match (normalized)."""
    return pred.strip().lower() == ref.strip().lower()


def calculate_bertscore(predictions: List[str], references: List[str], device: str) -> Optional[float]:
    """Calculate BERTScore if available."""
    try:
        import bert_score
        from bert_score import BERTScorer
        
        # Try multilingual first, fallback to uncased
        try:
            scorer = BERTScorer(lang="en", device=device, model_type="bert-base-multilingual-cased")
        except:
            scorer = BERTScorer(lang="en", device=device, model_type="bert-base-uncased")
        
        P, R, F1 = scorer.score(predictions, references, verbose=False)
        return float(F1.mean().item())
    except Exception as e:
        logger.debug(f"BERTScore calculation failed: {e}")
        return None


def run_inference(
    model_name: str,
    model: Any,
    tokenizer: Optional[Any],
    test_data: List[Dict],
    device: str,
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    timeout_per_example: int = 60
) -> Dict[str, Any]:
    """Run inference on test data and collect metrics."""
    results = {
        'model_name': model_name,
        'examples_processed': 0,
        'success_count': 0,
        'failure_count': 0,
        'inference_times': [],
        'predictions': [],
        'references': [],
        'similarities': [],
        'exact_matches': 0,
        'errors': []
    }
    
    data_to_process = test_data[:max_examples] if max_examples else test_data
    
    # Check if using adapter (has generate method and name attribute)
    is_adapter = hasattr(model, 'generate') and hasattr(model, 'name')
    
    if is_adapter and tokenizer is None:
        # Adapters handle tokenization internally
        pass
    elif not is_adapter and tokenizer is None:
        raise ValueError(f"Tokenizer required for non-adapter model {model_name}")
    
    for i, example in enumerate(tqdm(data_to_process, desc=f"Inferencing {model_name}")):
        prompt = example.get('prompt', example.get('instruction', example.get('input', '')))
        reference = example.get('reference', example.get('output', example.get('target', '')))
        
        if not prompt:
            results['failure_count'] += 1
            results['errors'].append(f"Example {i}: No prompt found")
            continue
        
        start_time = time.time()
        
        try:
            # Use adapter if available
            if is_adapter:
                try:
                    from adapters.base import GenerationConfig
                    config = GenerationConfig(max_tokens=256, temperature=0.7)
                    response = model.generate(prompt, config)
                    prediction = response.text if hasattr(response, 'text') else str(response)
                except Exception as e:
                    raise RuntimeError(f"Adapter generation failed: {e}")
            else:
                # Direct model inference
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                # Remove token_type_ids if present
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
                
                if device != "cpu" and TORCH_AVAILABLE:
                    inputs = inputs.to(device)
                else:
                    inputs = {k: v.cpu() if TORCH_AVAILABLE else v for k, v in inputs.items()}
                
                with torch.no_grad() if TORCH_AVAILABLE else no_op():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
                    )
                
                # Decode output
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            
            # Check timeout
            if inference_time > timeout_per_example:
                results['failure_count'] += 1
                results['errors'].append(f"Example {i}: Timeout ({inference_time:.2f}s)")
                continue
            
            results['success_count'] += 1
            results['inference_times'].append(inference_time)
            results['predictions'].append(prediction)
            
            if reference:
                results['references'].append(reference)
                sim = simple_similarity(prediction, reference)
                results['similarities'].append(sim)
                if exact_match(prediction, reference):
                    results['exact_matches'] += 1
            
            results['examples_processed'] += 1
            
        except Exception as e:
            results['failure_count'] += 1
            error_msg = f"Example {i}: {str(e)}"
            results['errors'].append(error_msg)
            logger.debug(f"Error processing example {i} for {model_name}: {e}")
    
    # Calculate BERTScore if references available
    if results['references'] and len(results['references']) == len(results['predictions']):
        bertscore = calculate_bertscore(results['predictions'], results['references'], device)
        results['bertscore'] = bertscore
    
    return results


class no_op:
    """Context manager that does nothing."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def print_evaluation_dashboard(
    stats_before: Dict,
    stats_after: Dict,
    evaluation_time: float,
    model_results: Dict[str, Dict],
    evaluation_success: bool
):
    """Print evaluation dashboard matching train_production.py style."""
    print("\n" + "="*80)
    print(" " * 25 + "EVALUATION STATISTICS DASHBOARD")
    print("="*80)
    
    # Evaluation Summary
    print("\nEVALUATION SUMMARY")
    print("-" * 80)
    print(f"  Status: {'SUCCESS' if evaluation_success else 'FAILED'}")
    print(f"  Duration: {evaluation_time:.2f} seconds ({evaluation_time/60:.2f} minutes)")
    print(f"  Started: {stats_before['timestamp']}")
    print(f"  Ended: {stats_after['timestamp']}")
    
    # GPU Statistics
    print("\nGPU STATISTICS")
    print("-" * 80)
    if stats_after['gpu'].get('available'):
        gpu = stats_after['gpu']
        print(f"  Device: {gpu.get('device_name', 'Unknown')}")
        print(f"  Memory Allocated: {gpu.get('memory_allocated_gb', 0):.2f} GB")
        print(f"  Memory Reserved: {gpu.get('memory_reserved_gb', 0):.2f} GB")
        print(f"  Peak Memory Usage: {gpu.get('max_memory_allocated_gb', 0):.2f} GB")
        if gpu.get('memory_allocated_gb', 0) > 0:
            print(f"  Utilization: {(gpu.get('memory_allocated_gb', 0) / 24 * 100):.1f}% (assuming 24GB GPU)")
    else:
        print("  Status: No GPU available (evaluation on CPU)")
    
    # CPU & Memory Statistics
    print("\nCPU & MEMORY STATISTICS")
    print("-" * 80)
    if isinstance(stats_after['cpu'].get('percent'), (int, float)):
        print(f"  CPU Usage: {stats_after['cpu']['percent']:.1f}%")
        print(f"  CPU Cores: {stats_after['cpu'].get('count', 'N/A')}")
    else:
        print(f"  CPU Usage: {stats_after['cpu'].get('percent', 'N/A')}")
    
    if 'memory' in stats_after and stats_after['memory']:
        mem = stats_after['memory']
        print(f"  RAM Used: {mem.get('used_gb', 0):.2f} GB / {mem.get('total_gb', 0):.2f} GB")
        print(f"  RAM Usage: {mem.get('percent', 0):.1f}%")
    
    # Model Status
    print("\nMODEL STATUS")
    print("-" * 80)
    if model_results:
        ready_count = 0
        error_count = 0
        warn_count = 0
        
        for model_name, result in model_results.items():
            status = result.get('status', 'UNKNOWN')
            model_type = result.get('model_type', 'Unknown')
            status_msg = result.get('status_message', '')
            
            if status == 'READY':
                symbol = "READY"
                ready_count += 1
            elif status == 'ERROR':
                symbol = "ERROR"
                error_count += 1
            else:
                symbol = "WARN"
                warn_count += 1
            
            print(f"  {symbol:5s} {model_name:30s} [{model_type:20s}] {status_msg[:40]}")
        
        print(f"\n  Total Models: {len(model_results)}")
        print(f"  Ready: {ready_count} | Errors: {error_count} | Warnings: {warn_count}")
    else:
        print("  No model results available")
    
    # Performance Metrics
    print("\nPERFORMANCE METRICS")
    print("-" * 80)
    
    for model_name, result in model_results.items():
        if result.get('status') != 'READY':
            continue
        
        metrics = result.get('metrics', {})
        examples = metrics.get('examples_processed', 0)
        avg_time = metrics.get('avg_inference_time', 0.0)
        bertscore = metrics.get('bertscore')
        avg_sim = metrics.get('avg_similarity', 0.0)
        exact_matches = metrics.get('exact_matches', 0)
        outputs_path = result.get('outputs_path', 'N/A')
        
        print(f"\n  Model: {model_name}")
        print(f"    Examples Processed: {examples}")
        print(f"    Avg Inference Time: {avg_time:.3f}s")
        if bertscore is not None:
            print(f"    BERTScore (F1): {bertscore:.4f}")
        print(f"    Avg Similarity: {avg_sim:.4f}")
        print(f"    Exact Matches: {exact_matches}")
        print(f"    Outputs Saved: {outputs_path}")
    
    print("\n" + "="*80)
    print(" " * 28 + "END OF STATISTICS")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser("DER Model Evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test data JSONL file")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference (default: 1)")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to process per model")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per model load/inference in seconds")
    parser.add_argument("--skip-remote", action="store_true",
                       help="Skip remote model downloads")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--config", type=str, default="./config/models_production.yaml",
                       help="Path to model config YAML")
    parser.add_argument("--models-dir", type=str, default="../models",
                       help="Directory to scan for local models")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Adjust device for CPU
    if args.device == "cpu" or (args.device.startswith("cuda") and TORCH_AVAILABLE and not torch.cuda.is_available()):
        args.device = "cpu"
        logger.info("Using CPU for evaluation")
    
    # Collect statistics before
    logger.info("Collecting system statistics...")
    stats_before = get_system_stats()
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    test_data = []
    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        logger.info(f"Loaded {len(test_data)} examples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return 1
    
    # Discover models
    logger.info("Discovering models...")
    all_models = {}
    
    # From config
    if os.path.exists(args.config):
        config_models = discover_models_from_config(args.config, args.device)
        all_models.update(config_models)
    
    # From local directory
    local_models = discover_models_from_directory(args.models_dir, args.device)
    all_models.update(local_models)
    
    if not all_models:
        logger.error("No models discovered!")
        return 1
    
    logger.info(f"Discovered {len(all_models)} models")
    
    # Create output directories
    output_dir = Path("./evaluation_outputs")
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Evaluate each model
    model_results = {}
    start_time = time.time()
    
    for model_name, model_info in all_models.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*80}")
        
        result = {
            'model_name': model_name,
            'status': 'ERROR',
            'model_type': model_info.get('type', 'unknown'),
            'status_message': '',
            'metrics': {},
            'outputs_path': '',
            'error': None
        }
        
        try:
            # Load model
            model, tokenizer, error_msg = load_model_safely(
                model_name, model_info, args.device, args.timeout
            )
            
            if model is None:
                result['status'] = 'ERROR'
                result['status_message'] = error_msg or "Failed to load model"
                result['error'] = error_msg
                model_results[model_name] = result
                continue
            
            # Run inference
            inference_results = run_inference(
                model_name,
                model,
                tokenizer,
                test_data,
                args.device,
                batch_size=args.batch_size,
                max_examples=args.max_examples,
                timeout_per_example=args.timeout
            )
            
            # Calculate metrics
            metrics = {
                'examples_processed': inference_results['examples_processed'],
                'success_count': inference_results['success_count'],
                'failure_count': inference_results['failure_count'],
                'avg_inference_time': (
                    sum(inference_results['inference_times']) / len(inference_results['inference_times'])
                    if inference_results['inference_times'] else 0.0
                ),
                'bertscore': inference_results.get('bertscore'),
                'avg_similarity': (
                    sum(inference_results['similarities']) / len(inference_results['similarities'])
                    if inference_results['similarities'] else 0.0
                ),
                'exact_matches': inference_results['exact_matches']
            }
            
            result['status'] = 'READY'
            result['status_message'] = f"Processed {metrics['examples_processed']} examples"
            result['metrics'] = metrics
            
            # Save outputs
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            outputs_path = model_output_dir / "predictions.jsonl"
            
            with open(outputs_path, 'w', encoding='utf-8') as f:
                for i, pred in enumerate(inference_results['predictions']):
                    output = {
                        'model': model_name,
                        'prediction': pred,
                        'reference': inference_results['references'][i] if i < len(inference_results['references']) else None,
                        'similarity': inference_results['similarities'][i] if i < len(inference_results['similarities']) else None
                    }
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
            
            result['outputs_path'] = str(outputs_path)
            logger.info(f"Saved outputs to {outputs_path}")
            
            # Cleanup
            if hasattr(model, 'unload'):
                try:
                    model.unload()
                except:
                    pass
            elif TORCH_AVAILABLE:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['status_message'] = f"Exception: {str(e)[:50]}"
            result['error'] = traceback.format_exc()
            logger.error(f"Error evaluating {model_name}: {e}")
        
        model_results[model_name] = result
    
    evaluation_time = time.time() - start_time
    stats_after = get_system_stats()
    
    # Determine success
    evaluation_success = any(r.get('status') == 'READY' for r in model_results.values())
    
    # Print dashboard
    print_evaluation_dashboard(
        stats_before,
        stats_after,
        evaluation_time,
        model_results,
        evaluation_success
    )
    
    # Save evaluation report
    report_path = output_dir / "evaluation_report.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'device': args.device,
        'test_data_path': args.data,
        'num_test_examples': len(test_data),
        'evaluation_time_seconds': evaluation_time,
        'success': evaluation_success,
        'model_results': {
            name: {
                'status': r['status'],
                'model_type': r['model_type'],
                'status_message': r['status_message'],
                'metrics': r['metrics'],
                'outputs_path': r['outputs_path'],
                'error': r.get('error')
            }
            for name, r in model_results.items()
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    return 0 if evaluation_success else 1


if __name__ == "__main__":
    sys.exit(main())

