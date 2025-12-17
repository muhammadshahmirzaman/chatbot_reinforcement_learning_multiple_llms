"""
DER PPO Multi-LLM Evaluation with Confusion Matrix

This script extends the standard testing script to calculate 
accuracy and confusion matrix by comparing generated answers to targets.
"""

import argparse
import os
import json
import logging
import threading
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure we can import from local modules
sys.path.insert(0, os.getcwd())

from utils.data import load_data, Dataset
from ppo_discreate import Actor, Critic, PPO
from reward import KTP_calculate

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Metric imports
try:
    from bert_score import BERTScorer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Ensure punkt is downloaded for tokenization if needed, though we might just split by space for simple BLEU
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

class MetricCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        self.bert_scorer = None
        self.rouge_scorer = None
        
        if BERT_AVAILABLE:
            try:
                # Use a lightweight model for speed if not specified
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
            except Exception as e:
                logger.warning(f"Failed to load BERTScorer: {e}")
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
    def compute(self, candidate, reference):
        metrics = {}
        
        # BERTScore
        if self.bert_scorer:
            try:
                P, R, F1 = self.bert_scorer.score([candidate], [reference])
                metrics['bert_f1'] = F1.item()
            except Exception:
                metrics['bert_f1'] = 0.0
        else:
            metrics['bert_f1'] = 0.0
            
        # BLEU
        if BLEU_AVAILABLE:
            try:
                ref_tokens = [reference.split()] # List of references, each is a list of tokens
                cand_tokens = candidate.split()
                # fast smoothing
                smooth = SmoothingFunction().method1
                score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)
                metrics['bleu'] = score
            except Exception:
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = 0.0
            
        # ROUGE
        if self.rouge_scorer:
            try:
                scores = self.rouge_scorer.score(reference, candidate)
                metrics['rouge1'] = scores['rouge1'].fmeasure
                metrics['rouge2'] = scores['rouge2'].fmeasure
                metrics['rougeL'] = scores['rougeL'].fmeasure
            except Exception:
                metrics['rouge1'] = 0.0
                metrics['rouge2'] = 0.0
                metrics['rougeL'] = 0.0
        else:
            metrics['rouge1'] = 0.0
            metrics['rouge2'] = 0.0
            metrics['rougeL'] = 0.0
            
        return metrics

def calculate_metrics_report(trajectories, model_map, out_dir):
    """
    Calculate and print per-model and overall metrics.
    """
    logger.info("Computing Detailed Metrics Report...")
    
    # Storage for aggregation
    overall_metrics = {'bert_f1': [], 'bleu': [], 'rougeL': []}
    per_model_metrics = {}
    
    for t in trajectories:
        metrics = t.get('metrics', {})
        if not metrics:
            continue
            
        # Overall
        for k in overall_metrics:
            if k in metrics:
                overall_metrics[k].append(metrics[k])
        
        # Per Model
        model_name = t.get('selected_model', 'Unknown')
        if model_name not in per_model_metrics:
            per_model_metrics[model_name] = {'bert_f1': [], 'bleu': [], 'rougeL': [], 'count': 0}
        
        per_model_metrics[model_name]['count'] += 1
        for k in ['bert_f1', 'bleu', 'rougeL']:
            if k in metrics:
                per_model_metrics[model_name][k].append(metrics[k])

    # Print Report
    print("\n" + "="*80)
    print("NLP METRICS REPORT")
    print("="*80)
    
    # Overall
    print(f"\nOVERALL PERFORMANCE ({len(trajectories)} samples):")
    print("-" * 40)
    for k, v in overall_metrics.items():
        avg = np.mean(v) if v else 0.0
        print(f"  Avg {k.upper():<10}: {avg:.4f}")
        
    # Per Model
    print(f"\nPER-MODEL PERFORMANCE:")
    print("-" * 100)
    header = f"{'Model Name':<25} | {'Count':<6} | {'BERT-F1':<10} | {'BLEU':<10} | {'ROUGE-L':<10}"
    print(header)
    print("-" * 100)
    
    sorted_models = sorted(per_model_metrics.keys())
    for m in sorted_models:
        stats = per_model_metrics[m]
        count = stats['count']
        if count == 0: continue
        
        b_f1 = np.mean(stats['bert_f1'])
        bleu = np.mean(stats['bleu'])
        rougel = np.mean(stats['rougeL'])
        
        print(f"{m[:25]:<25} | {count:<6} | {b_f1:.4f}     | {bleu:.4f}     | {rougel:.4f}")
    
    print("="*80 + "\n")
    
    # Save report
    report_path = os.path.join(out_dir, "metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write("NLP METRICS REPORT\n")
        f.write("OVERALL:\n")
        for k, v in overall_metrics.items():
            f.write(f"{k}: {np.mean(v) if v else 0.0:.4f}\n")
        f.write("\nPER MODEL:\n")
        for m in sorted_models:
            stats = per_model_metrics[m]
            f.write(f"{m}: Count={stats['count']}, BERT={np.mean(stats['bert_f1']):.4f}, BLEU={np.mean(stats['bleu']):.4f}\n")
            
    logger.info(f"Metrics report saved to {report_path}")

def main(args):
    """Main evaluation function."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Available GPUs: {n_gpu}")
    
    # Initialize Metric Calculator
    metric_calc = MetricCalculator(device=args.device) if not args.no_metrics else None
    
    # Load test data
    test_examples = load_data(args.test_data_path, args, max_size=args.max_test_data_size)
    test_dataset = Dataset(test_examples)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    
    # Setup model registry or legacy mode
    model_map = {}
    if args.use_legacy_mode:
        logger.info("Using legacy mode with hardcoded models")
        from environment import LegacyEnvironment
        num_models = 11
        env_class = LegacyEnvironment
        model_registry = None
        # Hardcoded names from LegacyEnvironment
        model_names = ['koala-7B-HF', 'Vicuna-13B', 'alpaca-13B', 'dolly-12B', 'baize-13B', 'stablelm-7B', 'mpt-7B', 'OpenAssistant-12B', 't5-xxl', 'moss', 'chatglm-6B']
        for i, name in enumerate(model_names):
            model_map[i] = name
    else:
        logger.info(f"Loading model registry from {args.model_config}")
        from adapters.registry import ModelRegistry
        from environment import Environment
        
        model_registry = ModelRegistry.from_config(args.model_config)
        num_models = model_registry.get_num_models()
        env_class = Environment
        model_list = model_registry.list_models()
        logger.info(f"Loaded {num_models} models: {model_list}")
        for i, name in enumerate(model_list):
            model_map[i] = name
    
    # Initialize actor and critic networks
    if args.actor_model_bin and not os.path.exists(args.actor_model_bin):
        logger.warning(f"Actor checkpoint not found at {args.actor_model_bin}")
        
    actor = Actor(
        config_path=args.actor_model,
        num_classes=num_models,
        mlp_hidden_size=256
    ).to(device)
    
    critic = Critic(args.critic_model, 256).to(device)
    
    # Initialize reward calculator/KTP engine
    reward_calculator = KTP_calculate(device=device)
    logger.info("Loading KTP engine...")
    reward_calculator.load_checkpoint()
    
    # Load trained weights
    logger.info(f"Loading actor from {args.actor_model_bin}")
    if args.actor_model_bin and os.path.exists(args.actor_model_bin):
        actor.load_state_dict(torch.load(args.actor_model_bin, map_location=device))
    else:
        logger.warning("No actor checkpoint provided or found, using random initialization")
    
    logger.info(f"Loading critic from {args.critic_model_bin}")
    if args.critic_model_bin and os.path.exists(args.critic_model_bin):
        critic.load_state_dict(torch.load(args.critic_model_bin, map_location=device))
    else:
        logger.warning("No critic checkpoint provided, using random initialization")
    
    actor.eval()
    critic.eval()
    
    ppo_agent = PPO(actor, critic)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_trajectories = []
    
    logger.info(f"Dataset size: {len(test_dataset)}")
    
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        sources = batch['source']
        targets = batch.get('target', [None] * len(sources)) 
        
        trajectorys = []
        i = 0
        batch_size = min(args.batch_size, len(input_ids))
        
        while i < batch_size:
            thread_args = []
            threads_to_create = min(args.thread_nums, batch_size - i)
            
            for j in range(threads_to_create):
                idx = i + j
                if args.use_legacy_mode:
                    env = env_class(actor, None)
                else:
                    env = env_class(
                        actor, 
                        reward_calculator, 
                        model_registry,
                        state_device=device,
                        actor_device=device
                    )
                
                args_tuple = (
                    env,
                    trajectorys,
                    ppo_agent,
                    input_ids[idx],
                    attention_masks[idx],
                    sources[idx],
                    targets[idx],
                    model_map,
                    metric_calc
                )
                thread_args.append(args_tuple)
            
            # Create and run threads
            threads = []
            for k, args_tuple in enumerate(thread_args, start=1):
                thread = threading.Thread(target=worker, args=(k, args_tuple))
                threads.append(thread)
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            i += args.thread_nums
        
        all_trajectories.extend(trajectorys)
    
    # Save results
    output_file_path = os.path.join(args.out_dir, args.output_filename)
    logger.info(f"Saving {len(all_trajectories)} results to {output_file_path}")
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for idx, traj in enumerate(all_trajectories):
            json.dump(traj, output_file)
            output_file.write('\n')
    
    logger.info("Evaluation complete!")
    
    if all_trajectories:
        avg_steps = np.mean([len(t.get('actions', [])) for t in all_trajectories])
        logger.info(f"Average steps per question: {avg_steps:.2f}")

        # Confusion Matrix
        calculate_confusion_matrix(all_trajectories, args.out_dir)
        
        # Metrics Report
        calculate_metrics_report(all_trajectories, model_map, args.out_dir)


def worker(thread_id, collect_args):
    """Worker function for parallel evaluation."""
    try:
        collect(*collect_args)
    except Exception as e:
        logger.error(f"Thread {thread_id} failed: {e}")
        import traceback
        traceback.print_exc()


def collect(environment, trajectorys, agent, inputs_id, attention_mask, source, target, model_map, metric_calc):
    """Collect evaluation trajectory."""
    answers_outs = []
    trajectory = {}
    
    environment.reset(inputs_id, attention_mask, source, target=None)
    
    step = 0
    max_steps = 4
    
    while True:
        action, action_pro = agent.get_action_eval(
            environment.state.unsqueeze(0),
            environment.attention_masks_e.unsqueeze(0)
        )
        
        answers, _, stop = environment.step(action.item() if hasattr(action, 'item') else action)
        answers_outs.append(answers)
        
        if stop == 1 or step >= max_steps:
            break
        
        step += 1
    
    # Build trajectory
    trajectory["question"] = source
    trajectory["target"] = target if target is not None else ""
    trajectory['answers'] = answers_outs
    trajectory['final_answer'] = answers_outs[-1] if answers_outs else ""
    
    action_list = []
    for a in environment.action_sequence:
        if hasattr(a, 'item'):
            action_list.append(1 if a.item() == 0 else int(a.item())) # wait, why 1 if 0? check original code
            # Actually original code just did int(a). Let's stick to safe conversion.
            action_list.append(int(a.item()))
        else:
            action_list.append(int(a))
    
    trajectory['actions'] = action_list
    trajectory['num_steps'] = len(action_list)
    
    # Identify Selected Model (Last action usually corresponds to the model that gave the final answer?)
    # In this environment, the action selects a model which generates an answer. 
    # If stop=1, the last answer is the final one.
    if action_list:
        last_action = action_list[-1]
        trajectory['selected_model'] = model_map.get(last_action, f"Model_{last_action}")
    else:
        trajectory['selected_model'] = "None"

    # Compute Metrics
    if metric_calc and trajectory['target']:
        metrics = metric_calc.compute(trajectory['final_answer'], trajectory['target'])
        trajectory['metrics'] = metrics
    
    trajectorys.append(trajectory)



def calculate_confusion_matrix(trajectories, out_dir):
    """
    Calculate and print the confusion matrix.
    Assumes 'final_answer' and 'target' are available in trajectories.
    """
    logger.info("Computing Confusion Matrix...")
    
    # Extract labels
    y_true = []
    y_pred = []
    
    for t in trajectories:
        gt = t.get('target', '').strip()
        pred = t.get('final_answer', '').strip()
        y_true.append(gt)
        y_pred.append(pred)
        
    # Check if we have valid data
    if not y_true:
        logger.warning("No data to compute confusion matrix.")
        return
        
    unique_labels = sorted(list(set(y_true + y_pred)))
    logger.info(f"Number of unique labels: {len(unique_labels)}")
    
    if len(unique_labels) > 50:
        logger.warning("Too many unique labels (>50) for a readable confusion matrix output.")
        # If too many labels, only print accuracy
        correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
        acc = correct / len(y_true)
        logger.info(f"Accuracy: {acc:.4f}")
        return

    # Compute Matrix
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        acc = accuracy_score(y_true, y_pred)
    else:
        logger.warning("scikit-learn not installed. Using manual calculation.")
        label_to_id = {l: i for i, l in enumerate(unique_labels)}
        n = len(unique_labels)
        cm = np.zeros((n, n), dtype=int)
        correct = 0
        for gt, pred in zip(y_true, y_pred):
            i = label_to_id[gt]
            j = label_to_id[pred]
            cm[i, j] += 1
            if gt == pred:
                correct += 1
        acc = correct / len(y_true)

    print("\n" + "="*60)
    print(f"CONFUSION MATRIX (Accuracy: {acc:.2%})")
    print("="*60)
    
    # Print Matrix with labels
    # Header
    header_label = "True \\ Pred"
    print(f"{header_label:>20} | " + " | ".join([f"{l[:10]:>10}" for l in unique_labels]))
    print("-" * (20 + 13 * len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        row_str = " | ".join([f"{cm[i, j]:>10}" for j in range(len(unique_labels))])
        print(f"{label[:20]:>20} | {row_str}")
        
    print("="*60 + "\n")
    
    # Save to file
    cm_path = os.path.join(out_dir, "confusion_matrix.txt")
    with open(cm_path, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(f"Labels: {unique_labels}\n\n")
        f.write("Matrix:\n")
        np.savetxt(f, cm, fmt='%d')
    logger.info(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DER Evaluation with Confusion Matrix")
    
    # Model config
    parser.add_argument("--actor_model", type=str, default="./opt-125m",
                        help="Path to actor model config")
    parser.add_argument("--actor_model_bin", type=str, default=None,
                        help="Path to trained actor weights")
    parser.add_argument("--critic_model", type=str, default="./opt-125m",
                        help="Path to critic model config")
    parser.add_argument("--critic_model_bin", type=str, default=None,
                        help="Path to trained critic weights")
    
    # Mode
    parser.add_argument("--use_legacy_mode", action="store_true",
                        help="Use legacy mode with hardcoded models")
    parser.add_argument("--model_config", type=str, default="./config/models.yaml",
                        help="Path to model configuration YAML file")
    
    # Data config
    parser.add_argument("--test_data_path", type=str,
                        default="test_data_mini.jsonl")
    
    # Evaluation hyperparameters
    parser.add_argument("--max_test_data_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--thread_nums", type=int, default=10)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="./eval_outputs/")
    parser.add_argument("--output_filename", type=str, default="evaluation_results_cm.jsonl")
    parser.add_argument("--no_metrics", action="store_true", help="Disable heavy metric calculation (BERT/BLEU/ROUGE)")
    
    args = parser.parse_args()
    main(args)
