"""
DER PPO Multi-LLM Testing/Evaluation Script

This script evaluates a trained PPO agent on test data,
generating answers using the learned model selection policy.
"""

import argparse
import os
import json
import logging
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import load_data, Dataset
from ppo_discreate import Actor, Critic, PPO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main evaluation function."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Available GPUs: {n_gpu}")
    
    # Load test data
    test_examples = load_data(args.test_data_path, args, max_size=args.max_test_data_size)
    test_dataset = Dataset(test_examples)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    
    # Setup model registry or legacy mode
    if args.use_legacy_mode:
        logger.info("Using legacy mode with hardcoded models")
        from environment import LegacyEnvironment
        num_models = 11
        env_class = LegacyEnvironment
        model_registry = None
    else:
        logger.info(f"Loading model registry from {args.model_config}")
        from adapters.registry import ModelRegistry
        from environment import Environment
        
        model_registry = ModelRegistry.from_config(args.model_config)
        num_models = model_registry.get_num_models()
        env_class = Environment
        logger.info(f"Loaded {num_models} models: {model_registry.list_models()}")
    
    # Initialize actor and critic networks
    actor = Actor(
        config_path=args.actor_model,
        num_classes=num_models,
        mlp_hidden_size=256
    ).to(device)
    
    critic = Critic(args.critic_model, 256).to(device)
    
    # Load trained weights
    logger.info(f"Loading actor from {args.actor_model_bin}")
    actor.load_state_dict(torch.load(args.actor_model_bin, map_location=device))
    
    logger.info(f"Loading critic from {args.critic_model_bin}")
    critic.load_state_dict(torch.load(args.critic_model_bin, map_location=device))
    
    actor.eval()
    critic.eval()
    
    ppo_agent = PPO(actor, critic)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_trajectories = []
    
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        sources = batch['source']
        
        trajectorys = []
        i = 0
        batch_size = min(args.batch_size, len(input_ids))
        
        while i < batch_size:
            thread_args = []
            threads_to_create = min(args.thread_nums, batch_size - i)
            
            for j in range(threads_to_create):
                # Create environment
                if args.use_legacy_mode:
                    env = env_class(actor, None)  # No reward calculator for eval
                else:
                    env = env_class(actor, None, model_registry)
                
                args_tuple = (
                    env,
                    trajectorys,
                    ppo_agent,
                    input_ids[i + j],
                    attention_masks[i + j],
                    sources[i + j],
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
    
    # Print summary statistics
    if all_trajectories:
        avg_steps = np.mean([len(t.get('actions', [])) for t in all_trajectories])
        logger.info(f"Average steps per question: {avg_steps:.2f}")


def worker(thread_id, collect_args):
    """Worker function for parallel evaluation."""
    try:
        collect(*collect_args)
        logger.debug(f"Thread {thread_id} completed")
    except Exception as e:
        logger.error(f"Thread {thread_id} failed: {e}")


def collect(environment, trajectorys, agent, inputs_id, attention_mask, source):
    """Collect evaluation trajectory."""
    answers_outs = []
    trajectory = {}
    
    # Reset environment (pass None for target since we're evaluating)
    environment.reset(inputs_id, attention_mask, source, target=None)
    
    step = 0
    max_steps = 4
    
    while True:
        # Get action using evaluation mode (deterministic)
        action, action_pro = agent.get_action_eval(
            environment.state.unsqueeze(0),
            environment.attention_masks_e.unsqueeze(0)
        )
        
        # Take action
        answers, _, stop = environment.step(action.item() if hasattr(action, 'item') else action)
        answers_outs.append(answers)
        
        # Check stopping condition
        if stop == 1 or step >= max_steps:
            break
        
        step += 1
    
    # Build trajectory
    trajectory["question"] = source
    trajectory['answers'] = answers_outs
    trajectory['final_answer'] = answers_outs[-1] if answers_outs else ""
    
    # Convert action sequence to list
    action_list = []
    for a in environment.action_sequence:
        if hasattr(a, 'item'):
            action_list.append(a.item())
        else:
            action_list.append(int(a))
    
    trajectory['actions'] = action_list
    trajectory['num_steps'] = len(action_list)
    
    logger.debug(f"Actions: {action_list}")
    trajectorys.append(trajectory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DER Evaluation - Dynamic Ensemble Reasoning for LLMs")
    
    # Model config
    parser.add_argument("--actor_model", type=str, default="./opt-125m",
                        help="Path to actor model config")
    parser.add_argument("--actor_model_bin", type=str, default="./checkpoints/actor_epoch_9.bin",
                        help="Path to trained actor weights")
    parser.add_argument("--critic_model", type=str, default="./opt-125m",
                        help="Path to critic model config")
    parser.add_argument("--critic_model_bin", type=str, default="./checkpoints/critic_epoch_9.bin",
                        help="Path to trained critic weights")
    
    # Mode
    parser.add_argument("--use_legacy_mode", action="store_true",
                        help="Use legacy mode with hardcoded models")
    parser.add_argument("--model_config", type=str, default="./config/models.yaml",
                        help="Path to model configuration YAML file")
    
    # Data config
    parser.add_argument("--test_data_path", type=str,
                        default="../datasets/test_data_prepared.jsonl")
    
    # Evaluation hyperparameters
    parser.add_argument("--max_test_data_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--thread_nums", type=int, default=10)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:3")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="./eval_outputs/")
    parser.add_argument("--output_filename", type=str, default="evaluation_results.jsonl")
    
    args = parser.parse_args()
    main(args)
