"""
DER PPO Multi-LLM Training Script

This script trains a PPO agent to dynamically select the best LLM
from a pool of models for answering questions.

Supports both:
- Config-driven training with adapter registry (new)
- Legacy mode with hardcoded models (backward compatible)
"""

import argparse
import os
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils.Utils import seed_everything, str2bool
from utils.data import load_data, Dataset
from ppo_discreate import Actor, Critic, PPO
from reward import Reward_calculate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main training function."""
    seed_everything(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Available GPUs: {n_gpu}")
    
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Load training data
    train_dataset = None
    if args.do_train:
        train_examples = load_data(args.train_data_path, args, max_size=args.max_train_data_size)
        train_dataset = Dataset(train_examples)
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    
    # Setup model registry or legacy mode
    if args.use_legacy_mode:
        logger.info("Using legacy mode with hardcoded models")
        from environment import LegacyEnvironment
        num_models = 11  # Original 11 models
        env_class = LegacyEnvironment
        model_registry = None
        model_costs = [0.007, 0.013, 0.013, 0.012, 0.013, 0.007, 0.007, 0.012, 0.011, 0.016, 0.006]
    else:
        logger.info(f"Loading model registry from {args.model_config}")
        from adapters.registry import ModelRegistry
        from environment import Environment
        
        model_registry = ModelRegistry.from_config(args.model_config)
        num_models = model_registry.get_num_models()
        env_class = Environment
        model_costs = [model_registry.get_model_cost(name) for name in model_registry.list_models()]
        
        logger.info(f"Loaded {num_models} models: {model_registry.list_models()}")
    
    # Initialize actor and critic networks
    actor = Actor(
        config_path=args.actor_model,
        num_classes=num_models,
        mlp_hidden_size=256
    ).to(device)
    
    critic = Critic(args.critic_model, 256).to(device)
    
    # Initialize PPO agent
    ppo_agent = PPO(actor, critic)
    
    # Initialize reward calculator
    reward_calculator = Reward_calculate()
    reward_calculator.load_checkpoint()
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Training loop
    for epoch in range(args.epochs):
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            sources = batch['source']
            targets = batch['target']
            
            trajectorys = []
            i = 0
            
            while i < min(args.batch_size, len(input_ids)):
                thread_args = []
                
                for j in range(min(args.thread_nums, len(input_ids) - i)):
                    # Create environment for this thread
                    if args.use_legacy_mode:
                        env = env_class(actor, reward_calculator)
                    else:
                        env = env_class(actor, reward_calculator, model_registry)
                    
                    args_tuple = (
                        env,
                        trajectorys,
                        ppo_agent,
                        input_ids[i + j],
                        attention_masks[i + j],
                        sources[i + j],
                        targets[i + j],
                        model_costs  # Pass model costs for reward shaping
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
            
            logger.debug(f"Collected {len(trajectorys)} trajectories")
            
            if len(trajectorys) < 4:
                logger.warning("Not enough trajectories collected, skipping PPO update.")
                continue
            
            # Log rewards and update
            rewards_mean, last_score_mean, score_mean = log_rewards(trajectorys)
            
            if not args.disable_wandb:
                wandb.log({
                    "rewards": rewards_mean,
                    "last_score": last_score_mean,
                    "score": score_mean,
                    "epoch": epoch
                })
            
            ppo_agent.update(trajectorys, args.per_batch_size, args.sample_size, device, wandb)
            save_model(actor, critic, epoch, args.save_dir)
    
    logger.info("Training complete!")


def worker(thread_id, collect_args):
    """Worker function for parallel trajectory collection."""
    try:
        collect(*collect_args)
        logger.debug(f"Thread {thread_id} completed")
    except Exception as e:
        logger.error(f"Thread {thread_id} failed: {e}")


def collect(environment, trajectorys, agent, inputs_id, attention_mask, source, target, model_costs):
    """Collect a trajectory by interacting with the environment."""
    rewards = []
    scores = []
    action_pros = []
    states = []
    old_values = []
    attention_masks_outs = []
    trajectory = {}
    
    # Reset environment
    environment.reset(inputs_id, attention_mask, source, target)
    states.append(inputs_id)
    attention_masks_outs.append(attention_mask)
    value = agent.get_value(inputs_id.unsqueeze(0), attention_mask.unsqueeze(0))
    old_values.append(value)
    
    step = 0
    max_steps = 4
    
    while True:
        # Get action from agent
        action, action_pro = agent.get_action(
            environment.state.unsqueeze(0),
            environment.attention_masks_e.unsqueeze(0)
        )
        action_pros.append(action_pro)
        
        # Take action in environment
        answers, score, stop = environment.step(action)
        
        # Calculate reward with model cost penalty
        action_idx = action.item() if hasattr(action, 'item') else action
        cost_penalty = model_costs[action_idx] if action_idx < len(model_costs) else 0.01
        
        if step > 0:
            if score[0] > scores[-1][0]:
                reward = score[0] - cost_penalty + 5 * (score[0] - scores[-1][0])
            else:
                reward = score[0] - cost_penalty - 5 * (scores[-1][0] - score[0])
        else:
            reward = score[0] - cost_penalty
        
        scores.append(score)
        
        # Check termination conditions
        if stop == 1 or step == max_steps:
            if stop == 1 and step <= max_steps:
                reward = reward + 2.0  # Bonus for early success
            elif stop == 0 and step == max_steps:
                reward = reward - 2.0  # Penalty for timeout
            rewards.append([reward])
            break
        else:
            rewards.append([reward])
            states.append(environment.state)
            attention_masks_outs.append(environment.attention_masks_e)
            value = agent.get_value(
                environment.state.unsqueeze(0),
                environment.attention_masks_e.unsqueeze(0)
            )
            old_values.append(value)
            step += 1
    
    # Build trajectory
    trajectory['states'] = states
    trajectory['rewards'] = rewards
    trajectory['scores'] = scores
    trajectory['old_action_pros'] = action_pros
    trajectory['old_values'] = old_values
    trajectory['actions'] = environment.action_sequence
    trajectory['attention_masks'] = attention_masks_outs
    
    logger.debug(f"Actions: {environment.action_sequence}, Scores: {scores}")
    trajectorys.append(trajectory)


def save_model(actor, critic, epoch, save_dir):
    """Save actor and critic checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    
    actor_save_path = os.path.join(save_dir, f"actor_epoch_{epoch}.bin")
    torch.save(actor.state_dict(), actor_save_path)
    
    critic_save_path = os.path.join(save_dir, f"critic_epoch_{epoch}.bin")
    torch.save(critic.state_dict(), critic_save_path)
    
    logger.info(f"Saved models at epoch {epoch}")


def log_rewards(trajectorys):
    """Calculate and log reward statistics."""
    sum_mean = 0
    scores_mean = 0
    sum_last = 0
    total = 0
    
    for traj in trajectorys:
        score = np.array(traj['scores'])
        scores_mean += np.mean(score)
        sum_last += traj['scores'][-1][0]
        reward = np.array(traj['rewards'])
        sum_mean += np.mean(reward)
        total += 1
    
    return sum_mean / total, sum_last / total, scores_mean / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DER Training - Dynamic Ensemble Reasoning for LLMs")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model config
    parser.add_argument("--actor_model", type=str, default="./opt-125m",
                        help="Path to actor model")
    parser.add_argument("--critic_model", type=str, default="./opt-125m",
                        help="Path to critic model")
    
    # Training mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--use_legacy_mode", type=str2bool, default=False,
                        help="Use legacy mode with hardcoded models (for backward compatibility)")
    
    # Model registry config
    parser.add_argument("--model_config", type=str, default="./config/models.yaml",
                        help="Path to model configuration YAML file")
    
    # Data config
    parser.add_argument("--train_data_path", type=str, 
                        default="../datasets/train_data_prepared.jsonl")
    
    # Training hyperparameters
    parser.add_argument("--max_train_data_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--per_batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--thread_nums", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    
    # Device config
    parser.add_argument("--device", type=str, default="cuda:3",
                        help="Device for training")
    
    # Wandb config
    parser.add_argument("--disable_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="DER-Training")
    parser.add_argument("--wandb_name", type=str, default="multi-llm-ppo")
    
    # Save path
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    main(args)
