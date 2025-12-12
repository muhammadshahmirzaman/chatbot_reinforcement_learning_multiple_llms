#!/usr/bin/env python
"""
Production Training Script for DER with Comprehensive Statistics

This script runs DER training with all available local models and provides
detailed statistics including GPU/CPU usage, performance metrics, and model status.
"""

import subprocess
import sys
import argparse
import time
import os
import json
from datetime import datetime

def get_system_stats():
    """Collect system statistics."""
    stats = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gpu': {},
        'cpu': {},
        'memory': {}
    }
    
    # GPU stats
    try:
        import torch
        if torch.cuda.is_available():
            stats['gpu']['available'] = True
            stats['gpu']['device_count'] = torch.cuda.device_count()
            stats['gpu']['current_device'] = torch.cuda.current_device()
            stats['gpu']['device_name'] = torch.cuda.get_device_name(0)
            
            # Memory stats
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
        
        # Memory stats
        mem = psutil.virtual_memory()
        stats['memory']['total_gb'] = mem.total / 1024**3
        stats['memory']['used_gb'] = mem.used / 1024**3
        stats['memory']['percent'] = mem.percent
    except ImportError:
        stats['cpu']['percent'] = 'N/A (install psutil)'
    
    return stats

def check_model_status(config_path="./config/models_production.yaml"):
    """Check status of all configured models."""
    model_status = {}
    
    try:
        sys.path.insert(0, '.')
        from adapters.registry import ModelRegistry
        from adapters.base import GenerationConfig
        
        registry = ModelRegistry.from_config(config_path)
        models = registry.list_models()
        
        for model_name in models:
            try:
                adapter = registry.get(model_name)
                # Quick test - don't actually load
                model_status[model_name] = {
                    'configured': True,
                    'type': adapter.__class__.__name__,
                    'status': 'Ready'
                }
            except Exception as e:
                model_status[model_name] = {
                    'configured': True,
                    'type': 'Unknown',
                    'status': f'Error: {str(e)[:50]}'
                }
    except Exception as e:
        model_status['ERROR'] = {'status': str(e)}
    
    return model_status

def print_statistics_dashboard(stats_before, stats_after, training_time, model_status, training_success):
    """Print comprehensive statistics dashboard."""
    print("\n" + "="*80)
    print(" " * 25 + "TRAINING STATISTICS DASHBOARD")
    print("="*80)
    
    # Training Summary
    print("\n📊 TRAINING SUMMARY")
    print("-" * 80)
    print(f"  Status: {'✅ SUCCESS' if training_success else '❌ FAILED'}")
    print(f"  Duration: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  Started: {stats_before['timestamp']}")
    print(f"  Ended: {stats_after['timestamp']}")
    
    # GPU Statistics
    print("\n🎮 GPU STATISTICS")
    print("-" * 80)
    if stats_after['gpu'].get('available'):
        gpu = stats_after['gpu']
        print(f"  Device: {gpu.get('device_name', 'Unknown')}")
        print(f"  Memory Allocated: {gpu.get('memory_allocated_gb', 0):.2f} GB")
        print(f"  Memory Reserved: {gpu.get('memory_reserved_gb', 0):.2f} GB")
        print(f"  Peak Memory Usage: {gpu.get('max_memory_allocated_gb', 0):.2f} GB")
        print(f"  Utilization: {(gpu.get('memory_allocated_gb', 0) / 24 * 100):.1f}% (assuming 24GB GPU)")
    else:
        print("  Status: ❌ No GPU available (training on CPU)")
    
    # CPU & Memory Statistics
    print("\n💻 CPU & MEMORY STATISTICS")
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
    print("\n🤖 MODEL STATUS")
    print("-" * 80)
    if model_status:
        running_count = 0
        error_count = 0
        
        for model_name, status in model_status.items():
            if 'Error' in status.get('status', ''):
                symbol = "❌"
                error_count += 1
            elif status.get('status') == 'Ready':
                symbol = "✅"
                running_count += 1
            else:
                symbol = "⚠️"
            
            model_type = status.get('type', 'Unknown')
            model_status_text = status.get('status', 'Unknown')
            print(f"  {symbol} {model_name:20s} [{model_type:20s}] {model_status_text}")
        
        print(f"\n  Total Models: {len(model_status)}")
        print(f"  Ready: {running_count} | Errors: {error_count}")
    else:
        print("  ⚠️ No model status available")
    
    # Performance Metrics
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 80)
    
    # Check for checkpoints
    checkpoint_dir = "./checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin')]
        print(f"  Checkpoints Created: {len(checkpoints)}")
        if checkpoints:
            print(f"  Latest Checkpoint: {sorted(checkpoints)[-1]}")
    else:
        print("  Checkpoints Created: 0 (directory not found)")
    
    # Try to read training logs if available
    if os.path.exists("training_log.json"):
        try:
            with open("training_log.json", 'r') as f:
                log = json.load(f)
            print(f"  Epochs Completed: {log.get('epochs_completed', 'N/A')}")
            print(f"  Trajectories Collected: {log.get('trajectories', 'N/A')}")
            print(f"  Average Reward: {log.get('avg_reward', 'N/A')}")
        except:
            pass
    
    print("\n" + "="*80)
    print(" " * 28 + "END OF STATISTICS")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser("DER Production Training")
    parser.add_argument("--setup", action="store_true",
                       help="First download and verify models before training")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run a quick test with minimal data")
    parser.add_argument("--full-training", action="store_true",
                       help="Run full production training")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Collect statistics before training
    print("\n📊 Collecting system statistics...")
    stats_before = get_system_stats()
    model_status = check_model_status()
    
    if args.setup:
        print("="*80)
        print("STEP 1: DOWNLOADING MODELS")
        print("="*80)
        result = subprocess.run([
            sys.executable, "setup_production_models.py",
            "--download", "--verify"
        ])
        if result.returncode != 0:
            print("\n✗ Model setup failed!")
            print_statistics_dashboard(stats_before, get_system_stats(), 0, model_status, False)
            return 1
        
        print("\n✓ Models ready!")
    
    training_success = False
    training_time = 0
    
    if args.quick_test:
        print("\n" + "="*80)
        print("RUNNING QUICK TEST")
        print("="*80)
        cmd = [
            sys.executable, "train_ppo_multi.py",
            "--model_config", "./config/models_production.yaml",
            "--train_data_path", "./test_data_mini.jsonl",
            "--epochs", "1",
            "--batch_size", "2",
            "--max_train_data_size", "5",
            "--device", args.device,
            "--thread_nums", "2"
        ]
        
    elif args.full_training:
        print("\n" + "="*80)
        print("RUNNING FULL TRAINING")
        print("="*80)
        cmd = [
            sys.executable, "train_ppo_multi.py",
            "--model_config", "./config/models_production.yaml",
            "--epochs", str(args.epochs),
            "--device", args.device
        ]
    else:
        print("Usage:")
        print("  # Setup models first time:")
        print("  python train_production.py --setup")
        print("")
        print("  # Quick test:")
        print("  python train_production.py --quick-test --device cpu")
        print("")
        print("  # Full training:")
        print("  python train_production.py --full-training --epochs 20 --device cuda:0")
        print("")
        print("  # Setup + test:")
        print("  python train_production.py --setup --quick-test")
        
        # Still show stats
        print_statistics_dashboard(stats_before, get_system_stats(), 0, model_status, False)
        return 0
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run training and measure time
    start_time = time.time()
    result = subprocess.run(cmd)
    training_time = time.time() - start_time
    
    training_success = (result.returncode == 0)
    
    # Collect statistics after training
    stats_after = get_system_stats()
    model_status_after = check_model_status()
    
    # Print comprehensive statistics dashboard
    print_statistics_dashboard(
        stats_before,
        stats_after,
        training_time,
        model_status_after,
        training_success
    )
    
    if training_success:
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nCheckpoints saved in: ./checkpoints/")
    else:
        print("\n" + "="*80)
        print("✗ TRAINING FAILED")
        print("="*80)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

