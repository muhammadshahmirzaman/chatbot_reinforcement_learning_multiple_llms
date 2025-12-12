"""Diagnostic script to test DER training step-by-step."""
import sys
import os
import torch

# Setup
os.chdir('c:/Users/Default/Desktop/multiple_llms/chatbot_reinforcement_learning_multiple_llms/DER')
sys.path.insert(0, '.')

print("=" * 60)
print("DER Training Diagnostic")
print("=" * 60)

# Test 1: Load data
print("\n1. Loading data...")
from utils.data import load_data, Dataset

try:
    examples = load_data("../datasets/train_data_prepared.jsonl", type('obj', (), {'max_seq_len': 512})(), max_size=3)
    print(f"✓ Loaded {len(examples)} examples")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Load model registry
print("\n2. Loading model registry...")
from adapters.registry import ModelRegistry

try:
    registry = ModelRegistry.from_config("./config/models_test.yaml")
    print(f"✓ Loaded {registry.get_num_models()} models")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Initialize components
print("\n3. Initializing components...")
device = torch.device("cpu")

try:
    from ppo_discreate import Actor, Critic
    from reward import KTP_calculate
    
    actor = Actor("./opt-125m", num_classes=registry.get_num_models(), mlp_hidden_size=256).to(device)
    print(f"✓ Actor initialized on {device}")
    
    reward_calc = KTP_calculate(device=str(device))
    reward_calc.load_checkpoint()
    print(f"✓ Reward calculator initialized")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create environment
print("\n4. Creating environment...")
try:
    from environment import Environment
    
    env = Environment(
        actor, 
        reward_calc, 
        registry,
        state_device=str(device),
        actor_device=str(device)
    )
    print(f"✓ Environment created with {env.num_models} models")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test a single step
print("\n5. Testing environment step...")
try:
    # Prepare a sample
    dataset = Dataset(examples[:1])
    sample = dataset[0]
    
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    source = sample['source']
    target = sample['target']
    
    print(f"  Source: {source[:50 ]}...")
    print(f"  Target: {target[:50]}...")
    
    # Reset environment
    env.reset(input_ids, attention_mask, source, target)
    print("  ✓ Environment reset")
    
    # Take action
    action = 0  # Select first model
    answer, score, stop = env.step(action)
    
    print(f"  ✓ Step completed")
    print(f"    Answer: {answer[:80]}...")
    print(f"    Score: {score}")
    print(f"    Stop: {stop}")
    
except Exception as e:
    print(f"✗ Error during step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL DIAGNOSTIC TESTS PASSED!")
print("=" * 60)
print("\nThe DER system is ready for training.")
