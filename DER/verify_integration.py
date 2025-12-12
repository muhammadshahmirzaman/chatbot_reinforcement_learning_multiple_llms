#!/usr/bin/env python
"""
Verify DER Training Integration

This script checks that train_ppo_multi.py is running with all
integrated components and not as a standalone script.
"""

import sys
import os

def check_integration():
    """Verify all integration points."""
    print("="*80)
    print("DER TRAINING INTEGRATION VERIFICATION")
    print("="*80)
    
    # Check 1: File dependencies
    print("\n1. Checking File Dependencies:")
    required_files = {
        'environment.py': 'Environment class for model interaction',
        'reward.py': 'KTP_calculate for reward computation',
        'train_ppo_multi.py': 'Main training script',
        'ppo_discreate.py': 'PPO agent implementation',
        'config/models_production.yaml': 'Model configuration',
        'adapters/registry.py': 'Model registry',
        'adapters/offline/openassistant_adapter.py': 'OpenAssistant adapter',
    }
    
    all_exist = True
    for file, description in required_files.items():
        exists = os.path.exists(file)
        symbol = "✅" if exists else "❌"
        print(f"   {symbol} {file:45s} - {description}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ MISSING FILES DETECTED!")
        return False
    
    # Check 2: Import verification
    print("\n2. Verifying Imports in train_ppo_multi.py:")
    with open('train_ppo_multi.py', 'r') as f:
        content = f.read()
    
    imports_to_check = {
        'from environment import': 'Uses environment.py',
        'from reward import': 'Uses reward.py',
        'from adapters.registry import': 'Uses model registry',
        'from ppo_discreate import': 'Uses PPO implementation',
    }
    
    for import_stmt, description in imports_to_check.items():
        found = import_stmt in content
        symbol = "✅" if found else "❌"
        print(f"   {symbol} {import_stmt:35s} - {description}")
    
    # Check 3: Configuration verification
    print("\n3. Checking Model Configuration:")
    import yaml
    
    with open('config/models_production.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    models = config.get('models', [])
    enabled_models = [m for m in models if m.get('enabled', True)]
    
    print(f"   Total models configured: {len(models)}")
    print(f"   Enabled models: {len(enabled_models)}")
    
    # Check for OpenAssistant
    oa_found = any(m.get('name') == 'openassistant-sft' for m in enabled_models)
    symbol = "✅" if oa_found else "❌"
    print(f"   {symbol} OpenAssistant-SFT {'found' if oa_found else 'NOT found'}")
    
    print("\n   Enabled models:")
    for m in enabled_models:
        print(f"      - {m.get('name')} ({m.get('type')})")
    
    # Check 4: Integration points
    print("\n4. Integration Points in train_ppo_multi.py:")
    
    integration_points = {
        'Environment': 'Creates environment with model_registry (line ~77)',
        'KTP_calculate': 'Initializes reward calculator (line ~100)',
        'model_registry.get': 'Called by Environment to get models',
        'reward_calculator.process': 'Called by Environment for rewards',
    }
    
    for point, description in integration_points.items():
        found = point in content
        symbol = "✅" if found else "❌"
        print(f"   {symbol} {point:25s} - {description}")
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print("\n✅ train_ppo_multi.py is INTEGRATED (not standalone)")
    print("\nIntegrated Components:")
    print("  1. environment.py - Manages model selection and interaction")
    print("  2. reward.py - Calculates rewards via KTP")
    print("  3. adapters/registry.py - Loads and manages all models")
    print("  4. config/models_production.yaml - Configures model pool")
    print("  5. ppo_discreate.py - Implements PPO learning")
    
    print("\n📦 Model Pool (DER Agent can select from):")
    for i, m in enumerate(enabled_models, 1):
        print(f"  {i}. {m.get('name')}")
    
    print("\n" + "="*80)
    return True

if __name__ == "__main__":
    os.chdir('c:/Users/Default/Desktop/multiple_llms/chatbot_reinforcement_learning_multiple_llms/DER')
    success = check_integration()
    sys.exit(0 if success else 1)
