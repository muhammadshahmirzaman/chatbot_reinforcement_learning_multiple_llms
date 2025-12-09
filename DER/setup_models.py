"""
DER Model Setup Script

This script downloads all required pre-trained models for the DER 
(Dynamic Ensemble Reasoning) system.

Required Models:
1. opt-125m - Facebook's OPT model for tokenization and actor/critic networks
2. bart-large-cnn - For BART scoring (reward calculation)
3. bert-base-multilingual-cased - For BERT scoring (reward calculation)

Optional (commented out - requires additional setup):
4. bleurt-base-128 - For BLEURT scoring

Usage:
    python setup_models.py
"""

import os
import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("=" * 60)
    print("DER Model Setup Script")
    print("=" * 60)
    
    # Ensure transformers is installed
    try:
        from transformers import AutoTokenizer, AutoModel, OPTModel
        print("[✓] transformers is installed")
    except ImportError:
        print("[!] Installing transformers...")
        install_package("transformers")
        from transformers import AutoTokenizer, AutoModel, OPTModel
    
    # Ensure torch is installed
    try:
        import torch
        print(f"[✓] PyTorch is installed (version: {torch.__version__})")
    except ImportError:
        print("[!] Installing PyTorch...")
        install_package("torch")
        import torch
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    models_to_download = [
        {
            "name": "opt-125m",
            "hf_name": "facebook/opt-125m",
            "local_path": os.path.join(script_dir, "opt-125m"),
            "description": "OPT-125M for tokenization and actor/critic networks"
        },
        {
            "name": "bart-large-cnn",
            "hf_name": "facebook/bart-large-cnn",
            "local_path": os.path.join(script_dir, "bart-large-cnn"),
            "description": "BART-Large-CNN for BART scoring"
        },
        {
            "name": "bert-base-multilingual-cased",
            "hf_name": "bert-base-multilingual-cased",
            "local_path": os.path.join(script_dir, "bert-base-multilingual-cased"),
            "description": "BERT for BERTScore calculation"
        }
    ]
    
    print("\n" + "-" * 60)
    print("Downloading models to:", script_dir)
    print("-" * 60)
    
    for i, model_info in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] {model_info['description']}")
        print(f"    HuggingFace: {model_info['hf_name']}")
        print(f"    Local path:  {model_info['local_path']}")
        
        if os.path.exists(model_info['local_path']):
            print(f"    [✓] Already exists, skipping...")
            continue
        
        try:
            print(f"    [↓] Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_info['hf_name'])
            tokenizer.save_pretrained(model_info['local_path'])
            
            print(f"    [↓] Downloading model...")
            if "opt" in model_info['name']:
                model = OPTModel.from_pretrained(model_info['hf_name'])
            else:
                model = AutoModel.from_pretrained(model_info['hf_name'])
            model.save_pretrained(model_info['local_path'])
            
            print(f"    [✓] Successfully downloaded!")
        except Exception as e:
            print(f"    [✗] Error: {e}")
    
    # Create bart_score directory and placeholder
    bart_score_dir = os.path.join(script_dir, "bart_score")
    if not os.path.exists(bart_score_dir):
        os.makedirs(bart_score_dir)
        print(f"\n[!] Created directory: {bart_score_dir}")
        print("    Note: You may need to download bart_score.pth separately")
        print("    The BARTScorer will be initialized from the bart-large-cnn model")
    
    # Create bleurt directory placeholder
    bleurt_dir = os.path.join(script_dir, "bleurt")
    if not os.path.exists(bleurt_dir):
        os.makedirs(bleurt_dir)
        print(f"\n[!] Created directory: {bleurt_dir}")
        print("    Note: BLEURT requires separate installation:")
        print("    pip install bleurt @ https://github.com/google-research/bleurt.git")
        print("    And download bleurt-base-128 checkpoint")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("\nModel Directory Structure:")
    print(f"  {script_dir}/")
    print("  ├── opt-125m/              (Actor/Critic + Tokenizer)")
    print("  ├── bart-large-cnn/        (BART Scoring)")
    print("  ├── bert-base-multilingual-cased/  (BERT Scoring)")
    print("  ├── bart_score/            (BART Score weights)")
    print("  └── bleurt/                (BLEURT checkpoint)")
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_minimal.txt")
    print("2. Run training: python train_ppo_multi.py")
    print("3. Run testing: python test_ppo_multi.py")


if __name__ == "__main__":
    main()
