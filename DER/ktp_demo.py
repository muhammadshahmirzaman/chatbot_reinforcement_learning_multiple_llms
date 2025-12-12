#!/usr/bin/env python
"""
Interactive KTP Demo - See how KTP works with manual input

This script demonstrates the KTP mechanism interactively,
allowing you to see the reward and feedback generation process.
"""

import sys
sys.path.insert(0, '.')

from reward import KTP_calculate

def interactive_ktp_demo():
    """Interactive demo of KTP mechanism."""
    print("="*60)
    print("KTP (Knowledge Transfer Prompt) Interactive Demo")
    print("="*60)
    print("\nThis shows how KTP evaluates answers automatically.")
    print("In real training, this happens without human input!\n")
    
    # Initialize KTP
    ktp = KTP_calculate(device='cpu')
    ktp.load_checkpoint()
    
    print("✓ KTP loaded and ready\n")
    
    while True:
        print("-" * 60)
        
        # Get inputs
        question = input("\nEnter a question (or 'quit' to exit): ").strip()
        if question.lower() == 'quit':
            break
        
        generated_answer = input("Enter LLM's generated answer: ").strip()
        ground_truth = input("Enter the ground truth answer: ").strip()
        
        # Process through KTP
        print("\n" + "="*60)
        print("KTP PROCESSING...")
        print("="*60)
        
        reward, next_state = ktp.process(question, generated_answer, ground_truth)
        
        # Display results
        print(f"\n📊 REWARD SCORE: {reward:.4f}")
        print(f"   (Higher is better, max = 1.0)")
        
        print(f"\n📝 NEXT STATE (with KTP feedback):")
        print(f"   {next_state}")
        
        print("\n" + "="*60)
        print("INTERPRETATION:")
        print("="*60)
        if reward > 0.8:
            print("✅ Excellent! Generated answer is very similar to ground truth")
        elif reward > 0.6:
            print("👍 Good! Generated answer is reasonably close")
        elif reward > 0.4:
            print("⚠️  Fair. Generated answer needs improvement")
        else:
            print("❌ Poor. Generated answer is quite different from ground truth")
        
        print("\nThe 'Next State' includes feedback that would guide")
        print("the next LLM iteration in actual training.")
        print()

def example_demo():
    """Run with example data."""
    print("="*60)
    print("KTP Example Demo (Automated)")
    print("="*60)
    
    ktp = KTP_calculate(device='cpu')
    ktp.load_checkpoint()
    
    examples = [
        {
            "question": "What is the capital of France?",
            "generated": "Paris is the capital of France.",
            "ground_truth": "The capital of France is Paris."
        },
        {
            "question": "What is 2+2?",
            "generated": "Four.",
            "ground_truth": "2+2 equals 4."
        },
        {
            "question": "Explain machine learning",
            "generated": "ML uses algorithms.",
            "ground_truth": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}")
        print('='*60)
        print(f"Question: {ex['question']}")
        print(f"Generated: {ex['generated']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        
        reward, next_state = ktp.process(
            ex['question'], 
            ex['generated'], 
            ex['ground_truth']
        )
        
        print(f"\n→ Reward: {reward:.4f}")
        print(f"→ Next State: {next_state[:100]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("KTP Interactive Demo")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode (ask for your input)")
    parser.add_argument("--example", action="store_true",
                       help="Run with example data")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_ktp_demo()
    elif args.example:
        example_demo()
    else:
        print("Usage:")
        print("  python ktp_demo.py --interactive   # Manual input mode")
        print("  python ktp_demo.py --example       # Run examples")
