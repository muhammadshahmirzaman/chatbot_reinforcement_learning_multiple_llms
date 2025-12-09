"""
Parquet to JSONL Converter for DER Training Pipeline

This script converts parquet files (train and test) into JSONL format
required by the DER (Dynamic Ensemble Reasoning) system.

The DER system expects JSONL files with the following structure per line:
{
    "id": "<unique_identifier>",
    "instruction": "<task instruction>",
    "input": "<question or query>",
    "output": "<target answer>"
}

Usage:
    python convert_parquet_to_jsonl.py
    
    Optional arguments:
        --train_input: Path to train parquet file (default: train-00000-of-00001.parquet)
        --test_input: Path to test parquet file (default: test-00000-of-00001.parquet)
        --train_output: Path for train JSONL output (default: train_data_prepared.jsonl)
        --test_output: Path for test JSONL output (default: test_data_prepared.jsonl)
        --instruction: Custom instruction text (default: "Solve the following math problem step by step.")
"""

import pandas as pd
import json
import argparse
import os
from pathlib import Path


def convert_parquet_to_jsonl(
    input_parquet_path: str,
    output_jsonl_path: str,
    instruction: str,
    id_prefix: str = ""
) -> int:
    """
    Convert a parquet file to JSONL format for DER training.
    
    Args:
        input_parquet_path: Path to the input parquet file
        output_jsonl_path: Path for the output JSONL file
        instruction: The instruction text to use for all examples
        id_prefix: Prefix for the ID (e.g., 'train_' or 'test_')
    
    Returns:
        Number of records converted
    """
    print(f"Loading parquet file: {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    
    print(f"Found {len(df)} records with columns: {list(df.columns)}")
    
    # Validate required columns exist
    if 'question' not in df.columns:
        raise ValueError(f"Expected 'question' column in parquet file, got: {list(df.columns)}")
    if 'answer' not in df.columns:
        raise ValueError(f"Expected 'answer' column in parquet file, got: {list(df.columns)}")
    
    converted_count = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Create the DER-compatible format
            record = {
                "id": f"{id_prefix}{idx}",
                "instruction": instruction,
                "input": str(row['question']).strip(),
                "output": str(row['answer']).strip()
            }
            
            # Write as JSON line
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            converted_count += 1
    
    print(f"Successfully converted {converted_count} records to: {output_jsonl_path}")
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to JSONL format for DER training"
    )
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    parser.add_argument(
        "--train_input",
        type=str,
        default=str(script_dir / "train-00000-of-00001.parquet"),
        help="Path to the training parquet file"
    )
    parser.add_argument(
        "--test_input",
        type=str,
        default=str(script_dir / "test-00000-of-00001.parquet"),
        help="Path to the test parquet file"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default=str(script_dir / "train_data_prepared.jsonl"),
        help="Output path for training JSONL file"
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default=str(script_dir / "test_data_prepared.jsonl"),
        help="Output path for test JSONL file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Solve the following math problem step by step.",
        help="Instruction text to use for all examples"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip converting training data"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip converting test data"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Parquet to JSONL Converter for DER")
    print("=" * 60)
    
    total_records = 0
    
    # Convert training data
    if not args.skip_train:
        if os.path.exists(args.train_input):
            print(f"\n[1/2] Converting training data...")
            count = convert_parquet_to_jsonl(
                input_parquet_path=args.train_input,
                output_jsonl_path=args.train_output,
                instruction=args.instruction,
                id_prefix="train_"
            )
            total_records += count
        else:
            print(f"\n[1/2] Training file not found: {args.train_input}")
    else:
        print("\n[1/2] Skipping training data conversion")
    
    # Convert test data
    if not args.skip_test:
        if os.path.exists(args.test_input):
            print(f"\n[2/2] Converting test data...")
            count = convert_parquet_to_jsonl(
                input_parquet_path=args.test_input,
                output_jsonl_path=args.test_output,
                instruction=args.instruction,
                id_prefix="test_"
            )
            total_records += count
        else:
            print(f"\n[2/2] Test file not found: {args.test_input}")
    else:
        print("\n[2/2] Skipping test data conversion")
    
    print("\n" + "=" * 60)
    print(f"Conversion complete! Total records converted: {total_records}")
    print("=" * 60)
    
    # Show sample output
    if not args.skip_train and os.path.exists(args.train_output):
        print("\nSample output (first record from training data):")
        print("-" * 40)
        with open(args.train_output, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
