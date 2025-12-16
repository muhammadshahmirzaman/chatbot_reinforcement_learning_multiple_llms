#!/usr/bin/env python
"""
Lightweight verifier for the train_production.py workflow on CPU.

What it does:
- Optionally cleans the checkpoints directory for a fresh run.
- Runs `train_production.py --quick-test --device cpu` to keep runtime short.
- Validates that training finishes successfully and that checkpoint files are created.
- Prints a concise report with paths to the generated artifacts.

Usage:
    python verify_train_production_cpu.py
    python verify_train_production_cpu.py --keep-checkpoints
    python verify_train_production_cpu.py --train-script train_production.py --device cpu
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser("Verify train_production CPU quick-test")
    parser.add_argument("--train-script", type=str, default="train_production.py",
                        help="Path to train_production.py (relative to DER/ unless absolute)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for the verification run (default: cpu)")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Do not delete existing checkpoints before the run")
    parser.add_argument("--workdir", type=str, default="",
                        help="Repository root containing the DER folder. "
                             "If empty, auto-detect based on this script location.")
    args = parser.parse_args()

    # Auto-detect repo root (parent of this file's directory) if not provided.
    if args.workdir:
        repo_root = Path(args.workdir).resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent

    der_dir = repo_root / "DER"

    if "chat" not in str(sys.executable):
        print(f"WARNING: current python executable is {sys.executable}. Activate the 'chat' venv first (./chat/Scripts/Activate.ps1).")

    # Resolve train script path
    train_script_arg = Path(args.train_script)
    script_path = train_script_arg if train_script_arg.is_absolute() else (der_dir / train_script_arg)

    checkpoints_dir = der_dir / "checkpoints"

    if not script_path.exists():
        print(f"✗ train script not found: {script_path}")
        return 1

    if not args.keep_checkpoints and checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        print(f"✓ Removed existing checkpoints at {checkpoints_dir}")

    cmd = [
        sys.executable,
        str(script_path),
        "--quick-test",
        "--device",
        args.device,
    ]

    print(f"▶ Running: {' '.join(cmd)} (cwd={der_dir})")
    code, out, err = run_cmd(cmd, cwd=der_dir)

    if code != 0:
        print("✗ Training failed (non-zero exit code)")
        print("---- stdout ----")
        print(out)
        print("---- stderr ----")
        print(err)
        return code

    created = checkpoints_dir.exists() and any(checkpoints_dir.glob("*.bin"))
    checkpoint_list = sorted(str(p.name) for p in checkpoints_dir.glob("*.bin")) if checkpoints_dir.exists() else []

    print("---- stdout ----")
    print(out)
    print("---- stderr ----")
    print(err)

    if created:
        print(f"✓ Checkpoints present at {checkpoints_dir}")
        print(f"  Files: {checkpoint_list}")
        print("✓ Verification succeeded.")
        return 0
    else:
        print("✗ No checkpoints were created.")
        return 2


if __name__ == "__main__":
    sys.exit(main())

