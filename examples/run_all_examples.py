#!/usr/bin/env python3
"""
Simple script to run all examples in the PyEtSimul project.
    python tmp/run_all_examples.py          # uses current Python interpreter
    python tmp/run_all_examples.py --use-uv # uses 'uv run python'

"""

import subprocess
import sys
from pathlib import Path

# Check if --use-uv flag is passed
use_uv = "--use-uv" in sys.argv
if use_uv:
    sys.argv.remove("--use-uv")

# Get project root directory
project_root = Path(__file__).parent.parent
examples_dir = project_root / "examples"

# List all Python example files, excluding this script and any __pycache__
example_files = []

# Add examples from main examples directory
main_examples = sorted([f for f in examples_dir.glob("*.py") if f.name != "run_all_examples.py"])
example_files.extend(main_examples)

# Add examples from experiments subdirectory
experiments_dir = examples_dir / "experiments"
if experiments_dir.exists():
    experiment_examples = sorted([f for f in experiments_dir.glob("*.py") if not f.name.startswith("config")])
    example_files.extend(experiment_examples)

print(f"Found {len(example_files)} example files:")
for file in example_files:
    # Show relative path from examples directory
    relative_path = file.relative_to(examples_dir)
    print(f"  - {relative_path}")

print("\n" + "=" * 60)
print("Running all examples...")
print("=" * 60)

failed_examples = []
successful_examples = []

for example_file in example_files:
    relative_path = example_file.relative_to(examples_dir)
    print(f"\n🔄 Running: {relative_path}")
    print("-" * 40)

    cmd = (
            ["uv", "run", "python", str(example_file)]
            if use_uv
            else [sys.executable, str(example_file)]
        )

    try:
        result = subprocess.run(
            cmd, cwd=project_root, capture_output=False, timeout=60
        )

        if result.returncode == 0:
            print(f"✅ {relative_path} completed successfully")
            successful_examples.append(str(relative_path))
        else:
            print(f"❌ {relative_path} failed with exit code {result.returncode}")
            failed_examples.append(str(relative_path))

    except subprocess.TimeoutExpired:
        print(f"⏱️  {relative_path} timed out after 60 seconds")
        failed_examples.append(f"{relative_path} (timeout)")
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user during {relative_path}")
        break
    except Exception as e:
        print(f"💥 {relative_path} crashed: {e}")
        failed_examples.append(f"{relative_path} (crashed)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Successful: {len(successful_examples)}")
print(f"❌ Failed: {len(failed_examples)}")

if successful_examples:
    print("\nSuccessful examples:")
    for name in successful_examples:
        print(f"  ✅ {name}")

if failed_examples:
    print("\nFailed examples:")
    for name in failed_examples:
        print(f"  ❌ {name}")

print(f"\nTotal examples processed: {len(successful_examples) + len(failed_examples)}")
