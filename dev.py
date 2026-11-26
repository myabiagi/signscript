import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "start": "src/main.py",
    "prepare-alphabet": "src/alphabet/prepare_alphabet.py",
    "train-alphabet": "src/alphabet/train_alphabet.py",
    "train-random-forest": "src/alphabet/train_random_forest.py",
    "train-combined": "src/alphabet/train_combined.py",
    "train-geometric": "src/alphabet/train_geometric.py",
    "prepare-pugeault": "src/alphabet/prepare_pugeault.py",
    "train-pugeault": "src/alphabet/train_pugeault.py",
    "test-final": "src/alphabet/test_final.py",
}

def main():
    # Check that a valid command was provided
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python dev.py <command>")
        print("Available commands:", ", ".join(COMMANDS))
        return

    # Get the script corresponding to the command
    script = PROJECT_ROOT / COMMANDS[sys.argv[1]]

    # Include src so scripts in subfolders can import shared modules
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")

    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [src_path, env.get("PYTHONPATH")])
    )

    # Run the script using the current Python environment
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True
    )

if __name__ == "__main__":
    main()
