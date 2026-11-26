"""Development commands for SignScript."""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "start": "src/main.py",
    "start-dynamic": "src/dynamic/live_dynamic.py",
    "prepare-static": "src/static/prepare_data.py",
    "train-static": "src/static/train.py",
    "eval-static": "src/static/evaluate.py",
    "prepare-dynamic": "src/dynamic/prepare_dynamic.py",
    "train-dynamic": "src/dynamic/train_dynamic.py",
    "eval-dynamic": "src/dynamic/eval_dynamic.py",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python dev.py <command> [options]")
        print("Available commands:", ", ".join(COMMANDS))
        return

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print("Available commands:", ", ".join(COMMANDS))
        sys.exit(1)

    script = PROJECT_ROOT / COMMANDS[command]

    if not script.exists():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")

    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [src_path, env.get("PYTHONPATH")],
        )
    )

    subprocess.run(
        [sys.executable, str(script), *sys.argv[2:]],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

if __name__ == "__main__":
    main()
