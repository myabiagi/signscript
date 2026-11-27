"""Development commands for SignScript.

Primary product path: python dev.py start

Layout:
  src/   — runtime (web API, models, shared helpers)
  lab/   — prepare / train / eval / collect
  web/   — browser UI
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    # Product (browser demo)
    "start": "src/web_api.py",
    # Static train / eval
    "collect-static": "lab/static/collect_static.py",
    "prepare-static": "lab/static/prepare_data.py",
    "train-static": "lab/static/train.py",
    "eval-static": "lab/static/evaluate.py",
    # Dynamic train / eval
    "prepare-dynamic": "lab/dynamic/prepare_dynamic.py",
    "train-dynamic": "lab/dynamic/train_dynamic.py",
    "eval-dynamic": "lab/dynamic/eval_dynamic.py",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python dev.py <command> [options]")
        print("Primary: python dev.py start")
        print("Available commands:", ", ".join(COMMANDS))
        return

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print("Available commands:", ", ".join(COMMANDS))
        sys.exit(1)

    script = PROJECT_ROOT / COMMANDS[command]

    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [src_path, env.get("PYTHONPATH")])
    )

    subprocess.run(
        [sys.executable, str(script), *sys.argv[2:]],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
