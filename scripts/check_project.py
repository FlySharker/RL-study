"""Check example syntax and optionally CLI entry points without training."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SHARED_MODULES = {"sb3_ppo_mujoco_common.py"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="Run each example with --help.")
    args = parser.parse_args()
    sources = sorted((ROOT / "examples").rglob("*.py"))
    sources += sorted((ROOT / "scripts").glob("*.py"))
    failures = []
    for source in sources:
        try:
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"{source.relative_to(ROOT)}: {exc}")
    print(f"Syntax checked: {len(sources)} Python files", flush=True)

    if args.cli:
        for source in sources:
            if "examples" not in source.relative_to(ROOT).parts or source.name in SHARED_MODULES:
                continue
            try:
                result = subprocess.run(
                    [sys.executable, str(source), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                if result.returncode:
                    failures.append(f"{source.relative_to(ROOT)}: {result.stderr or result.stdout}")
                print(f"{'OK' if result.returncode == 0 else 'FAIL'} {source.relative_to(ROOT)}", flush=True)
            except subprocess.TimeoutExpired:
                failures.append(f"{source.relative_to(ROOT)}: --help timed out after 60s")
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"Result: {len(failures)} failure(s)")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
