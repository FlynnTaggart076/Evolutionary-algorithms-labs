"""Run the four complete experimental series and rebuild their reports."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    print("\nRunning: " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-only", action="store_true", help="Use existing results and only rebuild PDFs")
    parser.add_argument("--render", action="store_true", help="Also render all PDF pages to tools/_qa")
    args = parser.parse_args()
    if not args.reports_only:
        run("-m", "pytest", "Lab1", "Lab2", "Lab3", "Lab4", "tools", "-q")
        for entry in ["Lab1/run.py", "Lab2/main.py", "Lab3/main.py", "Lab4/experiment.py"]:
            run(entry)
    run("tools/build_reports.py", *(["--render"] if args.render else []))
    run("tools/verify_results.py", "--output", "verification.json")


if __name__ == "__main__":
    main()
