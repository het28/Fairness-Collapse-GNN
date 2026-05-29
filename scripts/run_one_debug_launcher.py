#!/usr/bin/env python3
"""
Launcher for run_one under the debugger. Avoids path resolution through /AppleInternal
by changing to project root (from this script's location) before importing the module.
Usage: python3 scripts/run_one_debug_launcher.py [same args as run_one]
"""
import os
import sys

# Project root = parent of directory containing this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
os.chdir(_project_root)
sys.path.insert(0, _project_root)

# Run the module with remaining args
from uncertainty_eval.run_one import run_one, _cli

if __name__ == "__main__":
    _cli()
