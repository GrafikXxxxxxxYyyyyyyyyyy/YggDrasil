#!/usr/bin/env python3
"""Run Diffusers vs YggDrasil compare for SDXL. Delegates to shared script."""
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "compare_diffusers_yggdrasil.py"
sys.exit(subprocess.run([sys.executable, str(_SCRIPT), "--model", "sdxl"] + sys.argv[1:]).returncode)
