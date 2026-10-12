"""The command-line scripts, run as on the cluster: each in a fresh Python process."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_python(code, **env):
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}", **env}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def test_importing_the_training_script_does_not_start_jax():
    """XLA reads XLA_FLAGS when its backend starts, so `train.py --deterministic` only works if no import
    (gplan.jax.* included) creates a JAX array before main() sets the flag."""
    out = run_python(
        "import runpy; from jax._src import xla_bridge as xb; "
        f"runpy.run_path({str(REPO / 'scripts' / 'train.py')!r}, run_name='not_main'); "
        "print('started' if xb.backends_are_initialized() else 'idle')")
    assert out.strip().endswith("idle")
