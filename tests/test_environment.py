"""The test session and every process it starts stay off the GPU (see conftest.py): JAX would
otherwise pre-allocate most of the GPU memory once per process, which can freeze a machine whose
GPU shares memory with the CPU (DGX Spark / GB10)."""

import os
import subprocess
import sys

import pytest

ON_GPU = os.environ.get("GPLAN_TESTS_ON_GPU") == "1"
CHECK = ("import jax, torch; print(jax.default_backend(), torch.cuda.is_available(), "
         "__import__('os').environ['XLA_PYTHON_CLIENT_PREALLOCATE'])")


@pytest.mark.skipif(ON_GPU, reason="GPLAN_TESTS_ON_GPU=1")
def test_this_process_and_subprocesses_run_on_cpu():
    import jax
    import torch
    assert jax.default_backend() == "cpu" and not torch.cuda.is_available()
    out = subprocess.run([sys.executable, "-c", CHECK], env=dict(os.environ), capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["cpu", "False", "false"]


def test_jax_never_preallocates():
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
