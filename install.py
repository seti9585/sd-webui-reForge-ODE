"""Install torchdiffeq only when it is not already available."""

import importlib.util

if importlib.util.find_spec("torchdiffeq") is None:
    from launch import run_pip
    run_pip("install torchdiffeq", "torchdiffeq for reForge-ODE Sampler")
