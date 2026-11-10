"""Representation-extraction entry points.

Imports are intentionally lazy so that lightweight SAE utilities can be used
without requiring the full Transformers/Diffusers stack at import time.
"""


def run_llm_extraction(config):
    from .llm import run_llm_extraction as _run
    return _run(config)


def run_sd15_extraction(config):
    from .sd15 import run_sd15_extraction as _run
    return _run(config)


def run_flux2_extraction(config):
    from .flux2 import run_flux2_extraction as _run
    return _run(config)


__all__ = ["run_llm_extraction", "run_sd15_extraction", "run_flux2_extraction"]
