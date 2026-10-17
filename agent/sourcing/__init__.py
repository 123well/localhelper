"""Model sourcing: candidate registry + download/convert recipes.

The agent ships without weights (see README). This module encodes the
decision matrix from the spec: a frozen 7B SSM (Mamba-2/RWKV-6 class),
INT4, ONNX for XDNA NPU. Facts are NOT expected in weights — only
reasoning/instruction-following; everything factual comes via RAG.
"""
from .candidates import CANDIDATES, pick_candidate, recipe_for

__all__ = ["CANDIDATES", "pick_candidate", "recipe_for"]
