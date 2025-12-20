# schemes/__init__.py
"""
Schemes Registry

Available optimization schemes for LLM workflow optimization.
"""

from .base import BaseScheme
from .beam import BeamInnerLoopEngine, Candidate
# from .ecwo import ECWOScheme
from .veto import VETOScheme
from .aflow import AFlowScheme

# --------------------------------------------------
# Scheme Registry
# --------------------------------------------------

SCHEME_REGISTRY = {
    # "ecwo": ECWOScheme,
    "veto": VETOScheme,
    "aflow": AFlowScheme,  # Fixed typo: was "alfow"
}


def get_scheme(name: str, args) -> BaseScheme:
    """
    Get a scheme instance by name.
    
    Args:
        name: Scheme name (ecwo, veto, aflow)
        args: Configuration arguments
        
    Returns:
        Instantiated scheme
        
    Raises:
        ValueError: If scheme name not found
    """
    name_lower = name.lower()
    if name_lower not in SCHEME_REGISTRY:
        available = ", ".join(SCHEME_REGISTRY.keys())
        raise ValueError(f"Unknown scheme '{name}'. Available: {available}")
    
    return SCHEME_REGISTRY[name_lower](args)


def list_schemes() -> list:
    """List available scheme names."""
    return list(SCHEME_REGISTRY.keys())


__all__ = [
    "BaseScheme",
    "BeamInnerLoopEngine",
    "Candidate",
    # "ECWOScheme",
    "VETOScheme",
    "AFlowScheme",
    "SCHEME_REGISTRY",
    "get_scheme",
    "list_schemes",
]