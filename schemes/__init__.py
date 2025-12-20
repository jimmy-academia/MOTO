# schemes/__init__.py
"""
Scheme Registry

Available schemes:
- ecwo: Edge-Cloud Workflow Optimization (test-time adaptation)
- aflow: AFlow - Agentic Workflow Generation (ICLR 2025)
"""

from .ecwo import ECWOScheme
from .aflow import AFlowScheme

SCHEME_REGISTRY = {
    'ecwo': ECWOScheme,
    'aflow': AFlowScheme,
}


def get_scheme(scheme_name: str, args):
    """
    Get a scheme instance by name.
    
    Args:
        scheme_name: Name of the scheme (e.g., 'ecwo', 'aflow')
        args: Arguments namespace to pass to scheme constructor
        
    Returns:
        Instantiated scheme object
        
    Raises:
        ValueError: If scheme_name is not in registry
    """
    if scheme_name not in SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown scheme: {scheme_name}. "
            f"Available: {list(SCHEME_REGISTRY.keys())}"
        )
    
    return SCHEME_REGISTRY[scheme_name](args)