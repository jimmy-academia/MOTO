from .clover import CloverScheme
from .ecwo import ECWOScheme

SCHEME_REGISTRY = {
    'clover': CloverScheme,
    'ecwo': ECWOScheme,
}

def get_scheme(scheme_name, args):
    if scheme_name not in SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme: {scheme_name}. Available: {list(SCHEME_REGISTRY.keys())}")
    
    return SCHEME_REGISTRY[scheme_name](args)