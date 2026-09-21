"""
Kinematics module for Feynman integral computations.

Provides tools for working with momentum products, Mandelstam variables,
and kinematic constraints.

Functions
---------
create_momentum_products
    Create momentum dot product dictionary for external legs.
get_momentum_product
    Retrieve a momentum product with automatic ordering.
validate_momentum_products_complete
    Validate that all required momentum products are present.
create_mandelstam_variables
    Create Mandelstam variable symbols.
mandelstam_constraints
    Generate kinematic constraints for Mandelstam variables.
KinematicInvariants
    Dataclass holding the standard independent invariants for n external legs.
standard_invariants
    Express the external dot products in the standard planar invariants.

Examples
--------
>>> from feynkit.kinematics import create_momentum_products
>>>
>>> # Create momentum products for 3 external legs
>>> p_dot = create_momentum_products(n_external=3, use_mandelstam=True)
>>> print(p_dot)
{(1, 2): -p1^2/2 - p2^2/2 + p3^2/2, (1, 3): -p1^2/2 + p2^2/2 - p3^2/2, (2, 3): p1^2/2 - p2^2/2 - p3^2/2}
"""

from .mandelstam import (
    KinematicInvariants,
    create_mandelstam_variables,
    mandelstam_constraints,
    standard_invariants,
)
from .momentum import (
    create_momentum_products,
    get_momentum_product,
    validate_momentum_products_complete,
)

__all__ = [
    "create_momentum_products",
    "get_momentum_product",
    "validate_momentum_products_complete",
    "create_mandelstam_variables",
    "mandelstam_constraints",
    "KinematicInvariants",
    "standard_invariants",
]
