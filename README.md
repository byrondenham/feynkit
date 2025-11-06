# Feynkit

**A comprehensive toolkit for symbolic Feynman integral computations**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Feynkit is a modern Python package for symbolic manipulation and analysis of Feynman integrals using parametric representations. It provides a clean, well-documented API for working with:

- **Feynman graph structures** with full validation
- **Symanzik polynomials** (U and F)
- **Multiple parametrisations** (Schwinger, Feynman, Lee-Pomeransky)
- **GKZ hypergeometric systems** and differential equations
- **Toric ideal computations** for integration-by-parts identities
- **Visualisation tools** for Newton polytopes and geometric structures
- **LaTeX export** for publication-ready documents

## Installation

### From Source (Development)

```bash
git clone https://github.com/byrondenham/feynkit.git
cd feynkit
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install feynkit
```

## Quick Start

```python
import sympy as sp
from feynkit import Edge, Graph
from feynkit.polynomials import calculate_symanzik_polynomials
from feynkit.parametrisations import create_parametrisations
from feynkit.kinematics import create_momentum_products

# Define a simple bubble diagram
m1, m2 = sp.symbols('m1 m2', nonnegative=True)
nu1, nu2 = sp.symbols('nu1 nu2', positive=True)

# Internal propagators
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)

# External legs
ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

# Build graph
graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

# Compute Symanzik polynomials
D = sp.Symbol('D', positive=True)
nus = {1: nu1, 2: nu2}
momentum_products = create_momentum_products(n_external=2, use_mandelstam=True)

U, F = calculate_symanzik_polynomials(graph, momentum_products)
print(f"U = {U}")
print(f"F = {F}")

# Create all parametrisations
parametrisations = create_parametrisations(
    graph=graph,
    dimension=D,
    loop_count=1,
    propagator_exponents=nus,
    momentum_products=momentum_products
)

# Access individual parametrisations
schwinger = parametrisations.schwinger
feynman = parametrisations.feynman
lee_pomeransky = parametrisations.lee_pomeransky
```

## Features

### Modular Architecture

Feynkit is designed with a clean separation of concerns:

- **`feynkit.core`**: Graph data structures and validation
- **`feynkit.polynomials`**: Symanzik polynomial computation
- **`feynkit.parametrisations`**: Schwinger, Feynman, and Lee-Pomeransky representations
- **`feynkit.systems`**: GKZ hypergeometric systems and Euler operators
- **`feynkit.algebra`**: Toric ideals and Gröbner basis computations
- **`feynkit.kinematics`**: Momentum products and Mandelstam variables
- **`feynkit.visualisation`**: TikZ and geometric plotting tools
- **`feynkit.io`**: LaTeX and text export utilities

### Type-Safe and Well-Documented

Every function includes:
- Complete type hints for IDE support
- Comprehensive docstrings with mathematical formulas
- Usage examples
- References to the literature

### Thoroughly Tested

- Comprehensive test suite with pytest
- High code coverage
- Validated against known results from the literature

## Documentation

Full documentation is available at [feynkit.readthedocs.io](https://feynkit.readthedocs.io) (coming soon).

For now, see the `examples/` directory for Jupyter notebooks demonstrating common use cases.

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/byrondenham/feynkit.git
cd feynkit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=feynkit --cov-report=html

# Run specific test file
pytest tests/core/test_graph.py
```

### Code Quality

```bash
# Format code with black
black feynkit tests

# Lint with ruff
ruff check feynkit tests

# Type check with mypy
mypy feynkit
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Feynkit is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use Feynkit in your research, please cite:

```bibtex
@software{feynkit2024,
  author = {Denham, Byron},
  title = {Feynkit: A Toolkit for Feynman Integral Computations},
  year = {2024},
  url = {https://github.com/byrondenham/feynkit}
}
```

## References

Feynkit implements methods from:

- Weinzierl, S. (2022). "Feynman Integrals: A Comprehensive Treatment for Students and Researchers." Springer.
- de la Cruz, L. (2019). "Feynman integrals as A-hypergeometric functions." JHEP 12, 123.
- Lee, R.N. and Pomeransky, A.A. (2013). "Critical points and master integrals." JHEP 11, 165.
- Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994). "Discriminants, Resultants and Multidimensional Determinants." Birkhäuser.

## Acknowledgments

This project builds upon the foundations laid by the original `fp` package, with significant refactoring for improved modularity, maintainability, and usability.
