# Feynkit

**A comprehensive toolkit for symbolic Feynman integral computations**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Feynkit is a comprehensive Python toolkit for symbolic computation of Feynman integrals using modern algebraic geometry and differential equations techniques.

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
