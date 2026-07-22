# Contributing to catgen

Thank you for your interest in contributing to **catgen**! We welcome contributions in the form of bug reports, feature requests, documentation improvements, and code patches.

## Quick Start

### Reporting Issues

Found a bug or have a feature request? Please [open an issue on GitHub](https://github.com/Italbytz/pypi-catgen/issues) with:

- Clear description of the problem
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Python version, OS, and catgen version

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/Italbytz/pypi-catgen.git
cd pypi-catgen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_simulation.py -v

# Run with coverage
pytest tests/ --cov=catgen --cov-report=html
```

### Writing Code

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code following existing style:**
   - Use docstrings for all public functions and classes
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
   - Keep functions focused and testable

3. **Add tests for new functionality:**
   - Test normal cases, edge cases, and error conditions
   - Ensure test names are descriptive (e.g., `test_dnf_multiterm_generation`)
   - Aim for >80% coverage on new code

4. **Update documentation:**
   - Docstrings in code (Google-style format)
   - README.md if introducing new public API
   - CHANGELOG.md (add to `[Unreleased]` section)

### Example: Adding a New Dataset Generator

**Dataset generators** in catgen should follow this pattern:
- Return `(X, y)` tuples with NumPy arrays
- Accept `random_state` for reproducibility
- Located in `src/catgen/datasets/` submodules

For **SNP simulation**, use existing high-level wrappers (`generate_snp_glm_dataset()`).
If you need low-level simulation metadata (interaction descriptions, MAF, probabilities),
use `simulate_snp_glm()` directly from `catgen.simulation.snp` module.

1. **Implement in appropriate submodule** (e.g., `src/catgen/datasets/boolean_concepts.py`):
   ```python
   def generate_my_dataset(n_obs: int, n_features: int, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
       """Generate custom dataset.
       
       Parameters
       ----------
       n_obs : int
           Number of observations.
       n_features : int
           Number of features.
       random_state : int, optional
           Random seed for reproducibility.
       
       Returns
       -------
       X : np.ndarray, shape (n_obs, n_features)
           Feature matrix.
       y : np.ndarray, shape (n_obs,)
           Target vector (binary).
       """
       # ... implementation ...
       return X, y
   ```

2. **Export from subpackage** (`src/catgen/datasets/__init__.py`):
   ```python
   from catgen.datasets.boolean_concepts import generate_my_dataset
   __all__ = [..., "generate_my_dataset"]
   ```

3. **Export from root** (`src/catgen/__init__.py`):
   ```python
   from catgen.datasets import generate_my_dataset
   __all__ = [..., "generate_my_dataset"]
   ```

4. **Add tests** (`tests/test_datasets.py` or similar):
   ```python
   def test_my_dataset_shape():
       X, y = generate_my_dataset(n_obs=100, n_features=10, random_state=42)
       assert X.shape == (100, 10)
       assert y.shape == (100,)
       assert np.all(np.isin(y, [0, 1]))
   ```

5. **Update CHANGELOG.md** (under `[Unreleased]`):
   ```markdown
   - `generate_my_dataset()`: Custom dataset generator with detailed description
   ```

### Submitting a Pull Request

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a pull request on GitHub with:
   - Descriptive title
   - Summary of changes
   - Reference to any related issues (#123)

3. Ensure CI passes (tests, linting, etc.)

4. Respond to review feedback; maintainers will merge once approved

## Code Style

- **Format:** PEP 8
- **Docstrings:** Google-style (see existing functions)
- **Type hints:** Encouraged for function signatures
- **Imports:** Sort with standard library first, then third-party, then local

**Example:**
```python
"""Concise one-liner description.

Longer explanation if needed (e.g., background, use case, references).

Parameters
----------
param1 : str
    Description.
param2 : int, optional
    Description. Default is 10.

Returns
-------
result : np.ndarray
    Description of output.

Examples
--------
>>> X, y = generate_my_dataset(n_obs=100, n_features=5)
>>> X.shape
(100, 5)
"""
```

## Release Workflow (Maintainers)

1. Update CHANGELOG.md: move `[Unreleased]` content to new version section
2. Bump version in `pyproject.toml` following [semver](https://semver.org/)
3. Update `__version__` in `src/catgen/__init__.py`
4. Create git tag: `git tag v0.1.1`
5. Build: `python3 -m build`
6. Upload: `twine upload dist/*`
7. Push tag: `git push origin v0.1.1`

## Questions?

- Open an issue on [GitHub](https://github.com/Italbytz/pypi-catgen/issues)
- Check existing [issues](https://github.com/Italbytz/pypi-catgen/issues?q=is%3Aissue) and [discussions](https://github.com/Italbytz/pypi-catgen/discussions)

---

**Thank you for contributing to catgen!** 🎉
