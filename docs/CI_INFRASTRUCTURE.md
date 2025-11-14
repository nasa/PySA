# Continuous Integration Infrastructure

This document describes the CI/CD infrastructure for the PySA project.

## Overview

The PySA project uses GitHub Actions for continuous integration and continuous deployment. The CI system ensures code quality, functionality, and compatibility across multiple Python versions.

## Workflows

### Main CI Workflow (`ci.yml`)

The primary workflow that orchestrates all quality checks. It runs on every push to `main` and `ci-testing` branches, and on all pull requests.

**Jobs:**
- **Lint**: Checks code formatting using YAPF
- **Test**: Runs unit tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- **Coverage**: Uploads coverage reports to Codecov
- **Examples**: Validates example scripts work correctly
- **Tutorials**: Executes Jupyter notebook tutorials
- **All Checks Passed**: Final gate that ensures all previous jobs succeeded

**Features:**
- Parallel execution of test matrix
- Dependency caching for faster builds
- Coverage report generation
- Artifact upload for tutorial outputs
- Concurrency control to cancel outdated runs

### Code Coverage Workflow (`coverage.yml`)

Dedicated workflow for generating and uploading code coverage reports.

**Features:**
- Generates XML, HTML, and terminal coverage reports
- Uploads to Codecov for tracking over time
- Stores HTML reports as artifacts (30-day retention)
- Adds coverage summary to GitHub workflow summary

### Individual Component Workflows

#### Python PyTest (`python-pytest.yml`)
- Runs unit tests across all supported Python versions
- Generates coverage reports
- Uses pytest with parallel execution (`pytest-xdist`)

#### Python Code Formatting (`python-yapf.yml`)
- Validates code follows Google style guide
- Uses YAPF formatter version 0.32.0
- Runs on Python 3.11

#### Python Examples (`python-example.yml`)
- Tests example scripts to ensure they run without errors
- Tests on Python 3.9 and 3.11
- Includes example_ising.py, example_qubo.py, and example_ais.py

#### Python Tutorials (`python-tutorials.yml`)
- Executes Jupyter notebooks using Papermill
- Tests on Python 3.9 and 3.11
- Uploads notebook outputs as artifacts
- Includes: example_ising.ipynb, hpo_demo.ipynb, ising_tutorial.ipynb, RBM_tutorial.ipynb

### PR Summary Comment Workflow (`pr-comment.yml`)

Automatically posts a summary comment on pull requests with CI results.

**Features:**
- Triggered when CI workflow completes
- Posts or updates a summary comment on the PR
- Includes status emoji, conclusion, and link to full workflow run
- Helps reviewers quickly see if CI passed

## Supported Python Versions

The CI system tests against the following Python versions:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

**Note:** Python 3.7 support was dropped as it reached end-of-life in June 2023.

## Dependencies

### Core Dependencies
- numpy
- scipy
- numba
- pandas
- tqdm
- more_itertools

### CI-Specific Dependencies
- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-xdist: Parallel test execution
- yapf: Code formatting

### Tutorial Dependencies
- papermill: Notebook execution
- matplotlib: Plotting
- plotly: Interactive visualizations
- jupyter: Notebook support
- hyperopt: Hyperparameter optimization
- torch & torchvision: Deep learning (for RBM tutorial)

### Example Dependencies
- termplotlib: Terminal plotting

## Caching Strategy

All workflows use pip caching to speed up dependency installation:
```yaml
- uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: 'pip'
```

This caches the pip packages based on `requirements.txt`, significantly reducing build times.

## Coverage Reporting

Coverage reports are:
1. Generated using `pytest-cov`
2. Uploaded to Codecov for historical tracking
3. Stored as artifacts in GitHub Actions
4. Displayed in GitHub workflow summaries

To view coverage locally:
```bash
pytest --cov=pysa --cov-report=html tests/
open htmlcov/index.html
```

## Artifact Retention

- **Coverage Reports**: 7 days
- **Tutorial Outputs**: 7 days
- **HTML Coverage Reports**: 30 days

## Running Tests Locally

### Unit Tests
```bash
# Install test dependencies
pip install -e . pytest pytest-cov pytest-xdist

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=pysa --cov-report=term tests/

# Run in parallel
pytest -n auto tests/
```

### Code Formatting
```bash
# Install yapf
pip install yapf==0.32.0

# Check formatting
yapf --style=google -d -r .

# Auto-fix formatting
yapf --style=google -i -r .
```

### Examples
```bash
# Install dependencies
pip install -e . termplotlib

# Run examples
python examples/example_ising.py
python examples/example_qubo.py
python examples/example_ais.py
```

### Tutorials
```bash
# Install dependencies
pip install -e . papermill matplotlib plotly jupyter hyperopt torch torchvision

# Execute notebooks
cd tutorials/
papermill example_ising.ipynb output.ipynb
```

## Troubleshooting

### Test Failures

1. **Import Errors**: Ensure all dependencies are installed with `pip install -e .`
2. **Numba Compilation**: Numba may take time on first run; this is normal
3. **Random Test Failures**: Some tests use random sampling; check if failures are consistent

### Coverage Issues

- Coverage may be lower on certain Python versions due to conditional code
- Ensure tests actually execute the code you expect them to cover
- Use `pytest --cov-report=html` to see line-by-line coverage

### Formatting Issues

- Run `yapf --style=google -i -r .` to auto-fix formatting
- Common issues: line length (80 chars), indentation, spacing

### Tutorial Failures

- Tutorials may fail if external dependencies (torch, etc.) aren't properly installed
- Some tutorials require significant compute time; consider using `|| true` for optional tutorials
- Check that input data files are present in the tutorials directory

## Best Practices

1. **Before Pushing**: Run tests locally to catch issues early
2. **Format Code**: Run yapf before committing
3. **Update Tests**: Add tests for new features
4. **Check Coverage**: Aim for >80% coverage on new code
5. **Monitor CI**: Check CI results on your PRs
6. **Review Artifacts**: Download and review tutorial outputs if changes affect notebooks

## Maintenance

### Updating GitHub Actions

Periodically update action versions:
```yaml
- uses: actions/checkout@v4  # Check for v5, etc.
- uses: actions/setup-python@v5  # Check for newer versions
```

### Updating Dependencies

1. Update `requirements.txt`
2. Test locally with new versions
3. Update CI workflows if needed
4. Run full CI pipeline to ensure compatibility

### Adding New Python Versions

When a new Python version is released:
1. Add to test matrix in workflows
2. Test locally first
3. Update README badges
4. Update setup.py classifiers

## Security

### Dependency Scanning

Consider adding:
- Dependabot for automated dependency updates
- CodeQL for security scanning
- pip-audit for vulnerability checking

### Secrets Management

- Never commit secrets to the repository
- Use GitHub Secrets for sensitive data
- Use environment variables in CI workflows

## Contributing

When contributing to PySA:
1. Ensure all CI checks pass
2. Add tests for new functionality
3. Update documentation as needed
4. Follow the code style (enforced by yapf)
5. Check that examples and tutorials still work

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
- [YAPF Documentation](https://github.com/google/yapf)

## Contact

For CI/CD issues, please:
1. Check this documentation first
2. Review GitHub Actions logs
3. Open an issue with:
   - Workflow run link
   - Error messages
   - Steps to reproduce
