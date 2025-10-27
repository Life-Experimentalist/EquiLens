# Contributing to EquiLens

Thank you for your interest in contributing to EquiLens! This document provides guidelines and instructions for contributing.

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (3.13 recommended)
- [UV package manager](https://github.com/astral-sh/uv)
- Docker Desktop (for container testing)
- Git

### Development Setup

```powershell
# Fork and clone repository
git clone https://github.com/YOUR_USERNAME/EquiLens.git
cd EquiLens

# Add upstream remote
git remote add upstream https://github.com/Life-Experimentalist/EquiLens.git

# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run pytest
```

## 📝 Development Workflow

### 1. Create a Branch

```powershell
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clear, concise commit messages
- Follow existing code style
- Add tests for new features
- Update documentation as needed

### 3. Code Quality

```powershell
# Format code
uv run ruff format .

# Check for issues
uv run ruff check --fix .

# Run type checking
uv run mypy src/equilens

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_specific.py -v
```

### 4. Commit Changes

```powershell
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new bias category detection"

# Or
git commit -m "fix: resolve Docker networking issue"
```

**Commit Message Format:**

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 5. Push and Create PR

```powershell
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Go to: https://github.com/Life-Experimentalist/EquiLens/compare
```

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/unit/test_feature.py
import pytest
from equilens.core.feature import YourFeature

def test_feature_functionality():
    """Test basic functionality"""
    feature = YourFeature()
    result = feature.process()
    assert result is not None

def test_feature_edge_case():
    """Test edge cases"""
    feature = YourFeature()
    with pytest.raises(ValueError):
        feature.process(invalid_input)
```

### Running Tests

```powershell
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=equilens --cov-report=html

# Run specific markers
uv run pytest -m "not slow"

# Run integration tests
uv run pytest tests/integration/
```

## 📚 Documentation

### Updating Documentation

- **Code comments**: Use clear, concise docstrings
- **README.md**: Update for new features
- **docs/**: Add detailed guides for major features
- **CHANGELOG.md**: Document all changes

### Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description of function.

    Longer description explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is invalid

    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

## 🎨 Code Style

EquiLens follows:
- **PEP 8** for Python code style
- **Type hints** for all functions
- **Ruff** for linting and formatting
- **Black-compatible** line length (88 chars)

### Style Checklist

- [ ] Type hints on all functions
- [ ] Docstrings for public functions
- [ ] No unused imports
- [ ] Descriptive variable names
- [ ] Comments for complex logic
- [ ] Tests for new functionality

## 🐛 Reporting Bugs

### Before Submitting

1. Check [existing issues](https://github.com/Life-Experimentalist/EquiLens/issues)
2. Verify bug in latest version
3. Collect relevant information:
   - OS and version
   - Python version
   - EquiLens version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log output.

**Environment:**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.5]
- EquiLens version: [e.g., 2.0.0]
- Docker version: [e.g., 24.0.0]

**Additional context**
Any other relevant information.
```

## 💡 Feature Requests

### Suggesting Features

1. Check [existing feature requests](https://github.com/Life-Experimentalist/EquiLens/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
2. Open new issue with **enhancement** label
3. Describe:
   - Use case
   - Proposed solution
   - Alternatives considered
   - Additional context

## 🔧 Pull Request Process

### Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Added/updated tests for changes
- [ ] Updated documentation
- [ ] Updated CHANGELOG.md
- [ ] Commits follow commit message format
- [ ] Branch is up to date with main

### PR Checklist

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Documentation
- [ ] Updated README.md
- [ ] Updated relevant docs/
- [ ] Added/updated docstrings
- [ ] Updated CHANGELOG.md

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Notes
Any other relevant information.
```

### Review Process

1. Maintainer reviews code
2. CI/CD checks pass
3. Requested changes addressed
4. Final approval
5. Merge to main

## 🏗️ Project Structure

```
EquiLens/
├── src/equilens/          # Main package
│   ├── core/              # Core functionality
│   ├── cli.py             # CLI interface
│   ├── web_ui.py          # Gradio web UI
│   └── gradio_ui.py       # Alternative UI
├── src/Phase1_CorpusGenerator/  # Corpus generation
├── src/Phase2_ModelAuditor/     # Model auditing
├── src/Phase3_Analysis/         # Analytics
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── public/                # Web assets
```

## 🎯 Development Focus Areas

We're especially interested in contributions for:

1. **New Bias Categories** - Additional bias detection types
2. **Model Support** - Integration with more LLM platforms
3. **Visualizations** - Enhanced analytics charts
4. **Performance** - Optimization improvements
5. **Documentation** - Better guides and examples
6. **Testing** - Increased test coverage

## 📞 Getting Help

- **GitHub Discussions**: Ask questions
- **GitHub Issues**: Report bugs
- **Email**: krishnagsvv@gmail.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to EquiLens!** 🎉
