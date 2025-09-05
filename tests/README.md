# EquiLens Tests

This directory contains all test files for the EquiLens project, including enhanced auditor validation and bias impact analysis.

## Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for full workflows
- `conftest.py` - Pytest configuration and fixtures
- `test_enhanced_features.py` - Enhanced auditor feature demonstration
- `validate_enhanced_auditor.py` - Quick validation for enhanced auditor
- `test_configurations.py` - Comprehensive configuration and bias impact testing

## Enhanced Auditor Tests

### Quick Validation
```bash
# Validate enhanced auditor implementation
python tests/validate_enhanced_auditor.py
```

### Feature Testing
```bash
# Test enhanced features (structured output, sampling)
python tests/test_enhanced_features.py
```

### Bias Impact Analysis
```bash
# Comprehensive configuration testing
python tests/test_configurations.py
```

## Running Tests

```bash
# Run all pytest tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src/equilens
```

## Test Categories

### Unit Tests
- ETA functionality testing
- Interactive components testing

### Integration Tests
- Dynamic concurrency testing
- Enhanced audit fixes testing
- Resume CLI testing
- Full audit workflow testing
