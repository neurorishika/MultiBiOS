# Contributing to MultiBiOS

We welcome contributions from the neuroscience and engineering communities! This guide will help you get started with contributing to the MultiBiOS project.

## Getting Started

### Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/neurorishika/MultiBiOS.git
   cd MultiBiOS
   ```

2. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Run tests:**
   ```bash
   poetry run pytest tests/
   ```

4. **Format your code:**
   ```bash
   poetry run black multibios/
   poetry run ruff check multibios/
   ```

5. **Update documentation** if needed

6. **Commit and push:**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Code Standards

### Python Code

- **Formatting:** Use `black` for code formatting
- **Linting:** Use `ruff` for linting and style checks
- **Type hints:** Add type annotations for new functions
- **Docstrings:** Document all public functions and classes

```python
def compile_protocol(protocol_data: Dict[str, Any], seed: int = 42) -> CompiledProtocol:
    """Compile a YAML protocol into hardware timing data.
    
    Args:
        protocol_data: Raw protocol dictionary from YAML
        seed: Random seed for reproducible randomization
        
    Returns:
        Compiled protocol with timing arrays and device mappings
        
    Raises:
        ProtocolError: If protocol contains timing conflicts
    """
```

### Firmware Code

- **Arduino style:** Follow Arduino/C++ conventions
- **Comments:** Document timing-critical sections extensively
- **Constants:** Use meaningful names for pin assignments and timing values

### Documentation

- **Markdown:** Follow standard Markdown formatting
- **Examples:** Include practical examples for new features
- **Links:** Use relative links for internal documentation references

## Testing Guidelines

### Unit Tests

- Add tests in `tests/` for all new protocol features
- Test both valid and invalid inputs
- Ensure tests are deterministic (use fixed seeds)
- Test edge cases and error conditions

```python
def test_protocol_compilation():
    """Test that valid protocols compile correctly."""
    protocol = {
        "protocol": {"name": "Test", "timing": {"sample_rate": 1000}},
        "sequence": [{"phase": "test", "duration": 1000, "times": 1, "actions": []}]
    }
    compiled = compile_protocol(protocol, seed=42)
    assert compiled.total_samples == 1000
```

### Integration Tests

- Test complete protocol execution in dry-run mode
- Validate timing guardrails with realistic scenarios
- Test visualization generation

### Hardware Tests

- If you have access to hardware, test critical changes
- Document any hardware-specific test procedures
- Include photos/videos of test setups if helpful

## Documentation Updates

When making changes, update relevant documentation:

### Schema Changes

- Update `docs/protocol.md` with new YAML syntax
- Modify `config/example_protocol.yaml` with examples
- Update `config/hardware.yaml` if adding new device types

### Feature Additions

- Add sections to appropriate documentation files
- Include practical examples and use cases
- Update the FAQ if addressing common questions

### Breaking Changes

- Clearly document migration steps
- Update all example files
- Consider backwards compatibility options

## Contribution Types

### 🐛 Bug Fixes

- Include a clear description of the bug
- Provide steps to reproduce the issue
- Add a test case that would have caught the bug

### ✨ New Features

- Discuss major features in an issue first
- Ensure features align with project goals
- Include comprehensive tests and documentation

### 📚 Documentation Improvements

- Fix typos and improve clarity
- Add missing examples or explanations
- Improve formatting and organization

### 🔧 Hardware Support

- Support for new DAQ devices or valve controllers
- Additional safety features or interlocks
- Performance optimizations

## Design Principles

When contributing, keep these principles in mind:

### 🎯 Deterministic Behavior

- All randomization should respect the configured seed
- Timing behavior should be predictable and reproducible
- Avoid non-deterministic operations in critical paths

### 🛡️ Safety First

- Hardware safety features should never be optional
- Timing guardrails prevent dangerous conditions
- Fail-safe defaults for all configuration options

### 📊 Data Provenance

- Log all relevant parameters and timing information
- Ensure experimental conditions can be exactly reproduced
- Maintain backwards compatibility for data formats

### ⚡ Performance

- Hardware timing is critical - avoid blocking operations
- Optimize for the common case (standard protocols)
- Consider memory usage for long experiments

## Release Process

### Versioning

We follow semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes to APIs or protocols
- **Minor** (X.Y.0): New features, backwards compatible
- **Patch** (X.Y.Z): Bug fixes, no new features

### Changelog

- Document all changes in `CHANGELOG.md`
- Include migration notes for breaking changes
- Credit all contributors

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Celebrate contributions of all sizes

### Communication

- Use GitHub issues for bug reports and feature requests
- Tag issues appropriately (`bug`, `enhancement`, `documentation`)
- Provide sufficient context for others to understand and reproduce issues

## Recognition

Contributors are recognized in:
- `AUTHORS.md` file in the repository
- Release notes for significant contributions
- Documentation credits where appropriate

## Getting Help

- **GitHub Issues:** For bugs and feature requests
- **GitHub Discussions:** For questions and community support
- **Email:** For sensitive issues or collaboration inquiries

---

Thank you for contributing to MultiBiOS! Your efforts help advance neuroscience research through better experimental tools.
