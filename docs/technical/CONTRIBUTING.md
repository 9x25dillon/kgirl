# Contributing to Numbskull

Thank you for your interest in contributing to Numbskull! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of embedding systems and machine learning

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/numbskull.git
   cd numbskull
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development dependencies**
   ```bash
   pip install pytest black flake8 mypy
   ```

5. **Run tests**
   ```bash
   cd advanced_embedding_pipeline
   python simple_test.py
   ```

## 🎯 How to Contribute

### Types of Contributions

1. **Bug Reports**: Report issues you find
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Fix bugs or add features
4. **Documentation**: Improve documentation
5. **Testing**: Add tests or improve test coverage

### Contribution Workflow

1. **Create an Issue**
   - Describe the problem or feature request
   - Provide context and examples
   - Assign appropriate labels

2. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/your-bug-fix
   ```

3. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run basic tests
   python simple_test.py
   
   # Run full integration tests (if applicable)
   python integration_test.py
   
   # Run linting
   flake8 .
   black --check .
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Provide clear description
   - Link to related issues
   - Request reviews from maintainers

## 📝 Coding Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions small and focused

### Example Code Style

```python
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExampleEmbedder:
    """
    Example embedder following coding standards.
    
    Args:
        config: Configuration for the embedder
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Implementation here
        return [0.1, 0.2, 0.3]  # Example return
```

### Documentation Standards

- Use clear, concise language
- Provide examples for complex features
- Update README.md for user-facing changes
- Add docstrings for all public APIs

### Testing Standards

- Write tests for new functionality
- Aim for >80% test coverage
- Test both success and failure cases
- Use descriptive test names

```python
import pytest
from advanced_embedding_pipeline import ExampleEmbedder


class TestExampleEmbedder:
    """Test suite for ExampleEmbedder."""
    
    def test_embed_text_success(self):
        """Test successful text embedding."""
        embedder = ExampleEmbedder({})
        result = embedder.embed_text("test text")
        assert len(result) > 0
        assert all(isinstance(x, (int, float)) for x in result)
    
    def test_embed_text_empty_input(self):
        """Test embedding with empty input."""
        embedder = ExampleEmbedder({})
        with pytest.raises(ValueError):
            embedder.embed_text("")
```

## 🧪 Testing

### Running Tests

```bash
# Basic functionality test
python simple_test.py

# Full integration test (requires external services)
python integration_test.py

# Comprehensive demo
python demo.py

# Unit tests (when available)
pytest tests/
```

### Adding New Tests

1. Create test files in the `tests/` directory
2. Use descriptive test names
3. Test both positive and negative cases
4. Mock external dependencies when possible

## 📚 Documentation

### Types of Documentation

1. **README.md**: Main project documentation
2. **Code Comments**: Inline code documentation
3. **Docstrings**: Function and class documentation
4. **Examples**: Usage examples and tutorials

### Writing Documentation

- Use clear, simple language
- Provide practical examples
- Keep documentation up-to-date
- Use consistent formatting

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Try the latest version
3. Reproduce the issue
4. Gather relevant information

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9]
- Numbskull Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other relevant information.
```

## 🔍 Code Review Process

### For Contributors

- Respond to review feedback promptly
- Make requested changes
- Ask questions if feedback is unclear
- Test changes after addressing feedback

### For Reviewers

- Be constructive and respectful
- Focus on code quality and correctness
- Provide specific, actionable feedback
- Approve when changes meet standards

## 📋 Pull Request Guidelines

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] No unnecessary debug code
```

## 🏷️ Labels and Milestones

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information is requested

### PR Labels

- `ready for review`: Ready for code review
- `work in progress`: Still being worked on
- `needs testing`: Requires additional testing
- `breaking change`: Changes existing behavior

## 🚀 Release Process

1. **Version Bumping**: Update version in `setup.py` and `__init__.py`
2. **Changelog**: Update CHANGELOG.md with new features/fixes
3. **Testing**: Ensure all tests pass
4. **Tagging**: Create git tag for release
5. **Documentation**: Update documentation if needed

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/9x25dillon/numbskull/issues)
- **Discussions**: [GitHub Discussions](https://github.com/9x25dillon/numbskull/discussions)
- **Email**: [Your contact information]

## 📄 License

By contributing to Numbskull, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Numbskull! 🎉
