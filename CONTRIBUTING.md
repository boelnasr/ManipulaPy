# Contributing to ManipulaPy

Thank you for your interest in contributing to ManipulaPy! We welcome contributions that help improve this project and make it more robust, reliable, and user-friendly.

---

## Contribution Guidelines

### How to Contribute

- Open an issue if you notice a bug, have a feature request, or a question.
- Fork the repository and create a new branch for your feature or fix.
- Ensure your code adheres to PEP8 and includes tests where applicable.
- Submit a Pull Request (PR) with a clear description of what it does.

### Testing

- We use `pytest` to run our test suite.
- Tests should cover new features and significant changes to existing logic.
- Include clear docstrings and comments in your test cases.

---

## Development

### Build & Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_kinematics.py -v

# Run a single test
python -m pytest tests/test_control.py::TestManipulatorController::test_pid_control -v

# Run tests by marker
python -m pytest tests/ -v -m "not (cuda or vision or simulation)"

# Run with coverage
python -m pytest tests/ --cov=ManipulaPy --cov-report=term-missing

# Lint (check only)
python -m black --check ManipulaPy/ tests/
python -m isort --check-only ManipulaPy/ tests/
python -m flake8 ManipulaPy/ tests/

# Lint (auto-fix)
python -m black ManipulaPy/ tests/
python -m isort ManipulaPy/ tests/

# Build docs
python -m sphinx -b html docs/source docs/build/html
```

### CI/CD

- **test.yml**: Python 3.8/3.9/3.10/3.11 matrix, PyTorch CPU, Codecov. Env: `SKIP_CUDA_TESTS=true`, `SKIP_VISION_TESTS=true`, `SKIP_SIMULATION_TESTS=true`
- **lint.yml**: black --check + flake8, auto-commits formatting fixes
- **pypi-publish.yml**: Triggered by `v*` tags, publishes to PyPI via `PYPI_API_TOKEN`

For architecture details, class hierarchy, GPU/CPU strategy, and code conventions,
see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## AI Usage Policy

We value transparency and responsibility in software development. While generative AI tools (e.g., GitHub Copilot, ChatGPT, CodeWhisperer) can be helpful in drafting and brainstorming, **all contributions must be human-verified and reviewed**.

### Acceptable AI Use

You may use AI tools to:
- Generate boilerplate code or documentation drafts.
- Explore alternative implementations (after reviewing and testing).
- Assist with initial responses to issues (after editing for clarity and correctness).

### Unacceptable AI Use

You must **not**:
- Submit AI-generated code or text **without understanding and testing it**.
- Use AI to auto-respond to issues or PRs without human oversight.
- Include unexplained or unverifiable code from AI tools.

---

##  Maintainer Commitment

As the project maintainer:
- I may use AI tools to accelerate documentation or draft code—but **every line is reviewed and validated before merging**.
- I take full responsibility for all merged content, whether AI-assisted or not.
- Users and contributors will receive **thoughtful, human-reviewed support**.

This policy aligns with [JOSS guidance](https://joss.theoj.org/about#ai-policy), which states:
> "Authors are responsible for understanding and explaining submitted code and its provenance, and should respond in good faith to reviewer questions about LLM use as they would with any other topic."

---

##  Contributor AI Use Policy

If you use AI assistance for your contribution:
- Mention it in your pull request or commit message.
- Only submit code you fully understand and have tested.
- Be prepared to explain your changes.

---

## 🔄 Policy Evolution

This AI policy may evolve as community practices and journal guidelines (including JOSS) develop. We welcome suggestions and questions.

---

Last updated: 2025-07-26
