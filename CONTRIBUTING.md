# Contributing to Business Entity Resolution Pipeline

First off, thank you for considering contributing to our project! 🎉 Enterprise Entity Resolution at 10M+ scale requires precision, scalable algorithms, and community collaboration.

---

## 📜 Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Pull Requests](#pull-requests)
3. [Development Setup](#development-setup)
4. [Coding & Design Standards](#coding--design-standards)
5. [Testing & Submission Validation](#testing--submission-validation)
6. [Commit Guidelines](#commit-guidelines)

---

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues to ensure it hasn't already been reported. When filing a bug:
- Use the **Bug Report** template.
- Provide a clear, descriptive title.
- Include reproduction steps and sample entity strings (e.g., specific Devanagari/Tamil transliteration or French address cases).
- Describe expected vs actual behavior.

### Suggesting Enhancements
Feature requests are always welcome!
- Use the **Feature Request** template.
- Explain why this enhancement would be useful to reaching the $\ge 0.998$ $F_{0.5}$ score target.
- Suggest architectural reduction or efficiency ideas for blocking and feature extraction.

### Pull Requests
1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```
2. Implement your changes following our coding standards.
3. Verify your changes against the official competition validator:
   ```bash
   python Deputy_pipe/Datasets/student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir Deputy_pipe/Datasets/student_resource/dataset/test
   ```
4. Push to your fork and submit a Pull Request!

---

## Development Setup

1. **Clone and Virtual Environment:**
   ```bash
   git clone https://github.com/naveencmy/business_entity_resolution.git
   cd business_entity_resolution
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r code/business_entity_resolution/requirements.txt
   ```

2. **Core Dependencies:**
   - `python >= 3.10`
   - `xgboost` (CUDA GPU enabled if available)
   - `scikit-learn`, `scipy`, `pandas`, `numpy`

---

## Coding & Design Standards

- **Strict Tab Separation:** All TSV readers and writers MUST use `sep="\t"` or `csv.reader(f, delimiter="\t")`. Never use comma splitting on raw rows.
- **Open-Set Country Partitioning:** Never hard-code filters or encoders to only `{US, India}`. Test datasets include `France` and potentially other countries.
- **Transliteration & Normalization:** Keep text processing stateless and idempotent.
- **Memory Discipline:** Because datasets exceed 10 million rows, avoid accumulating giant unbounded Python lists in memory. Stream records in chunks or partition by country.

---

## Testing & Submission Validation

All PRs that modify candidate blocking or classification must pass:
1. Candidate recall gate ($\ge 92\%+$).
2. Reduction ratio gate ($> 99.998\%$).
3. Submission validator gate (`exit code 0`).

---

## Commit Guidelines

We follow Conventional Commits:
- `feat:` A new feature (e.g., new blocking key, transliteration improvement)
- `fix:` A bug fix (e.g., address parsing edge case)
- `docs:` Documentation updates
- `perf:` Performance optimizations (e.g., vectorizing feature extraction)
- `test:` Adding or refactoring tests
- `refactor:` Code refactoring without changing external behavior
