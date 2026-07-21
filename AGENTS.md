# Repository Guidelines

## Project Structure & Module Organization

This repository is a ComfyUI custom node package for generating Anima prompts.

- `__init__.py` registers the three ComfyUI nodes.
- `nodes/` contains the local/OpenAI LLM loaders and prompt generator node.
- `services/` contains provider adapters, tag indexing, CSV validation, and the prompt pipeline.
- `data/tags.csv` is the validated tag taxonomy used at runtime.
- `tests/` contains focused pytest tests for node registration, retrieval, validation, and generation.
- `README.md` documents installation, tag scopes, and node behavior.

Keep UI-facing node definitions in `nodes/`; reusable logic belongs in `services/`.

## Build, Test, and Development Commands

There is no separate build step. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests --import-mode=importlib
python -m compileall -q __init__.py nodes services tests
```

The first command installs the OpenAI backend. Install a CPU, CUDA, or ROCm-compatible
`llama-cpp-python` build separately for local GGUF inference. The pytest suite uses fake
providers and temporary CSV files, so it requires neither an API key nor a model file.

## Coding Style & Naming Conventions

Use four-space indentation, modern Python type hints, and `from __future__ import annotations`.
Follow existing naming: `snake_case` for functions and variables, `PascalCase` for classes,
and uppercase names for module constants. Prefer standard-library solutions and keep changes
within the existing pipeline/index boundaries. No formatter or linter is configured; match
the surrounding style and run `compileall` before submitting.

## Testing Guidelines

Use pytest. Name files `test_*.py` and tests `test_<behavior>`. Add the smallest regression
test that exercises public behavior, especially for CSV trust-boundary validation, category
scope selection, output escaping, and LLM response recovery. Run the complete suite after
changes to shared services.

## Commit & Pull Request Guidelines

History is minimal and has no formal convention. Use short imperative subjects such as
`fix random candidate coverage`. Keep commits narrowly scoped. Pull requests should explain
the behavior change, list verification commands, and note README or workflow compatibility
changes. Include screenshots only when node inputs or outputs visibly change.

## Security & Configuration

Prefer the `OPENAI_API_KEY` environment variable. Directly entered keys are serialized into
ComfyUI workflows; never commit or share workflows containing credentials. Do not weaken
`tags.csv` schema validation or log API keys and provider secrets.
