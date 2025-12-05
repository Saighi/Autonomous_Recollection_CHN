# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds core C++ logic (e.g., network dynamics, utilities); headers live in `include/`.
- `simulations/*.cc` are entrypoints that link against the core; binaries are written to `bin/` and intermediates to `obj/`.
- `data/` stores generated configs, trained networks, and simulation outputs; treat large artifacts as non-versioned.
- `scripts/` contains the Python helper workflow (`utils.py`, `workflow.py`) for generating patterns/configs and launching C++ runs.
- The `makefile` auto-discovers simulations; add new simulations as `simulations/<name>.cc`.

## Build, Test, and Development Commands
- `make` — build all simulations; `make bin/<name>` builds one (`make bin/write`).
- `make run_write` / `make run_sleep` — build then execute the corresponding binaries.
- `make list` — show discovered simulations; `make clean` / `make distclean` remove artifacts.
- Python workflow: import from `scripts/workflow.py` to generate configs and call `run_cpp("write"|"sleep", config_path)` for reproducible experiments.

## Coding Style & Naming Conventions
- C++17, compiled with `g++` using `-pthread -g -Wall -O3 -march=native`; keep code warning-free.
- Use 4-space indentation, brace-on-newline style (as in `src/utils.cc`); prefer `snake_case` for functions/variables and `PascalCase` for classes.
- Protect headers with uppercase include guards; keep includes ordered stdlib-first, project-second.
- Favor `const` references for inputs, minimize global state, and mirror existing file naming (`write.cc`, `sleep.cc`).

## Testing Guidelines
- No formal test suite; validate changes by rebuilding and running the relevant simulations (`make run_write`, then `make run_sleep` on the produced data).
- For new features, add a minimal parameter set to exercise the path and confirm outputs land under `data/` with expected filenames.
- When altering Python helpers, run a small dry run (e.g., `python - <<'PY' ...` using `run_cpp`) to ensure generated configs match C++ expectations.

## Commit & Pull Request Guidelines
- Keep commit subjects concise and action-oriented (history favors short, lowercased summaries); describe scope and impact in the body if needed.
- Reference related issues/experiments, list commands run, and note any new data directories created.
- Avoid committing large generated artifacts or binaries; ensure `make clean` leaves the tree ready for review.
- For PRs, include a brief reproduction recipe and sample output path names to help reviewers verify results.
