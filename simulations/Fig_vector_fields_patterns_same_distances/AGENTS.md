# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++17 sources (`*.cc`) — core logic (`simulations.cc`, `network.cc`, `utils.cc`).
- `include/`: Public headers (`*.hpp`).
- `obj/`: Build artifacts created by the Makefile.
- `simulations`: Built binary.
- `makefile`: Build rules. `.vscode/`: editor settings.
- Outputs: results are written under `../../../data/...` as configured by `foldername_results` in `src/simulations.cc`.

## Build, Test, and Development Commands
- `make` or `make -j`: Build `simulations` with `-O3 -std=c++17 -march=native -pthread`.
- `make run`: Build then run `./simulations` from the repo root.
- Clean artifacts: `rm -rf obj simulations` (no `clean` target yet).
- Manual build (alternative): `g++ -O3 -std=gnu++17 -march=native -pthread -Iinclude src/*.cc -o simulations`.

## Coding Style & Naming Conventions
- C++17, brace on same line, 4-space indentation, no tabs.
- Filenames use snake_case (`network.cc`, `utils.hpp`); headers end with `.hpp`.
- Prefer `std::` containers/algorithms; avoid raw pointers where possible.
- Keep headers minimal; place implementations in `src/`.

## Testing Guidelines
- No formal suite yet. For contributions, add small tests in `tests/` (GoogleTest or a simple executable) that compile with `-Iinclude src/*.cc`.
- Make runs deterministic by seeding RNGs in tests.
- Name tests `test_<area>.cc` and document expected outputs/assumptions.

## Commit & Pull Request Guidelines
- Commits: short, imperative, and specific (e.g., "tune inhibitory update", "add vector field export").
- PRs include: concise description, motivation, run steps (`make`, `make run`), and evidence (logs, sample output files). Link related issues when applicable.

## Security & Configuration Tips
- AVX intrinsics are used; ensure your CPU supports them (or adjust flags/includes).
- Workloads are multithreaded/heavy; adjust `max_threads` in `src/simulations.cc`.
- Change `foldername_results` in `src/simulations.cc` to control output location and avoid writing outside intended directories.

