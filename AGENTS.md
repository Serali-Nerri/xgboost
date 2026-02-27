# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains pipeline modules: `data_loader.py`, `preprocessor.py`, `model_trainer.py`, `evaluator.py`, `predictor.py`, `visualizer.py`, plus `src/utils/` for logging and model I/O helpers.
- Entry points are `train.py` (training/evaluation) and `predict.py` (inference).
- Configuration lives in `config/config.yaml`; prefer config-driven changes over hardcoded values.
- Data is organized under `data/raw/`, `data/processed/`, and `data/models/`.
- Generated artifacts go to `output/` and `logs/` (both git-ignored).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python train.py --config config/config.yaml`: run full training, CV, evaluation, and artifact export.
- `python predict.py --model output --input data/raw/all.csv --output output/predictions.csv`: run batch prediction with saved artifacts.
- `pyright`: run static type checking (configured by `pyrightconfig.json`).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear docstrings for public classes/functions.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes (for example, `ModelTrainer`, `DataLoader`).
- Keep modules focused: data handling in `data_loader/preprocessor`, training logic in `model_trainer`, metrics in `evaluator`.
- Prefer type hints on new or modified public APIs.

## Testing Guidelines
- Add tests in `tests/` using `test_*.py` naming.
- Use `pytest` style assertions for new coverage; run with `pytest -q`.
- For pipeline changes, include a smoke check by running `train.py` and one `predict.py` command, then verify expected files in `output/`.

## Commit & Pull Request Guidelines
- Current history mixes styles; standardize on Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Keep commits scoped and imperative (for example, `fix: guard missing features in predictor`).
- PRs should include: purpose, key files changed, config/data assumptions, and validation evidence (metrics such as RMSE/R²/COV, or command output).
- Link related issues and include updated plots/report paths when model behavior changes.
