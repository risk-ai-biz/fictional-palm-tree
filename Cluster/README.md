
# Clustered Optimizer (Additive + Pairwise)

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run demo
```bash
python clustered_optimizer.py
```

## Run tests
```bash
pytest -q
```

## Lint (ruff)
```bash
pip install ruff
ruff check .
```

## Type-check (mypy)
```bash
pip install mypy
mypy clustered_optimizer.py
```
