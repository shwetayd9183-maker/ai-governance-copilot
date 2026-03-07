---
description: "Enforce coding patterns and best practices for the AI governance copilot project, including data handling, model training, and governance logic."
---

# AI Governance Copilot Instructions

## Data Handling
- Standardize ALL data columns to `SNAKE_CASE` (no spaces, no mixed case).
- Always read crop/threshold configs from `config.py`, never hardcode values.
- Use `groupby(crop/district)` + `transform()` in feature engineering to prevent data leakage.
- Implement explicit checks before processing (file existence, column presence, data ranges).

## Coding Style
- Add type hints to all function signatures for clarity.
- Require docstrings for all public functions (include parameters, returns, example).
- Replace bare `except:` with specific exception types; add logging.

## Model Training
- Use `TimeSeriesSplit` for chronological data; NEVER random shuffle.
- Always use `.joblib` format (not pickle) for scikit-learn models.
- Apply class imbalance handling with `compute_class_weight('balanced')`.

## API and Resilience
- All external API calls require timeout, retry logic, and explicit error handling.

## Governance Logic
- Ensure economic justification: intervene only if prob_severe >= 0.6 AND expected_loss > fiscal_cost.