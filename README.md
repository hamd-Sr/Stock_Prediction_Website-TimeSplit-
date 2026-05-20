# Advanced Intraday Stock Signal Dashboard

A separate Streamlit app for intraday market research, signal generation, and paper-trading style evaluation.

## What it does

- Pulls Indian market data from Yahoo Finance using `yfinance`
- Builds technical, session-based, and volatility features
- Uses time-ordered validation so future data is not mixed into training
- Compares candidate models
- Calibrates probabilities when possible
- Tunes the trading threshold on validation data
- Shows a live-style signal, test metrics, and a strategy curve

## Why this design

This app is built around time-series best practices:

- `TimeSeriesSplit` is used because regular cross-validation can leak future information into earlier training folds.
- `CalibratedClassifierCV` is used to make probabilities more reliable.
- LightGBM is the preferred model when it is available, because it is a tree-based gradient boosting framework designed for efficiency and strong performance on tabular data.

## Files

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `.gitignore`

## Setup

### 1) Create a new GitHub repository
Use a new repo so your older app stays untouched.

### 2) Add the files
Place `app.py` and `requirements.txt` in the repository root.

### 3) Optional local install
If your environment supports it and you want the preferred boosting model:

```bash
pip install lightgbm
