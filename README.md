# Advanced Intraday Stock Signal Dashboard

A Streamlit app for Indian intraday stock research, signal generation, and paper-trading style evaluation.

## What this app does

- Downloads OHLCV data from Yahoo Finance using `yfinance`
- Builds technical, volatility, volume, session, and regime features
- Handles class imbalance
- Uses time-ordered validation to avoid future leakage
- Compares candidate models using:
  - PR-AUC
  - recall for class 1
  - balanced accuracy
- Calibrates probabilities when possible
- Tunes the trading threshold using validation PnL
- Evaluates the final model on a held-out test slice
- Refits a final live model on all labeled data for the latest signal

## Why this version is different

This version is designed to be more realistic for trading research than a simple demo app.

It includes:

- a less noisy default target
- stronger imbalance handling
- a smaller threshold search range so the model does not become too conservative
- a trend/regime filter
- validation-based threshold selection
- live signal output based on calibrated probabilities

## Important note

This app is for research and paper trading only.

It does **not** place real trades by itself.

For live order execution, connect a broker API separately and keep:
- signal generation
- order placement
- risk management

as separate layers.

## Project structure

```text
stock-intraday-advanced/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
