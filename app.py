"""Advanced Intraday Stock Signal Dashboard

What this app does:
- Downloads intraday or daily OHLCV data from Yahoo Finance via yfinance
- Builds technical + session-based features
- Trains candidate models on time-ordered data
- Uses TimeSeriesSplit for model selection
- Calibrates probabilities when possible
- Tunes the trading threshold on validation data
- Evaluates on a held-out test slice
- Refits a final live model on all labeled data for the latest signal

Educational use only. Not financial advice.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier  # type: ignore

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Advanced Intraday Signal Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Advanced Intraday Stock Signal Dashboard")
st.caption(
    "Time-series validation, calibrated probabilities, threshold tuning, and a paper-trading style signal view."
)


# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(symbol: str, market: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return "RELIANCE.NS"
    if "." in symbol:
        return symbol
    return f"{symbol}.NS" if market == "NSE" else f"{symbol}.BO"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for col in out.columns.to_flat_index():
            parts = [str(part) for part in col if part not in ("", None)]
            cols.append("_".join(parts) if parts else str(col))
        out.columns = cols
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo-style columns to Open/High/Low/Close/Volume."""
    out = flatten_columns(df)

    normalized = {str(c).replace(" ", "_").lower(): c for c in out.columns}

    def find_col(target: str):
        target_l = target.lower()
        if target_l in normalized:
            return normalized[target_l]
        for norm_name, original in normalized.items():
            if norm_name.endswith(f"_{target_l}") or norm_name.startswith(f"{target_l}_") or target_l in norm_name:
                return original
        return None

    rename_map = {}
    for canonical in ["Open", "High", "Low", "Close", "Volume"]:
        src = find_col(canonical)
        if src is not None and src != canonical:
            rename_map[src] = canonical

    out = out.rename(columns=rename_map)
    out.columns = [str(c).replace(" ", "_") for c in out.columns]

    # Fallback: if the first five columns look like OHLCV but names are odd, map positionally.
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing and len(out.columns) >= 5:
        pos_map = {}
        first_five = list(out.columns[:5])
        for i, canonical in enumerate(required):
            if canonical not in out.columns and i < len(first_five):
                pos_map[first_five[i]] = canonical
        out = out.rename(columns=pos_map)

    return out


def ensure_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_ohlcv_columns(df)
    needed = ["Open", "High", "Low", "Close"]
    if not all(c in out.columns for c in needed):
        raise ValueError(f"Could not standardize OHLC columns. Found columns: {list(out.columns)}")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out


@st.cache_data(ttl=300)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = standardize_ohlcv_columns(df)
    df = df.dropna(how="all")
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def intraday_vwap(frame: pd.DataFrame) -> pd.Series:
    if "Volume" not in frame.columns or frame["Volume"].isna().all():
        return pd.Series(index=frame.index, dtype=float)

    tp = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    vol = frame["Volume"].fillna(0)

    session_key = pd.Series(pd.DatetimeIndex(frame.index).date, index=frame.index)
    cum_pv = (tp * vol).groupby(session_key).cumsum()
    cum_vol = vol.groupby(session_key).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def build_features(raw: pd.DataFrame, horizon_bars: int, move_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    df = ensure_price_frame(raw).copy()
    idx = pd.DatetimeIndex(df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # Returns and momentum
    df["ret_1"] = close.pct_change()
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["log_ret_1"] = np.log(close / close.shift(1))
    df["log_ret_3"] = np.log(close / close.shift(3))
    df["momentum_5"] = close - close.shift(5)
    df["momentum_10"] = close - close.shift(10)

    # Trend
    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"dist_sma_{w}"] = close / df[f"sma_{w}"] - 1.0
        df[f"volatility_{w}"] = df["ret_1"].rolling(w).std()

    # Momentum indicators
    df["rsi_14"] = rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volatility / range
    df["atr_14"] = atr(high, low, close, 14)
    df["hl_range"] = (high - low) / close
    df["oc_range"] = (open_ - close) / close
    df["body"] = (close - open_) / close
    df["upper_wick"] = (high - np.maximum(open_, close)) / close
    df["lower_wick"] = (np.minimum(open_, close) - low) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid
    df["bb_z"] = (close - bb_mid) / bb_std

    # Volume
    df["vol_chg"] = volume.pct_change()
    df["vol_sma_20"] = volume.rolling(20).mean()
    df["vol_z"] = (volume - df["vol_sma_20"]) / volume.rolling(20).std()
    df["price_vol"] = df["ret_1"] * df["vol_chg"].fillna(0)

    vwap = intraday_vwap(df)
    df["vwap"] = vwap
    df["close_vwap"] = close / vwap - 1.0
    df["vwap_z"] = (close - vwap) / close

    # Session / clock features
    df["dow"] = idx.dayofweek
    df["hour"] = idx.hour
    df["minute"] = idx.minute
    df["minute_of_session"] = df.groupby(pd.Series(idx.date, index=df.index)).cumcount()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_minute"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_minute"] = np.cos(2 * np.pi * df["minute"] / 60)

    # Target
    df["future_return"] = close.shift(-horizon_bars) / close - 1.0
    df["target"] = (df["future_return"] > move_threshold).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude = {"future_return", "target"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return df, feature_cols


def make_time_split(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    train_end = max(int(n * train_ratio), 1)
    val_end = max(int(n * (train_ratio + val_ratio)), train_end + 1)
    val_end = min(val_end, n - 1)
    return train_end, val_end


def tscv_for_samples(n_samples: int) -> TimeSeriesSplit:
    if n_samples < 120:
        n_splits = 3
    elif n_samples < 240:
        n_splits = 4
    else:
        n_splits = 5
    return TimeSeriesSplit(n_splits=n_splits)


def build_candidate_models() -> Dict[str, object]:
    models: Dict[str, object] = {}

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=350,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    models["LogisticRegression"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )

    return models


def safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, proba))


@dataclass
class CandidateScore:
    name: str
    cv_accuracy: float
    cv_f1: float
    cv_balanced_accuracy: float
    cv_auc: float
    estimator: object


def score_model_timeseries(name: str, estimator: object, X_train: pd.DataFrame, y_train: pd.Series) -> CandidateScore:
    cv = tscv_for_samples(len(X_train))
    accs, f1s, bals, aucs = [], [], [], []

    for train_idx, valid_idx in cv.split(X_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = clone(estimator)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_va)[:, 1]
        else:
            proba = np.full(len(X_va), 0.5)

        accs.append(accuracy_score(y_va, pred))
        f1s.append(f1_score(y_va, pred, zero_division=0))
        bals.append(balanced_accuracy_score(y_va, pred))
        aucs.append(safe_auc(y_va, proba))

    return CandidateScore(
        name=name,
        cv_accuracy=float(np.nanmean(accs)),
        cv_f1=float(np.nanmean(f1s)),
        cv_balanced_accuracy=float(np.nanmean(bals)),
        cv_auc=float(np.nanmean(aucs)),
        estimator=estimator,
    )


def try_calibrated_fit(estimator: object, X: pd.DataFrame, y: pd.Series) -> object:
    cv = tscv_for_samples(len(X))
    try:
        calibrated = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=cv,
        )
        calibrated.fit(X, y)
        return calibrated
    except Exception:
        fallback = clone(estimator)
        fallback.fit(X, y)
        return fallback


def predict_proba_positive(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.full(len(X), 0.5)


def positions_from_probability(proba: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(proba >= threshold, 1, np.where(proba <= 1 - threshold, -1, 0))


def strategy_frame(
    y_future_return: pd.Series,
    proba: np.ndarray,
    threshold: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=y_future_return.index)
    frame["future_return"] = y_future_return
    frame["prob_up"] = proba
    frame["position"] = positions_from_probability(proba, threshold)

    turnover = np.abs(np.diff(np.r_[0, frame["position"].to_numpy()]))
    cost = turnover * (transaction_cost_bps / 10000.0)
    frame["transaction_cost"] = cost
    frame["strategy_return"] = frame["position"] * frame["future_return"] - frame["transaction_cost"]
    frame["buy_hold"] = frame["future_return"]
    frame["strategy_curve"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_curve"] = (1 + frame["buy_hold"]).cumprod()
    return frame.dropna()


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def threshold_search(
    y_future_return: pd.Series,
    proba: np.ndarray,
    transaction_cost_bps: float,
    start: float = 0.50,
    stop: float = 0.80,
    step: float = 0.01,
) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(start, stop + 1e-9, step), 2)

    for t in thresholds:
        sf = strategy_frame(y_future_return, proba, t, transaction_cost_bps)
        if sf.empty:
            continue
        net = sf["strategy_return"]
        mean = float(net.mean())
        std = float(net.std(ddof=0))
        sharpe_like = mean / std if std > 0 else -np.inf
        rows.append(
            {
                "threshold": t,
                "total_net_return": float((1 + net).prod() - 1),
                "mean_return": mean,
                "sharpe_like": sharpe_like,
                "trade_count": int((sf["position"] != 0).sum()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["total_net_return", "sharpe_like"], ascending=False).reset_index(drop=True)
    return out


def metric_cards_for_strategy(sf: pd.DataFrame) -> Dict[str, float]:
    active = sf[sf["position"] != 0].copy()
    win_rate = float((active["strategy_return"] > 0).mean()) if not active.empty else np.nan
    total_return = float((1 + sf["strategy_return"]).prod() - 1)
    buy_hold_return = float((1 + sf["buy_hold"]).prod() - 1)
    sharpe_like = float(sf["strategy_return"].mean() / sf["strategy_return"].std(ddof=0)) if sf["strategy_return"].std(ddof=0) > 0 else np.nan

    return {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "win_rate": win_rate,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown(sf["strategy_curve"]),
        "trade_count": int((sf["position"] != 0).sum()),
    }


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["NSE", "BSE"], index=0)
    ticker_input = st.text_input("Ticker", value="RELIANCE")
    period = st.selectbox("History window", ["30d", "60d", "3mo", "6mo"], index=1)
    interval = st.selectbox("Candles", ["5m", "15m", "30m", "60m", "1d"], index=0)
    horizon_bars = st.selectbox("Prediction horizon (bars)", [1, 2, 3, 5], index=1)
    move_threshold_pct = st.slider("Future move threshold (%)", 0.00, 1.00, 0.15, 0.05)
    transaction_cost_bps = st.slider("Transaction cost (bps)", 0.0, 20.0, 5.0, 0.5)
    refresh = st.button("Refresh data")

    st.markdown("---")
    st.write("Preferred model: **LightGBM** (if installed)")
    st.caption("Falls back to scikit-learn gradient boosting if LightGBM is unavailable.")


symbol = normalize_ticker(ticker_input, market)
if refresh:
    st.cache_data.clear()

st.subheader(f"Live view for {symbol}")

raw = load_data(symbol, period, interval)
if raw.empty:
    st.error("No data returned. Check the ticker, market suffix, period, or interval.")
    st.stop()

if len(raw) < 200:
    st.warning("Very small dataset returned. Metrics may be unstable.")

# Prepare data
move_threshold = move_threshold_pct / 100.0
feature_df, feature_cols = build_features(raw, horizon_bars=horizon_bars, move_threshold=move_threshold)

if len(feature_df) < 200:
    st.error("Not enough rows after feature engineering to train a reliable model.")
    st.stop()

X = feature_df[feature_cols].copy()
y = feature_df["target"].copy()
future_returns = feature_df["future_return"].copy()

if y.nunique() < 2:
    st.error("Target has only one class. Lower the future move threshold or use a longer history window.")
    st.stop()

train_end, val_end = make_time_split(len(X), train_ratio=0.70, val_ratio=0.15)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
fr_val = future_returns.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
fr_test = future_returns.iloc[val_end:]

# Candidate ranking
candidates = build_candidate_models()
leaderboard_rows = []
for name, estimator in candidates.items():
    score = score_model_timeseries(name, estimator, X_train, y_train)
    leaderboard_rows.append(
        {
            "model": score.name,
            "cv_accuracy": score.cv_accuracy,
            "cv_f1": score.cv_f1,
            "cv_balanced_accuracy": score.cv_balanced_accuracy,
            "cv_auc": score.cv_auc,
            "estimator": score.estimator,
        }
    )

leaderboard = pd.DataFrame(leaderboard_rows)
leaderboard = leaderboard.sort_values(["cv_f1", "cv_auc", "cv_accuracy"], ascending=False).reset_index(drop=True)

best_name = leaderboard.iloc[0]["model"]
best_estimator = leaderboard.iloc[0]["estimator"]

# Fit calibrated model on training set only
trained_model = try_calibrated_fit(best_estimator, X_train, y_train)

# Tune threshold on validation set to maximize net return
val_proba = predict_proba_positive(trained_model, X_val)
threshold_grid = threshold_search(fr_val, val_proba, transaction_cost_bps)
if threshold_grid.empty:
    best_threshold = 0.55
else:
    best_threshold = float(threshold_grid.iloc[0]["threshold"])

# Evaluate on test
test_proba = predict_proba_positive(trained_model, X_test)
test_pred = positions_from_probability(test_proba, best_threshold)
test_pred_binary = (test_pred == 1).astype(int)

test_sf = strategy_frame(fr_test, test_proba, best_threshold, transaction_cost_bps)
test_metrics = metric_cards_for_strategy(test_sf) if not test_sf.empty else {}

# Standard classification metrics
test_accuracy = accuracy_score(y_test, test_pred_binary)
test_f1 = f1_score(y_test, test_pred_binary, zero_division=0)
test_bal_acc = balanced_accuracy_score(y_test, test_pred_binary)
test_auc = safe_auc(y_test, test_proba)

# Refit final live model on all labeled data for the latest signal
final_model = try_calibrated_fit(best_estimator, X, y)
latest_X = X.iloc[[-1]]
latest_prob = float(predict_proba_positive(final_model, latest_X)[0])

if latest_prob >= best_threshold:
    live_signal = "BUY"
elif latest_prob <= 1 - best_threshold:
    live_signal = "SELL"
else:
    live_signal = "HOLD"

# -----------------------------
# Dashboard
# -----------------------------
plot_source = ensure_price_frame(raw).tail(300)

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=plot_source.index,
        open=plot_source["Open"],
        high=plot_source["High"],
        low=plot_source["Low"],
        close=plot_source["Close"],
        name="Price",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_source.index,
        y=plot_source["Close"].rolling(20).mean(),
        mode="lines",
        name="MA 20",
    )
)
fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_rangeslider_visible=False,
    legend_orientation="h",
)
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{feature_df['Close'].iloc[-1]:.2f}")
c2.metric("Latest up probability", f"{latest_prob:.2%}")
c3.metric("Live signal", live_signal)
c4.metric("Selected model", best_name)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Chosen threshold", f"{best_threshold:.2f}")
c6.metric("Test accuracy", f"{test_accuracy:.2%}")
c7.metric("Test F1", f"{test_f1:.2%}")
c8.metric("Test AUC", "N/A" if np.isnan(test_auc) else f"{test_auc:.2%}")

st.markdown("### Candidate leaderboard")
leaderboard_view = leaderboard.copy()
leaderboard_view["cv_accuracy"] = leaderboard_view["cv_accuracy"].round(4)
leaderboard_view["cv_f1"] = leaderboard_view["cv_f1"].round(4)
leaderboard_view["cv_balanced_accuracy"] = leaderboard_view["cv_balanced_accuracy"].round(4)
leaderboard_view["cv_auc"] = leaderboard_view["cv_auc"].round(4)
st.dataframe(
    leaderboard_view.drop(columns=["estimator"]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("### Threshold tuning on validation set")
if threshold_grid.empty:
    st.info("Threshold search could not be computed for the validation slice.")
else:
    st.dataframe(
        threshold_grid.head(10).round(4),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### Test-set strategy curve vs buy-and-hold")
if not test_sf.empty:
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=test_sf.index, y=test_sf["strategy_curve"], mode="lines", name="Strategy"))
    curve.add_trace(go.Scatter(x=test_sf.index, y=test_sf["buy_hold_curve"], mode="lines", name="Buy & Hold"))
    curve.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(curve, use_container_width=True)

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Strategy return", f"{test_metrics['total_return']:.2%}")
    sm2.metric("Buy & hold", f"{test_metrics['buy_hold_return']:.2%}")
    sm3.metric("Win rate", "N/A" if np.isnan(test_metrics["win_rate"]) else f"{test_metrics['win_rate']:.2%}")
    sm4.metric("Sharpe-like", "N/A" if np.isnan(test_metrics["sharpe_like"]) else f"{test_metrics['sharpe_like']:.2f}")
    sm5.metric("Max drawdown", f"{test_metrics['max_drawdown']:.2%}")
else:
    st.warning("No test strategy rows available after applying the current split and threshold.")

st.markdown("### Confusion matrix")
cm = confusion_matrix(y_test, test_pred_binary)
st.write(
    pd.DataFrame(
        cm,
        index=["Actual Down", "Actual Up"],
        columns=["Pred Down", "Pred Up"],
    )
)

st.markdown("### Recent predictions")
results = pd.DataFrame(index=X_test.index)
results["Close"] = feature_df.loc[X_test.index, "Close"]
results["future_return"] = fr_test
results["prob_up"] = test_proba
results["position"] = positions_from_probability(test_proba, best_threshold)
results["strategy_return"] = (
    results["position"] * results["future_return"]
    - np.abs(np.diff(np.r_[0, results["position"].to_numpy()])) * (transaction_cost_bps / 10000.0)
)
results["buy_hold"] = results["future_return"]
results = results.dropna()
results["strategy_curve"] = (1 + results["strategy_return"]).cumprod()
results["buy_hold_curve"] = (1 + results["buy_hold"]).cumprod()

show_cols = ["Close", "prob_up", "position", "strategy_return", "future_return"]
st.dataframe(results[show_cols].tail(20).round(4), use_container_width=True)

with st.expander("Classification report"):
    st.text(classification_report(y_test, test_pred_binary, zero_division=0))

st.info(
    "This dashboard is for research and paper trading only. Real-time execution should use a broker API, "
    "strong risk controls, and separate paper-trading validation before any live deployment."
)
