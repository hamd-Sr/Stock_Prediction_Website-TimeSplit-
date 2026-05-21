"""
Advanced Intraday Stock Signal Dashboard + Kite Connect Live Trading Scaffold

Default behavior:
- PAPER mode
- Model training from yfinance historical bars
- Live streaming via Kite Connect when authenticated
- Optional live order placement only after explicit enablement

Important:
- This is a trading system scaffold, not a guaranteed profitable strategy.
- Keep PAPER mode on until you have validated everything end-to-end.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

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
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier  # type: ignore

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None  # type: ignore
    HAS_LIGHTGBM = False

try:
    from kiteconnect import KiteConnect, KiteTicker  # type: ignore

    HAS_KITE = True
except Exception:
    KiteConnect = None  # type: ignore
    KiteTicker = None  # type: ignore
    HAS_KITE = False

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = Path("trading_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Advanced Intraday Stock Signal Dashboard", page_icon="📈", layout="wide")
st.title("📈 Advanced Intraday Stock Signal Dashboard")
st.caption(
    "Paper trading by default. Live execution via Kite Connect is possible after authentication and explicit enablement."
)


# -----------------------------
# Secrets / env helpers
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "..."


# -----------------------------
# General helpers
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

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing and len(out.columns) >= 5:
        first_five = list(out.columns[:5])
        pos_map = {}
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
def load_historical_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
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


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema20"] = ema20
    out["ema50"] = ema50
    out["trend_spread"] = ema20 / ema50 - 1.0
    out["trend_slope_20"] = ema20.pct_change(3)
    out["trend_slope_50"] = ema50.pct_change(5)
    out["regime_trend"] = ((out["trend_spread"] > 0) & (out["trend_slope_20"] > 0)).astype(int)
    return out


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = ensure_price_frame(raw).copy()
    idx = pd.DatetimeIndex(df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    df["ret_1"] = close.pct_change()
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["log_ret_1"] = np.log(close / close.shift(1))
    df["log_ret_3"] = np.log(close / close.shift(3))
    df["momentum_5"] = close - close.shift(5)
    df["momentum_10"] = close - close.shift(10)

    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"dist_sma_{w}"] = close / df[f"sma_{w}"] - 1.0
        df[f"volatility_{w}"] = df["ret_1"].rolling(w).std()

    df["rsi_14"] = rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

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

    df["vol_chg"] = volume.pct_change()
    df["vol_sma_20"] = volume.rolling(20).mean()
    df["vol_z"] = (volume - df["vol_sma_20"]) / volume.rolling(20).std()
    df["price_vol"] = df["ret_1"] * df["vol_chg"].fillna(0)

    vwap = intraday_vwap(df)
    df["vwap"] = vwap
    df["close_vwap"] = close / vwap - 1.0
    df["vwap_z"] = (close - vwap) / close

    df = regime_features(df)

    df["dow"] = idx.dayofweek
    df["hour"] = idx.hour
    df["minute"] = idx.minute
    df["minute_of_session"] = df.groupby(pd.Series(idx.date, index=df.index)).cumcount()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_minute"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_minute"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def build_training_frame(raw: pd.DataFrame, horizon_bars: int, move_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    df = build_feature_frame(raw).copy()
    close = df["Close"]
    df["future_return"] = close.shift(-horizon_bars) / close - 1.0
    df["target"] = (df["future_return"] > move_threshold).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude = {"future_return", "target"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df, feature_cols


# -----------------------------
# Model helpers
# -----------------------------
def make_sample_weights(y: pd.Series) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y)


def tscv_for_samples(n_samples: int) -> TimeSeriesSplit:
    if n_samples < 120:
        n_splits = 3
    elif n_samples < 240:
        n_splits = 4
    else:
        n_splits = 5
    return TimeSeriesSplit(n_splits=n_splits)


def predict_proba_positive(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.full(len(X), 0.5)


def fit_model_with_weights(model, X, y, sample_weight: np.ndarray):
    if HAS_LIGHTGBM and LGBMClassifier is not None and isinstance(model, LGBMClassifier):
        model.fit(X, y, sample_weight=sample_weight)
        return model

    if isinstance(model, HistGradientBoostingClassifier):
        model.fit(X, y, sample_weight=sample_weight)
        return model

    if hasattr(model, "steps"):
        last_step = model.steps[-1][0]
        fit_kwargs = {f"{last_step}__sample_weight": sample_weight}
        model.fit(X, y, **fit_kwargs)
        return model

    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def build_candidate_models(scale_pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {}

    if HAS_LIGHTGBM and LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=450,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    models["LogisticRegression"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )

    return models


@dataclass
class CandidateScore:
    name: str
    cv_accuracy: float
    cv_f1: float
    cv_recall_pos: float
    cv_pr_auc: float
    cv_balanced_accuracy: float
    estimator: object


def score_model_timeseries(name: str, estimator: object, X_train: pd.DataFrame, y_train: pd.Series) -> CandidateScore:
    cv = tscv_for_samples(len(X_train))
    accs, f1s, recalls, bals, aps = [], [], [], [], []

    for train_idx, valid_idx in cv.split(X_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        sample_weight = make_sample_weights(y_tr)
        model = clone(estimator)
        model = fit_model_with_weights(model, X_tr, y_tr, sample_weight)

        pred = model.predict(X_va)
        proba = predict_proba_positive(model, X_va)

        accs.append(accuracy_score(y_va, pred))
        f1s.append(f1_score(y_va, pred, zero_division=0))
        recalls.append(recall_score(y_va, pred, pos_label=1, zero_division=0))
        bals.append(balanced_accuracy_score(y_va, pred))
        aps.append(average_precision_score(y_va, proba))

    return CandidateScore(
        name=name,
        cv_accuracy=float(np.nanmean(accs)),
        cv_f1=float(np.nanmean(f1s)),
        cv_recall_pos=float(np.nanmean(recalls)),
        cv_pr_auc=float(np.nanmean(aps)),
        cv_balanced_accuracy=float(np.nanmean(bals)),
        estimator=estimator,
    )


def try_calibrated_fit(estimator: object, X: pd.DataFrame, y: pd.Series) -> object:
    cv = tscv_for_samples(len(X))
    try:
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        calibrated.fit(X, y)
        return calibrated
    except Exception:
        fallback = clone(estimator)
        sample_weight = make_sample_weights(y)
        try:
            fallback = fit_model_with_weights(fallback, X, y, sample_weight)
        except Exception:
            fallback.fit(X, y)
        return fallback


# -----------------------------
# Live signal / backtest helpers
# -----------------------------
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
    y_true: pd.Series | None = None,
    start: float = 0.50,
    stop: float = 0.65,
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

        row = {
            "threshold": t,
            "total_net_return": float((1 + net).prod() - 1),
            "mean_return": mean,
            "sharpe_like": sharpe_like,
            "trade_count": int((sf["position"] != 0).sum()),
        }

        if y_true is not None:
            pred_bin = (proba >= t).astype(int)
            row["recall_pos"] = recall_score(y_true, pred_bin, pos_label=1, zero_division=0)
            row["pr_auc"] = average_precision_score(y_true, proba)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["combined_score"] = (
        out["total_net_return"] * 0.60
        + out.get("recall_pos", pd.Series(0.0, index=out.index)) * 0.40
    )
    out = out.sort_values(["combined_score", "sharpe_like"], ascending=False).reset_index(drop=True)
    return out


def metric_cards_for_strategy(sf: pd.DataFrame) -> Dict[str, float]:
    active = sf[sf["position"] != 0].copy()
    win_rate = float((active["strategy_return"] > 0).mean()) if not active.empty else np.nan
    total_return = float((1 + sf["strategy_return"]).prod() - 1)
    buy_hold_return = float((1 + sf["buy_hold"]).prod() - 1)
    sharpe_like = (
        float(sf["strategy_return"].mean() / sf["strategy_return"].std(ddof=0))
        if sf["strategy_return"].std(ddof=0) > 0
        else np.nan
    )

    return {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "win_rate": win_rate,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown(sf["strategy_curve"]),
        "trade_count": int((sf["position"] != 0).sum()),
    }


def latest_live_signal(feature_frame: pd.DataFrame, model: object, feature_cols: List[str], threshold: float) -> Dict[str, object]:
    latest = feature_frame.iloc[[-1]].copy()
    latest_X = latest[feature_cols]
    prob_up = float(predict_proba_positive(model, latest_X)[0])

    if prob_up >= threshold:
        signal = "BUY"
    elif prob_up <= 1 - threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    regime_trend = int(latest["regime_trend"].iloc[0]) if "regime_trend" in latest.columns else 0
    if regime_trend == 0 and signal == "BUY":
        signal = "HOLD"

    return {
        "signal": signal,
        "prob_up": prob_up,
        "latest_timestamp": latest.index[-1],
        "regime_trend": regime_trend,
        "row": latest,
    }


def log_jsonl(event: dict, filename: str = "signals.jsonl") -> None:
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


# -----------------------------
# Broker adapters
# -----------------------------
@dataclass
class TradeOrder:
    timestamp: str
    mode: str
    side: str
    exchange: str
    symbol: str
    qty: int
    price: float
    signal_prob: float
    status: str
    order_id: str = ""
    reason: str = ""


class PaperBroker:
    def __init__(self, starting_cash: float = 100000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position_qty = 0
        self.avg_price = 0.0
        self.orders: List[TradeOrder] = []

    def get_position_qty(self) -> int:
        return self.position_qty

    def mark_to_market_pnl(self, ltp: float) -> float:
        return self.cash + self.position_qty * ltp - self.starting_cash

    def place_market_order(
        self,
        side: str,
        exchange: str,
        symbol: str,
        qty: int,
        price: float,
        signal_prob: float,
        reason: str = "",
    ) -> TradeOrder:
        if side == "BUY":
            cost = qty * price
            self.cash -= cost
            if self.position_qty == 0:
                self.avg_price = price
            else:
                self.avg_price = ((self.avg_price * self.position_qty) + cost) / (self.position_qty + qty)
            self.position_qty += qty
        else:  # SELL
            qty = min(qty, self.position_qty)
            proceeds = qty * price
            self.cash += proceeds
            self.position_qty -= qty
            if self.position_qty == 0:
                self.avg_price = 0.0

        order = TradeOrder(
            timestamp=datetime.now(IST).isoformat(),
            mode="PAPER",
            side=side,
            exchange=exchange,
            symbol=symbol,
            qty=qty,
            price=price,
            signal_prob=signal_prob,
            status="FILLED",
            order_id=f"PAPER-{len(self.orders)+1}",
            reason=reason,
        )
        self.orders.append(order)
        return order


class KiteBroker:
    def __init__(self, api_key: str, access_token: str):
        if not HAS_KITE or KiteConnect is None:
            raise RuntimeError("kiteconnect package is not installed.")
        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instrument_cache: Dict[Tuple[str, str], int] = {}

    def login_url(self) -> str:
        return self.kite.login_url()

    def generate_session(self, request_token: str, api_secret: str) -> dict:
        return self.kite.generate_session(request_token, api_secret)

    def lookup_instrument_token(self, exchange: str, symbol: str) -> int:
        key = (exchange, symbol)
        if key in self._instrument_cache:
            return self._instrument_cache[key]

        instruments = self.kite.instruments(exchange)
        match = next((x for x in instruments if x["tradingsymbol"] == symbol), None)
        if match is None:
            raise ValueError(f"Could not find {exchange}:{symbol} in instrument list.")
        token = int(match["instrument_token"])
        self._instrument_cache[key] = token
        return token

    def get_position_qty(self, exchange: str, symbol: str) -> int:
        pos = self.kite.positions().get("net", [])
        qty = 0
        for p in pos:
            if p.get("exchange") == exchange and p.get("tradingsymbol") == symbol:
                qty += int(p.get("quantity", 0))
        return qty

    def get_ltp(self, exchange: str, symbol: str) -> float:
        data = self.kite.ltp([f"{exchange}:{symbol}"])
        return float(data[f"{exchange}:{symbol}"]["last_price"])

    def place_market_order(
        self,
        side: str,
        exchange: str,
        symbol: str,
        qty: int,
        signal_prob: float,
        reason: str = "",
    ) -> TradeOrder:
        tx_type = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=tx_type,
            quantity=qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
            validity=self.kite.VALIDITY_DAY,
        )
        return TradeOrder(
            timestamp=datetime.now(IST).isoformat(),
            mode="LIVE",
            side=side,
            exchange=exchange,
            symbol=symbol,
            qty=qty,
            price=np.nan,
            signal_prob=signal_prob,
            status="SENT",
            order_id=str(order_id),
            reason=reason,
        )


class LiveBarBuilder:
    def __init__(self, interval_minutes: int = 5, max_bars: int = 1000):
        self.interval_minutes = max(1, int(interval_minutes))
        self.bars: deque = deque(maxlen=max_bars)
        self.current: Optional[dict] = None
        self.last_cum_volume: Optional[float] = None
        self.lock = threading.Lock()

    def _floor_ts(self, ts: datetime) -> datetime:
        minute = (ts.minute // self.interval_minutes) * self.interval_minutes
        return ts.replace(minute=minute, second=0, microsecond=0)

    def ingest_tick(self, tick: dict) -> None:
        price = tick.get("last_price") or tick.get("last_trade_price")
        if price is None:
            return

        ts = tick.get("exchange_timestamp") or tick.get("timestamp") or datetime.now(IST)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        else:
            ts = ts.astimezone(IST)

        bucket = self._floor_ts(ts)
        cum_volume = tick.get("volume_traded")
        volume_delta = 0.0
        if cum_volume is not None:
            cum_volume = float(cum_volume)
            if self.last_cum_volume is None:
                volume_delta = 0.0
            else:
                volume_delta = max(cum_volume - self.last_cum_volume, 0.0)
            self.last_cum_volume = cum_volume

        with self.lock:
            if self.current is None or self.current["start"] != bucket:
                if self.current is not None:
                    self.bars.append(self.current)
                self.current = {
                    "start": bucket,
                    "Open": float(price),
                    "High": float(price),
                    "Low": float(price),
                    "Close": float(price),
                    "Volume": float(volume_delta),
                }
            else:
                self.current["High"] = max(self.current["High"], float(price))
                self.current["Low"] = min(self.current["Low"], float(price))
                self.current["Close"] = float(price)
                self.current["Volume"] += float(volume_delta)

    def as_dataframe(self) -> pd.DataFrame:
        with self.lock:
            rows = list(self.bars)
            if self.current is not None:
                rows = rows + [self.current.copy()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("start")
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()

    def latest_price(self) -> Optional[float]:
        df = self.as_dataframe()
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])


class KiteFeedController:
    def __init__(self, broker: KiteBroker, instrument_token: int, builder: LiveBarBuilder):
        if not HAS_KITE or KiteTicker is None:
            raise RuntimeError("kiteconnect package is not installed.")
        self.broker = broker
        self.instrument_token = instrument_token
        self.builder = builder
        self.kws = KiteTicker(broker.api_key, broker.access_token)
        self.started = False
        self.last_error: str = ""

        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error

    def _on_connect(self, ws, response):
        try:
            ws.subscribe([self.instrument_token])
            mode = getattr(ws, "MODE_FULL", "full")
            ws.set_mode(mode, [self.instrument_token])
        except Exception as exc:
            self.last_error = str(exc)

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            self.builder.ingest_tick(tick)

    def _on_close(self, ws, code, reason):
        self.started = False

    def _on_error(self, ws, code, reason):
        self.last_error = f"{code}: {reason}"

    def start(self):
        if self.started:
            return
        self.started = True
        self.kws.connect(threaded=True)

    def stop(self):
        try:
            self.kws.close()
        except Exception:
            pass
        self.started = False


# -----------------------------
# Risk controls
# -----------------------------
@dataclass
class RiskConfig:
    max_qty: int = 1
    min_confidence: float = 0.55
    cooldown_minutes: int = 10
    max_trades_per_day: int = 4
    allow_shorts: bool = False


def trade_permitted(
    signal: str,
    prob_up: float,
    current_qty: int,
    risk: RiskConfig,
    last_trade_ts: Optional[pd.Timestamp],
    trades_today: int,
) -> Tuple[bool, str]:
    now = pd.Timestamp(datetime.now(IST))
    if prob_up < risk.min_confidence and signal == "BUY":
        return False, "Confidence below minimum"
    if trades_today >= risk.max_trades_per_day:
        return False, "Daily trade limit reached"
    if last_trade_ts is not None:
        elapsed = (now - last_trade_ts).total_seconds() / 60.0
        if elapsed < risk.cooldown_minutes:
            return False, "Cooldown active"

    if signal == "BUY" and current_qty > 0:
        return False, "Already long"
    if signal == "SELL" and current_qty <= 0:
        return False, "No long position to exit"

    return True, "OK"


def append_order_log(order: TradeOrder) -> None:
    path = LOG_DIR / "orders.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(order), default=str) + "\n")


# -----------------------------
# Session state init
# -----------------------------
def init_state():
    defaults = {
        "paper_broker": PaperBroker(),
        "kite_broker": None,
        "kite_feed": None,
        "live_builder": None,
        "live_feed_running": False,
        "last_trade_ts": None,
        "trades_today": 0,
        "last_executed_bar": None,
        "trade_log": pd.DataFrame(columns=["timestamp", "mode", "side", "symbol", "qty", "price", "signal_prob", "status", "order_id", "reason"]),
        "trained_model": None,
        "feature_cols": None,
        "training_frame": None,
        "leaderboard": None,
        "threshold_grid": None,
        "best_threshold": 0.60,
        "best_model_name": "",
        "latest_signal": {},
        "kite_access_token": get_secret("KITE_ACCESS_TOKEN", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Trading setup")
    mode = st.selectbox("Mode", ["PAPER", "LIVE"], index=0)
    market = st.selectbox("Market", ["NSE", "BSE"], index=0)
    ticker_input = st.text_input("Ticker", value="RELIANCE")
    period = st.selectbox("History window", ["30d", "60d", "3mo", "6mo"], index=1)
    interval = st.selectbox("Candles", ["1m", "5m", "15m", "30m"], index=1)
    horizon_bars = st.selectbox("Prediction horizon (bars)", [1, 2, 3, 5], index=1)
    move_threshold_pct = st.slider("Future move threshold (%)", 0.10, 1.50, 0.35, 0.05)
    transaction_cost_bps = st.slider("Transaction cost (bps)", 0.0, 20.0, 5.0, 0.5)

    st.markdown("---")
    st.subheader("Execution controls")
    max_qty = st.number_input("Max quantity", min_value=1, max_value=100000, value=1, step=1)
    min_confidence = st.slider("Minimum BUY confidence", 0.50, 0.90, 0.60, 0.01)
    cooldown_minutes = st.number_input("Cooldown minutes", min_value=0, max_value=240, value=10, step=1)
    max_trades_per_day = st.number_input("Max trades per day", min_value=1, max_value=100, value=4, step=1)
    allow_shorts = st.checkbox("Allow short entries", value=False)
    auto_execute = st.checkbox("Auto-execute live signals", value=False)
    st.caption("Keep this OFF until you have validated the system in paper mode.")

    st.markdown("---")
    st.subheader("Kite Connect secrets")
    api_key = get_secret("KITE_API_KEY", "")
    api_secret_default = get_secret("KITE_API_SECRET", "")
    access_token_secret = get_secret("KITE_ACCESS_TOKEN", "")

    st.write(f"API key loaded: {'yes' if api_key else 'no'}")
    st.write(f"API secret loaded: {'yes' if api_secret_default else 'no'}")
    st.write(f"Access token loaded: {'yes' if (st.session_state.kite_access_token or access_token_secret) else 'no'}")

    st.markdown("---")
    st.subheader("Session login")
    request_token = st.text_input("request_token from redirect URL", type="password")
    api_secret_input = st.text_input("API secret (only for session generation)", type="password")
    generate_session_btn = st.button("Generate session access token")
    start_feed_btn = st.button("Start live feed")
    stop_feed_btn = st.button("Stop live feed")
    refresh_btn = st.button("Refresh / retrain")
    st.markdown("---")
    st.write("Preferred model: LightGBM if installed")
    st.caption("Falls back to scikit-learn models if LightGBM is unavailable.")


symbol = normalize_ticker(ticker_input, market)
risk = RiskConfig(
    max_qty=int(max_qty),
    min_confidence=float(min_confidence),
    cooldown_minutes=int(cooldown_minutes),
    max_trades_per_day=int(max_trades_per_day),
    allow_shorts=bool(allow_shorts),
)

if refresh_btn:
    st.cache_data.clear()

# Access token resolution
resolved_access_token = st.session_state.kite_access_token or access_token_secret
kite_broker = None
if api_key and resolved_access_token and HAS_KITE:
    try:
        kite_broker = KiteBroker(api_key=api_key, access_token=resolved_access_token)
        st.session_state.kite_broker = kite_broker
    except Exception as exc:
        st.error(f"Kite broker init failed: {exc}")
        kite_broker = None

if generate_session_btn:
    if not api_key:
        st.error("Missing KITE_API_KEY.")
    elif not api_secret_input:
        st.error("Missing API secret.")
    elif not request_token:
        st.error("Paste the request_token from the redirect URL first.")
    elif not HAS_KITE:
        st.error("kiteconnect package is not installed.")
    else:
        try:
            temp_kite = KiteConnect(api_key=api_key)
            data = temp_kite.generate_session(request_token, api_secret_input)
            new_access_token = data["access_token"]
            st.session_state.kite_access_token = new_access_token
            st.success("Session generated. Copy the access token into secrets if you want persistence across reruns.")
            st.code(mask_secret(new_access_token))
            if st.session_state.kite_broker is None:
                st.session_state.kite_broker = KiteBroker(api_key=api_key, access_token=new_access_token)
        except Exception as exc:
            st.error(f"Could not generate session: {exc}")

if st.session_state.kite_broker is None and kite_broker is not None:
    st.session_state.kite_broker = kite_broker

# Live feed management
if start_feed_btn:
    if mode != "LIVE":
        st.warning("Switch Mode to LIVE before starting the live feed.")
    elif st.session_state.kite_broker is None:
        st.error("You need a valid Kite access token first.")
    else:
        try:
            st.session_state.live_builder = LiveBarBuilder(interval_minutes=int(interval.replace("m", "")))
            token = st.session_state.kite_broker.lookup_instrument_token(market, symbol.replace(".NS", "").replace(".BO", ""))
            feed = KiteFeedController(st.session_state.kite_broker, token, st.session_state.live_builder)
            feed.start()
            st.session_state.kite_feed = feed
            st.session_state.live_feed_running = True
            st.success("Live feed started.")
        except Exception as exc:
            st.error(f"Could not start live feed: {exc}")

if stop_feed_btn:
    if st.session_state.kite_feed is not None:
        st.session_state.kite_feed.stop()
    st.session_state.live_feed_running = False
    st.success("Live feed stopped.")

# auto refresh only when live feed is on
if st.session_state.live_feed_running:
    st_autorefresh(interval=5000, key="live_refresh")


# -----------------------------
# Data, model training, and scoring
# -----------------------------
st.subheader(f"Live view for {symbol}")

if mode == "LIVE" and st.session_state.kite_broker is not None and st.session_state.live_builder is not None:
    live_df = st.session_state.live_builder.as_dataframe()
else:
    live_df = pd.DataFrame()

hist_raw = load_historical_data(symbol, period, interval)
if hist_raw.empty and live_df.empty:
    st.error("No historical data and no live data available.")
    st.stop()

if not hist_raw.empty:
    hist_df, feature_cols = build_training_frame(
        hist_raw,
        horizon_bars=horizon_bars,
        move_threshold=float(move_threshold_pct) / 100.0,
    )
else:
    st.warning("Historical backfill is empty; live mode can still run if the feed is active.")
    hist_df, feature_cols = None, None

if hist_df is not None and len(hist_df) < 200:
    st.error("Not enough historical rows after feature engineering.")
    st.stop()

if hist_df is not None:
    X = hist_df[feature_cols].copy()
    y = hist_df["target"].copy()
    future_returns = hist_df["future_return"].copy()

    if y.nunique() < 2:
        st.error("Target has only one class. Raise the history window or lower the threshold.")
        st.stop()

    train_end = max(int(len(X) * 0.70), 1)
    val_end = min(max(int(len(X) * 0.85), train_end + 1), len(X) - 1)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    fr_val = future_returns.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    fr_test = future_returns.iloc[val_end:]

    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    scale_pos_weight = neg / pos

    candidates = build_candidate_models(scale_pos_weight=scale_pos_weight)
    leaderboard_rows = []
    for name, estimator in candidates.items():
        score = score_model_timeseries(name, estimator, X_train, y_train)
        leaderboard_rows.append(
            {
                "model": score.name,
                "cv_accuracy": score.cv_accuracy,
                "cv_f1": score.cv_f1,
                "cv_recall_pos": score.cv_recall_pos,
                "cv_pr_auc": score.cv_pr_auc,
                "cv_balanced_accuracy": score.cv_balanced_accuracy,
                "estimator": score.estimator,
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["cv_pr_auc", "cv_recall_pos", "cv_balanced_accuracy"], ascending=False
    ).reset_index(drop=True)
    st.session_state.leaderboard = leaderboard

    best_name = leaderboard.iloc[0]["model"]
    best_estimator = leaderboard.iloc[0]["estimator"]

    trained_model = try_calibrated_fit(best_estimator, X_train, y_train)

    val_proba = predict_proba_positive(trained_model, X_val)
    threshold_grid = threshold_search(
        fr_val,
        val_proba,
        float(transaction_cost_bps),
        y_true=y_val,
        start=0.50,
        stop=0.65,
        step=0.01,
    )

    if threshold_grid.empty:
        best_threshold = 0.55
    else:
        best_threshold = float(threshold_grid.iloc[0]["threshold"])

    val_pr_auc = average_precision_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else np.nan
    val_pred_binary = (val_proba >= best_threshold).astype(int)
    val_recall_pos = recall_score(y_val, val_pred_binary, pos_label=1, zero_division=0)
    if val_recall_pos == 0 and best_threshold > 0.60:
        best_threshold = 0.60

    test_proba = predict_proba_positive(trained_model, X_test)
    test_pred = positions_from_probability(test_proba, best_threshold)
    test_pred_binary = (test_pred == 1).astype(int)

    test_sf = strategy_frame(fr_test, test_proba, best_threshold, float(transaction_cost_bps))
    test_metrics = metric_cards_for_strategy(test_sf) if not test_sf.empty else {}

    test_accuracy = accuracy_score(y_test, test_pred_binary)
    test_f1 = f1_score(y_test, test_pred_binary, zero_division=0)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred_binary)
    test_auc = safe_auc(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else np.nan
    test_recall_pos = recall_score(y_test, test_pred_binary, pos_label=1, zero_division=0)

    final_model = try_calibrated_fit(best_estimator, X, y)
    st.session_state.trained_model = final_model
    st.session_state.feature_cols = feature_cols
    st.session_state.training_frame = hist_df
    st.session_state.best_threshold = best_threshold
    st.session_state.best_model_name = best_name
    st.session_state.threshold_grid = threshold_grid
else:
    trained_model = st.session_state.trained_model
    feature_cols = st.session_state.feature_cols
    best_threshold = st.session_state.best_threshold
    best_name = st.session_state.best_model_name
    test_sf = pd.DataFrame()
    test_metrics = {}
    test_accuracy = test_f1 = test_bal_acc = np.nan
    test_auc = test_pr_auc = test_recall_pos = np.nan
    val_pr_auc = val_recall_pos = np.nan
    leaderboard = st.session_state.leaderboard if st.session_state.leaderboard is not None else pd.DataFrame()
    threshold_grid = st.session_state.threshold_grid if st.session_state.threshold_grid is not None else pd.DataFrame()

# live prediction source
if not live_df.empty:
    live_feature_frame = build_feature_frame(live_df)
elif hist_df is not None:
    live_feature_frame = build_feature_frame(hist_raw).tail(1)
else:
    live_feature_frame = pd.DataFrame()

# If live feed has enough bars, use it; otherwise fall back to the historical last row
latest_signal = {}
if trained_model is not None and feature_cols is not None and not live_feature_frame.empty:
    available_cols = [c for c in feature_cols if c in live_feature_frame.columns]
    if len(available_cols) > 0:
        latest_row = live_feature_frame.iloc[[-1]][available_cols].copy()
        if len(latest_row.columns) == len(feature_cols):
            latest_signal = latest_live_signal(live_feature_frame, trained_model, feature_cols, best_threshold)
        else:
            # align missing cols with zeros so model can still score
            latest_row = live_feature_frame.iloc[[-1]].copy()
            for c in feature_cols:
                if c not in latest_row.columns:
                    latest_row[c] = 0.0
            latest_row = latest_row[feature_cols]
            prob_up = float(predict_proba_positive(trained_model, latest_row)[0])
            if prob_up >= best_threshold:
                signal = "BUY"
            elif prob_up <= 1 - best_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            latest_signal = {
                "signal": signal,
                "prob_up": prob_up,
                "latest_timestamp": live_feature_frame.index[-1],
                "regime_trend": int(live_feature_frame["regime_trend"].iloc[-1]) if "regime_trend" in live_feature_frame else 0,
                "row": live_feature_frame.iloc[[-1]],
            }

st.session_state.latest_signal = latest_signal


# -----------------------------
# UI
# -----------------------------
plot_source = None
if not live_df.empty:
    plot_source = ensure_price_frame(live_df).tail(300)
elif hist_raw is not None and not hist_raw.empty:
    plot_source = ensure_price_frame(hist_raw).tail(300)

if plot_source is not None and not plot_source.empty:
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
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No price chart data yet.")

current_close = float(plot_source["Close"].iloc[-1]) if plot_source is not None and not plot_source.empty else np.nan
current_prob = float(latest_signal.get("prob_up", np.nan)) if latest_signal else np.nan
current_signal = latest_signal.get("signal", "HOLD") if latest_signal else "HOLD"
current_regime = latest_signal.get("regime_trend", 0) if latest_signal else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{current_close:.2f}" if np.isfinite(current_close) else "N/A")
c2.metric("Latest up probability", f"{current_prob:.2%}" if np.isfinite(current_prob) else "N/A")
c3.metric("Live signal", current_signal)
c4.metric("Selected model", best_name if best_name else "N/A")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Chosen threshold", f"{best_threshold:.2f}")
c6.metric("Test accuracy", f"{test_accuracy:.2%}" if np.isfinite(test_accuracy) else "N/A")
c7.metric("Test PR-AUC", f"{test_pr_auc:.2%}" if np.isfinite(test_pr_auc) else "N/A")
c8.metric("Test recall (class 1)", f"{test_recall_pos:.2%}" if np.isfinite(test_recall_pos) else "N/A")

c9, c10, c11, c12 = st.columns(4)
c9.metric("Validation PR-AUC", f"{val_pr_auc:.2%}" if np.isfinite(val_pr_auc) else "N/A")
c10.metric("Validation recall (class 1)", f"{val_recall_pos:.2%}" if np.isfinite(val_recall_pos) else "N/A")
c11.metric("Test F1", f"{test_f1:.2%}" if np.isfinite(test_f1) else "N/A")
c12.metric("Test AUC", f"{test_auc:.2%}" if np.isfinite(test_auc) else "N/A")

st.markdown("### Current regime")
st.write("Trend regime" if current_regime == 1 else "Non-trend regime")

st.markdown("### Candidate leaderboard")
if leaderboard is not None and not leaderboard.empty:
    leaderboard_view = leaderboard.copy()
    for col in ["cv_accuracy", "cv_f1", "cv_recall_pos", "cv_pr_auc", "cv_balanced_accuracy"]:
        leaderboard_view[col] = leaderboard_view[col].round(4)
    st.dataframe(leaderboard_view.drop(columns=["estimator"]), use_container_width=True, hide_index=True)

st.markdown("### Threshold tuning on validation set")
if threshold_grid is not None and not threshold_grid.empty:
    st.dataframe(threshold_grid.head(10).round(4), use_container_width=True, hide_index=True)
else:
    st.info("Threshold grid not available yet.")

st.markdown("### Test-set strategy curve vs buy-and-hold")
if test_sf is not None and not test_sf.empty:
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=test_sf.index, y=test_sf["strategy_curve"], mode="lines", name="Strategy"))
    curve.add_trace(go.Scatter(x=test_sf.index, y=test_sf["buy_hold_curve"], mode="lines", name="Buy & Hold"))
    curve.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(curve, use_container_width=True)

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Strategy return", f"{test_metrics['total_return']:.2%}")
    sm2.metric("Buy & hold", f"{test_metrics['buy_hold_return']:.2%}")
    sm3.metric("Win rate", f"{test_metrics['win_rate']:.2%}" if np.isfinite(test_metrics["win_rate"]) else "N/A")
    sm4.metric("Sharpe-like", f"{test_metrics['sharpe_like']:.2f}" if np.isfinite(test_metrics["sharpe_like"]) else "N/A")
    sm5.metric("Max drawdown", f"{test_metrics['max_drawdown']:.2%}")
else:
    st.info("No test strategy rows available yet.")

st.markdown("### Confusion matrix")
if hist_df is not None:
    cm = confusion_matrix(y_test, test_pred_binary)
    st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))

st.markdown("### Recent predictions")
if hist_df is not None:
    results = pd.DataFrame(index=X_test.index)
    results["Close"] = hist_df.loc[X_test.index, "Close"]
    results["future_return"] = fr_test
    results["prob_up"] = test_proba
    results["position"] = positions_from_probability(test_proba, best_threshold)
    results["strategy_return"] = (
        results["position"] * results["future_return"]
        - np.abs(np.diff(np.r_[0, results["position"].to_numpy()])) * (float(transaction_cost_bps) / 10000.0)
    )
    results["buy_hold"] = results["future_return"]
    results = results.dropna()
    results["strategy_curve"] = (1 + results["strategy_return"]).cumprod()
    results["buy_hold_curve"] = (1 + results["buy_hold"]).cumprod()
    st.dataframe(results[["Close", "prob_up", "position", "strategy_return", "future_return"]].tail(20).round(4), use_container_width=True)

with st.expander("Classification report"):
    if hist_df is not None:
        st.text(classification_report(y_test, test_pred_binary, zero_division=0))

# -----------------------------
# Live trading execution
# -----------------------------
st.markdown("### Live / paper execution")
current_qty = 0
current_ltp = current_close if np.isfinite(current_close) else 0.0

if mode == "LIVE" and st.session_state.kite_broker is not None:
    try:
        current_qty = st.session_state.kite_broker.get_position_qty(market, symbol.replace(".NS", "").replace(".BO", ""))
    except Exception as exc:
        st.warning(f"Could not read live positions: {exc}")

elif mode == "PAPER":
    current_qty = st.session_state.paper_broker.get_position_qty()

st.write(f"Current position quantity: {current_qty}")

should_trade = False
trade_reason = ""
if latest_signal and current_signal in {"BUY", "SELL"}:
    permitted, trade_reason = trade_permitted(
        signal=current_signal,
        prob_up=float(current_prob) if np.isfinite(current_prob) else 0.0,
        current_qty=current_qty,
        risk=risk,
        last_trade_ts=st.session_state.last_trade_ts,
        trades_today=st.session_state.trades_today,
    )
    should_trade = permitted

st.write(f"Trade gate: {'OPEN' if should_trade else 'BLOCKED'}")
st.caption(trade_reason if trade_reason else "Waiting for the next decision.")

manual_trade_btn = st.button("Execute current signal once")
execute_now = auto_execute or manual_trade_btn

if execute_now and latest_signal and should_trade:
    ts = pd.Timestamp(datetime.now(IST))
    side = current_signal
    qty = risk.max_qty
    try:
        if mode == "PAPER":
            order = st.session_state.paper_broker.place_market_order(
                side=side,
                exchange=market,
                symbol=symbol.replace(".NS", "").replace(".BO", ""),
                qty=qty,
                price=float(current_ltp),
                signal_prob=float(current_prob),
                reason="Auto-executed live signal" if auto_execute else "Manual signal execution",
            )
        else:
            if st.session_state.kite_broker is None:
                raise RuntimeError("Kite broker is not ready.")
            order = st.session_state.kite_broker.place_market_order(
                side=side,
                exchange=market,
                symbol=symbol.replace(".NS", "").replace(".BO", ""),
                qty=qty,
                signal_prob=float(current_prob),
                reason="Auto-executed live signal" if auto_execute else "Manual signal execution",
            )

        st.session_state.last_trade_ts = ts
        st.session_state.trades_today += 1
        st.session_state.last_executed_bar = latest_signal.get("latest_timestamp")
        append_order_log(order)
        log_jsonl({"event": "order", **asdict(order)})
        trade_df = pd.DataFrame([asdict(order)])
        st.session_state.trade_log = pd.concat([st.session_state.trade_log, trade_df], ignore_index=True)

        if mode == "PAPER":
            st.success(f"PAPER order filled: {side} {qty}")
        else:
            st.success(f"LIVE order sent: {side} {qty}")
    except Exception as exc:
        st.error(f"Order execution failed: {exc}")

if not st.session_state.trade_log.empty:
    st.markdown("### Trade log")
    st.dataframe(st.session_state.trade_log.tail(20), use_container_width=True, hide_index=True)

if mode == "LIVE":
    st.warning(
        "Live trading is enabled only when you have a valid access token and you explicitly turn on auto-execution or press the execute button."
    )

st.info(
    "This is a production-style scaffold with broker connectivity, but you still need to validate it in PAPER mode, "
    "test slippage, and confirm your risk rules before going live."
)
