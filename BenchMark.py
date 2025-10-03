# BenchmarkStrategy.py
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    shares: int
    px: float
    notional: float


class BenchmarkStrategy:
    """
    Benchmark: buy on day 1, then do nothing.
    Works with a folder of per-ticker parquet files:
      - Required column: 'adj_close'
      - Optional column: 'volume' (if present, enforces participation cap)

    Two sizing modes (pick one):
      1) fixed_shares_per_ticker = X
      2) fixed_dollar_per_ticker = Y  (recommended for avoiding impact)

    Constraints:
      - initial_cash = 1_000_000 by default
      - skip if insufficient cash
      - no leverage, no shorting
      - optional max_participation = 0.05 (5% of day-1 volume), applied only if 'volume' exists
    """

    def __init__(
        self,
        data_dir: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_cash: float = 1_000_000.0,
        fixed_shares_per_ticker: Optional[int] = None,
        fixed_dollar_per_ticker: Optional[float] = None,
        max_participation: float = 0.05,  # 1–5% is stealth; we default to 5%
        tickers: Optional[Iterable[str]] = None,
        min_history_days: int = 252,  # drop very sparse series
        sort_tickers: bool = True,    # deterministic fill order
    ) -> None:
        if (fixed_shares_per_ticker is None) == (fixed_dollar_per_ticker is None):
            raise ValueError("Choose exactly one sizing mode: fixed_shares_per_ticker OR fixed_dollar_per_ticker")
        self.data_dir = data_dir
        self.initial_cash = float(initial_cash)
        self.fixed_shares = fixed_shares_per_ticker
        self.fixed_dollars = fixed_dollar_per_ticker
        self.max_participation = float(max_participation)
        self.user_tickers = set(t.upper() for t in tickers) if tickers else None
        self.min_history_days = int(min_history_days)
        self.sort_tickers = bool(sort_tickers)
        self.start = pd.to_datetime(start).tz_localize(None) if start else None
        self.end = pd.to_datetime(end).tz_localize(None) if end else None

        self._prices: pd.DataFrame = pd.DataFrame()   # adj_close panel
        self._volumes: Optional[pd.DataFrame] = None  # optional volume panel (same index/columns)
        self._holdings: Dict[str, int] = {}
        self._cash: float = self.initial_cash
        self._trades: List[Trade] = []
        self._runtime_sec: float = 0.0

    # ----------------- Public API -----------------

    def run(self) -> pd.DataFrame:
        """
        Executes the benchmark:
        - loads price data into a (dates x tickers) panel
        - on first date, buys according to sizing mode and constraints
        - returns a DataFrame with columns: ['holdings_value', 'cash', 'total_value']
        """
        t0 = time.perf_counter()
        self._load_panel()

        if self._prices.empty:
            raise RuntimeError("No price panel loaded. Check data_dir and filters.")

        first_day = self._prices.index[0]
        px0 = self._prices.loc[first_day]

        vol0 = None
        if self._volumes is not None:
            # Some tickers may have NA volume on day 1 → treat as unavailable (no cap)
            vol0 = self._volumes.loc[first_day]

        tickers = list(self._prices.columns)
        if self.sort_tickers:
            tickers.sort()

        # Place orders in order
        for tkr in tickers:
            p = px0.get(tkr, float("nan"))
            if not pd.notna(p) or p <= 0:
                continue

            # Determine target shares
            if self.fixed_shares is not None:
                target_shares = int(self.fixed_shares)
            else:
                # fixed_dollars per ticker → floor(dollars/price)
                target_shares = int(math.floor(self.fixed_dollars / p))

            if target_shares <= 0:
                continue

            # Participation cap (if day-1 volume is available)
            if vol0 is not None and pd.notna(vol0.get(tkr, float("nan"))):
                cap_shares = int(math.floor(self.max_participation * float(vol0[tkr])))
                if cap_shares >= 0:
                    target_shares = min(target_shares, cap_shares)

            if target_shares <= 0:
                continue

            cost = target_shares * p
            if cost <= self._cash:
                self._execute_buy(first_day, tkr, target_shares, p)
            # else: insufficient cash → skip

        # Build portfolio value series over the entire horizon
        holdings_vec = pd.Series(self._holdings, index=self._prices.columns).fillna(0).astype(int)
        holdings_value = (self._prices * holdings_vec).sum(axis=1)
        cash_series = pd.Series(self._cash, index=self._prices.index)
        nav = pd.DataFrame(
            {
                "holdings_value": holdings_value,
                "cash": cash_series,
            }
        )
        nav["total_value"] = nav["holdings_value"] + nav["cash"]

        self._runtime_sec = time.perf_counter() - t0
        return nav

    def trades(self) -> pd.DataFrame:
        """Trade blotter (one row per executed buy)."""
        if not self._trades:
            return pd.DataFrame(columns=["date", "ticker", "shares", "px", "notional"])
        return pd.DataFrame([t.__dict__ for t in self._trades]).sort_values(["date", "ticker"])

    def summary(self) -> Dict[str, float]:
        """Quick timing + final state."""
        last_value = float("nan")
        if not self._prices.empty:
            last_day = self._prices.index[-1]
            holdings_vec = pd.Series(self._holdings, index=self._prices.columns).fillna(0).astype(int)
            last_value = float((self._prices.loc[last_day] * holdings_vec).sum() + self._cash)
        return {
            "initial_cash": self.initial_cash,
            "final_cash": self._cash,
            "final_total_value": last_value,
            "num_trades": len(self._trades),
            "runtime_seconds": self._runtime_sec,
        }

    # ----------------- Internals ------------------

    def _execute_buy(self, date: pd.Timestamp, ticker: str, shares: int, px: float) -> None:
        notional = shares * px
        self._cash -= notional
        self._holdings[ticker] = self._holdings.get(ticker, 0) + shares
        self._trades.append(Trade(date=date, ticker=ticker, shares=shares, px=px, notional=notional))

    def _load_panel(self) -> None:
        """
        Loads per-ticker parquet files from self.data_dir and builds:
            prices panel: index=date, columns=tickers, values=adj_close
            volumes panel (optional): same shape if 'volume' column exists
        Applies date window [start, end] and min_history_days.
        """
        frames_prices: Dict[str, pd.Series] = {}
        frames_vols: Dict[str, pd.Series] = {}

        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".parquet")]
        if self.user_tickers:
            files = [f for f in files if os.path.splitext(f)[0].upper() in self.user_tickers]

        for f in files:
            tkr = os.path.splitext(f)[0].upper()
            path = os.path.join(self.data_dir, f)
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "adj_close" not in df.columns:
                # Allow 'price' as a fallback name if the user saved that earlier
                if "price" in df.columns:
                    df = df.rename(columns={"price": "adj_close"})
                else:
                    continue

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Optional volume support
            if "volume" in df.columns:
                frames_vols[tkr] = df["volume"]

            # apply date window
            if self.start is not None:
                df = df[df.index >= self.start]
            if self.end is not None:
                df = df[df.index <= self.end]

            # min history filter
            if len(df) < self.min_history_days:
                continue

            frames_prices[tkr] = df["adj_close"]

        if not frames_prices:
            self._prices = pd.DataFrame()
            self._volumes = None
            return

        prices = pd.DataFrame(frames_prices).dropna(how="all")
        # Align volumes (only for tickers that survived in prices)
        vols = None
        if frames_vols:
            vols = pd.DataFrame({k: v for k, v in frames_vols.items() if k in prices.columns})
            # Reindex volumes to prices index
            vols = vols.reindex(prices.index)

        # Drop columns with NA on the first day (cannot trade on day 1)
        first_day = prices.index[0]
        ok_cols = [c for c in prices.columns if pd.notna(prices.loc[first_day, c])]
        prices = prices[ok_cols]
        if vols is not None:
            vols = vols[ok_cols]

        self._prices = prices
        self._volumes = vols
