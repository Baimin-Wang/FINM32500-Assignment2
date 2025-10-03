# Strategy.py
from __future__ import annotations

import os
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    shares: int
    px: float
    notional: float


class Strategy(ABC):
    """
    Base class for trading strategies.

    Works with a folder of per-ticker parquet files:
      - Required column: 'adj_close'
      - Optional column: 'volume' (if present, enforces participation cap)

    Constraints:
      - initial_cash = $1,000,000
      - Only 1 share per buy signal (shares_per_signal = 1)
      - Act on previous day's signal (1-day lag)
      - No short positions allowed
      - Skip trades if insufficient cash
      - Optional max_participation = 0.05 (5% of daily volume)

    Output tracks:
      - holdings_value: Market value of all stock holdings
      - cash: Available cash balance
      - total_value: holdings_value + cash
    """

    def __init__(
        self,
        data_dir: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_cash: float = 1_000_000.0,
        shares_per_signal: int = 1,
        max_participation: float = 0.05,
        tickers: Optional[Iterable[str]] = None,
        min_history_days: int = 252,
        sort_tickers: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.initial_cash = float(initial_cash)
        self.shares_per_signal = int(shares_per_signal)
        self.max_participation = float(max_participation)
        self.user_tickers = set(t.upper() for t in tickers) if tickers else None
        self.min_history_days = int(min_history_days)
        self.sort_tickers = bool(sort_tickers)
        self.start = pd.to_datetime(start).tz_localize(None) if start else None
        self.end = pd.to_datetime(end).tz_localize(None) if end else None

        self._prices: pd.DataFrame = pd.DataFrame()
        self._volumes: Optional[pd.DataFrame] = None
        self._holdings: Dict[str, int] = {}
        self._cash: float = self.initial_cash
        self._trades: List[Trade] = []
        self._runtime_sec: float = 0.0

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for all tickers.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape as prices, containing 1 (buy), 0 (hold), -1 (sell)
        """
        pass

    def run(self) -> pd.DataFrame:
        """
        Executes the strategy:
        - loads price data into a (dates x tickers) panel
        - generates signals using the strategy-specific logic
        - executes trades based on signals
        - returns a DataFrame with columns: ['holdings_value', 'cash', 'total_value']
        """
        t0 = time.perf_counter()
        self._load_panel()

        if self._prices.empty:
            raise RuntimeError("No price panel loaded. Check data_dir and filters.")

        # Generate signals for all dates and tickers
        signals = self.generate_signals(self._prices)

        # Execute strategy day by day, acting on previous day's signal
        prev_signals = signals.shift(1).fillna(0)

        for i, date in enumerate(self._prices.index):
            px = self._prices.loc[date]
            sig = prev_signals.loc[date]  # Act on previous day's signal

            vol = None
            if self._volumes is not None:
                vol = self._volumes.loc[date]

            tickers = list(self._prices.columns)
            if self.sort_tickers:
                tickers.sort()

            for tkr in tickers:
                p = px.get(tkr, float("nan"))
                signal = sig.get(tkr, 0)

                if not pd.notna(p) or p <= 0:
                    continue

                current_position = self._holdings.get(tkr, 0)

                # Buy signal
                if signal == 1 and current_position == 0:
                    target_shares = self.shares_per_signal

                    # Participation cap (if volume available)
                    if vol is not None and pd.notna(vol.get(tkr, float("nan"))):
                        cap_shares = int(math.floor(self.max_participation * float(vol[tkr])))
                        if cap_shares >= 0:
                            target_shares = min(target_shares, cap_shares)

                    if target_shares <= 0:
                        continue

                    cost = target_shares * p
                    if cost <= self._cash:
                        self._execute_buy(date, tkr, target_shares, p)

                # Sell signal - sell all current holdings
                elif signal == -1 and current_position > 0:
                    self._execute_sell(date, tkr, current_position, p)

        # Build portfolio value series
        holdings_vec = pd.Series(self._holdings, index=self._prices.columns).fillna(0).astype(int)
        holdings_value = (self._prices * holdings_vec).sum(axis=1)
        cash_series = pd.Series(self._cash, index=self._prices.index)

        # Update cash over time based on trades
        cash_timeline = pd.Series(self.initial_cash, index=self._prices.index)
        for trade in self._trades:
            trade_date = trade.date
            if trade.shares > 0:  # Buy
                cash_timeline.loc[trade_date:] -= trade.notional
            else:  # Sell
                cash_timeline.loc[trade_date:] += abs(trade.notional)

        nav = pd.DataFrame({
            "holdings_value": holdings_value,
            "cash": cash_timeline,
        })
        nav["total_value"] = nav["holdings_value"] + nav["cash"]

        self._runtime_sec = time.perf_counter() - t0
        return nav

    def trades(self) -> pd.DataFrame:
        """Trade blotter (one row per executed trade)."""
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

    def _execute_buy(self, date: pd.Timestamp, ticker: str, shares: int, px: float) -> None:
        notional = shares * px
        self._cash -= notional
        self._holdings[ticker] = self._holdings.get(ticker, 0) + shares
        self._trades.append(Trade(date=date, ticker=ticker, shares=shares, px=px, notional=notional))

    def _execute_sell(self, date: pd.Timestamp, ticker: str, shares: int, px: float) -> None:
        notional = shares * px
        self._cash += notional
        self._holdings[ticker] = self._holdings.get(ticker, 0) - shares
        self._trades.append(Trade(date=date, ticker=ticker, shares=-shares, px=px, notional=-notional))

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
            vols = vols.reindex(prices.index)

        # Forward fill missing prices (up to 5 days)
        prices = prices.fillna(method='ffill', limit=5)

        self._prices = prices
        self._volumes = vols
