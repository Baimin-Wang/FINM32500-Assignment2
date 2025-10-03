# MACDStrategy.py
from __future__ import annotations

import pandas as pd
from Strategy import Strategy


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Signal Logic:
      - MACD Line = 12-day EMA - 26-day EMA
      - Signal Line = 9-day EMA of MACD Line
      - Buy (1) when MACD line crosses above signal line
      - Sell (-1) when MACD line crosses below signal line and we hold a position
      - Hold (0) otherwise
    """

    def __init__(
        self,
        data_dir: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs
    ) -> None:
        """
        Args:
            data_dir: Directory containing per-ticker parquet files
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(data_dir=data_dir, **kwargs)
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.signal_period = int(signal_period)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on MACD crossover.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape, containing 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast_period, adjust=False, min_periods=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period, adjust=False, min_periods=self.slow_period).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False, min_periods=self.signal_period).mean()

        # Initialize signals to 0 (hold)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Detect crossovers
        # Buy signal: MACD crosses above signal line
        macd_above = macd_line > signal_line
        macd_above_prev = macd_above.shift(1).fillna(False)
        bullish_crossover = macd_above & ~macd_above_prev

        signals[bullish_crossover] = 1

        # Sell signal: MACD crosses below signal line
        macd_below = macd_line < signal_line
        macd_below_prev = macd_below.shift(1).fillna(False)
        bearish_crossover = macd_below & ~macd_below_prev

        # Only sell if we previously had a buy signal
        prev_signals = signals.shift(1).fillna(0)
        signals[bearish_crossover & (prev_signals == 1)] = -1

        return signals
