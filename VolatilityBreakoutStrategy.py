# VolatilityBreakoutStrategy.py
from __future__ import annotations

import pandas as pd
from Strategy import Strategy


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy.

    Signal Logic:
      - Buy (1) when daily return > rolling 20-day standard deviation (volatility breakout)
      - Sell (-1) when holding and daily return < -rolling 20-day standard deviation
      - Hold (0) otherwise
    """

    def __init__(
        self,
        data_dir: str,
        lookback_window: int = 20,
        **kwargs
    ) -> None:
        """
        Args:
            data_dir: Directory containing per-ticker parquet files
            lookback_window: Rolling window for volatility calculation (default: 20)
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(data_dir=data_dir, **kwargs)
        self.lookback_window = int(lookback_window)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on volatility breakout.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape, containing 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate daily returns
        returns = prices.pct_change()

        # Calculate rolling volatility (standard deviation of returns)
        rolling_std = returns.rolling(window=self.lookback_window, min_periods=self.lookback_window).std()

        # Initialize signals to 0 (hold)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Buy signal: daily return > rolling std dev (breakout upward)
        signals[returns > rolling_std] = 1

        # Sell signal: daily return < -rolling std dev (breakout downward)
        prev_signals = signals.shift(1).fillna(0)
        signals[(returns < -rolling_std) & (prev_signals == 1)] = -1

        return signals
