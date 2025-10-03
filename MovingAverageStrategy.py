# MovingAverageStrategy.py
from __future__ import annotations

import pandas as pd
from Strategy import Strategy


class MovingAverageStrategy(Strategy):
    """
    Moving Average Crossover Strategy.

    Signal Logic:
      - Buy (1) when 20-day MA > 50-day MA
      - Sell (-1) when 20-day MA <= 50-day MA and we hold a position
      - Hold (0) otherwise
    """

    def __init__(
        self,
        data_dir: str,
        short_window: int = 20,
        long_window: int = 50,
        **kwargs
    ) -> None:
        """
        Args:
            data_dir: Directory containing per-ticker parquet files
            short_window: Short-term moving average period (default: 20)
            long_window: Long-term moving average period (default: 50)
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(data_dir=data_dir, **kwargs)
        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on moving average crossover.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape, containing 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate short and long moving averages
        ma_short = prices.rolling(window=self.short_window, min_periods=self.short_window).mean()
        ma_long = prices.rolling(window=self.long_window, min_periods=self.long_window).mean()

        # Initialize signals to 0 (hold)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Buy signal: short MA > long MA
        signals[ma_short > ma_long] = 1

        # Sell signal: short MA <= long MA
        # We'll mark as -1 when previously we had a buy signal
        prev_signals = signals.shift(1).fillna(0)
        signals[(ma_short <= ma_long) & (prev_signals == 1)] = -1

        return signals
