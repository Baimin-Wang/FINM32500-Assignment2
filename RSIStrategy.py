# RSIStrategy.py
from __future__ import annotations

import pandas as pd
from Strategy import Strategy


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy.

    Signal Logic:
      - RSI = 100 - (100 / (1 + RS))
      - RS = Average Gain / Average Loss over N periods
      - Buy (1) when RSI < 30 (oversold)
      - Sell (-1) when RSI > 70 (overbought) and we hold a position
      - Hold (0) otherwise
    """

    def __init__(
        self,
        data_dir: str,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        **kwargs
    ) -> None:
        """
        Args:
            data_dir: Directory containing per-ticker parquet files
            rsi_period: RSI calculation period (default: 14)
            oversold_threshold: RSI level below which to buy (default: 30)
            overbought_threshold: RSI level above which to sell (default: 70)
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(data_dir=data_dir, **kwargs)
        self.rsi_period = int(rsi_period)
        self.oversold_threshold = float(oversold_threshold)
        self.overbought_threshold = float(overbought_threshold)

    def _calculate_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for each ticker in the price panel.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape, containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving average of gains and losses
        avg_gains = gains.ewm(span=self.rsi_period, adjust=False, min_periods=self.rsi_period).mean()
        avg_losses = losses.ewm(span=self.rsi_period, adjust=False, min_periods=self.rsi_period).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gains / avg_losses

        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RSI levels.

        Args:
            prices: DataFrame with index=dates, columns=tickers, values=adj_close

        Returns:
            DataFrame with same shape, containing 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate RSI
        rsi = self._calculate_rsi(prices)

        # Initialize signals to 0 (hold)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Buy signal: RSI < oversold threshold (oversold condition)
        signals[rsi < self.oversold_threshold] = 1

        # Sell signal: RSI > overbought threshold (overbought condition)
        prev_signals = signals.shift(1).fillna(0)
        signals[(rsi > self.overbought_threshold) & (prev_signals == 1)] = -1

        return signals
