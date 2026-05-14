"""Mootdx loader: fast, no-auth data for A-shares via Tdx protocol.

Mootdx (https://github.com/mootdx/mootdx) provides direct access to Tdx
行情 servers. No API token required.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1D": 9,
    "1W": 5,
    "1M": 6,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "60m": 3,
}

_OFFSET_MAP = {
    9: 800,  # ~3.3 years
    5: 200,  # ~4 years
    6: 200,  # ~16 years
    0: 800,
    1: 800,
    2: 800,
    3: 800,
}


@register
class DataLoader:
    """Mootdx A-share loader (fast, no auth)."""

    name = "mootdx"
    markets = {"a_share"}
    requires_auth = False

    def is_available(self) -> bool:
        """Available if mootdx is installed."""
        try:
            from mootdx.quotes import Quotes  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mootdx.quotes import Quotes
            self._client = Quotes.factory(market="std", timeout=10)
        return self._client

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data via Mootdx.

        Args:
            codes: Symbol list (e.g. 600089.SH).
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD.
            interval: Bar size.
            fields: Ignored.

        Returns:
            Mapping symbol -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        frequency = _INTERVAL_MAP.get(interval)
        if frequency is None:
            logger.warning("mootdx does not support interval %s", interval)
            return {}

        offset = _OFFSET_MAP.get(frequency, 800)
        client = self._get_client()

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                symbol = code.split(".")[0]
                df = client.bars(symbol=symbol, frequency=frequency, offset=offset)
                if df is not None and not df.empty:
                    df = self._normalize(df)
                    # Filter by date range
                    df = df.loc[start_date:end_date]
                    if not df.empty:
                        result[code] = df
            except Exception as exc:
                logger.warning("mootdx failed for %s: %s", code, exc)
        return result

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Mootdx DataFrame to standard OHLCV schema.

        Mootdx columns: datetime, open, high, low, close, vol, amount, volume
        Note: volume is in shares (股数).
        """
        # If 'volume' already exists, use it. If not, rename 'vol'.
        if "volume" not in df.columns and "vol" in df.columns:
            df = df.rename(columns={"vol": "volume"})
        
        # Use index if it's already datetime, or convert 'datetime' column
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["trade_date"] = pd.to_datetime(df["datetime"])
                df = df.set_index("trade_date")
            else:
                # Fallback if neither index nor column is datetime
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Final selection of columns
        cols = ["open", "high", "low", "close", "volume"]
        # Ensure all columns exist and are numeric
        for col in cols:
            if col in df.columns:
                # If there are duplicate columns (e.g. volume and vol renamed to volume),
                # select only the first one to avoid errors.
                if isinstance(df[col], pd.DataFrame):
                    df[col] = df[col].iloc[:, 0]
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        df = df[cols].dropna(subset=["open", "high", "low", "close"])
        return df
