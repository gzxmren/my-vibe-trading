"""Tencent Finance loader: free, no-auth data for A-shares.

Uses the public Tencent Finance HTTP API. No API token required.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import requests

from backtest.loaders.base import validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1D": "day",
    "1W": "week",
    "1M": "month",
}

_OFFSET_MAP = {
    "day": 640,
    "week": 200,
    "month": 200,
}


@register
class DataLoader:
    """Tencent Finance A-share loader (free, no auth)."""

    name = "tencent"
    markets = {"a_share"}
    requires_auth = False

    def is_available(self) -> bool:
        """Always available as it uses requests."""
        return True

    def __init__(self) -> None:
        pass

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data via Tencent Finance.

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

        freq = _INTERVAL_MAP.get(interval)
        if freq is None:
            logger.warning("tencent does not support interval %s", interval)
            return {}

        offset = _OFFSET_MAP.get(freq, 640)
        result: Dict[str, pd.DataFrame] = {}

        for code in codes:
            try:
                df = self._fetch_one(code, freq, offset)
                if df is not None and not df.empty:
                    # Filter by date range
                    df = df.loc[start_date:end_date]
                    if not df.empty:
                        result[code] = df
            except Exception as exc:
                logger.warning("tencent failed for %s: %s", code, exc)
        return result

    def _fetch_one(self, code: str, freq: str, offset: int) -> Optional[pd.DataFrame]:
        """Fetch a single symbol."""
        digits, _, suffix = code.partition(".")
        prefix = suffix.lower() if suffix else ("sh" if digits.startswith(("60", "68")) else "sz")
        symbol = f"{prefix}{digits}"

        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},{freq},,,{offset},qfq"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or symbol not in data["data"]:
            return None

        key = f"qfq{freq}"
        if key not in data["data"][symbol]:
            # Try non-adjusted if qfq is missing
            key = freq
            if key not in data["data"][symbol]:
                return None

        raw_data = data["data"][symbol][key]
        if not raw_data:
            return None

        # Tencent format: [date, open, close, high, low, volume]
        df = pd.DataFrame(raw_data)
        df = df.iloc[:, :6]
        df.columns = ["trade_date", "open", "close", "high", "low", "volume"]

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Volume is in "手" (lots), convert to shares (股)
        df["volume"] = df["volume"] * 100

        # Reorder to standard OHLCV
        df = df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )
        return df
