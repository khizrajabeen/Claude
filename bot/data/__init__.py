"""Market data providers, one per asset class, behind a single router."""

from bot.data.base import DataProvider, ProviderError
from bot.data.alpaca import AlpacaProvider
from bot.data.crypto import CryptoProvider
from bot.data.equity import EquityProvider
from bot.data.router import DataRouter

__all__ = ["DataProvider", "ProviderError", "AlpacaProvider", "CryptoProvider", "EquityProvider",
           "DataRouter"]
