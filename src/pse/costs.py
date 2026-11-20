"""Philippine trading costs.

Rates follow the PSE investor fee schedule (pse.com.ph, Investing at PSE): broker
commission 0.25% plus 12% VAT, PSE transaction fee 0.005% plus VAT, SCCP fee 0.01%,
SRC fee 0.005%, and a 0.1% stock transaction tax on sales (RA 12214, effective 1 July 2025).
Tick sizes follow the PSE board lot and price fluctuation table. Half the quoted spread is
approximated by half a tick, which is a LOWER bound for thin stocks.
"""
from dataclasses import dataclass
import numpy as np

TICK_LOWER = np.array([0.0001, 0.01, 0.05, 0.25, 0.5, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
TICK_SIZE = np.array([0.0001, 0.001, 0.001, 0.005, 0.01, 0.01, 0.02, 0.05, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])


def tick_size(price):
    price = np.asarray(price, dtype=float)
    idx = np.clip(np.searchsorted(TICK_LOWER, price, side="right") - 1, 0, None)
    return TICK_SIZE[idx]


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0025
    vat: float = 0.12
    pse_fee: float = 0.00005
    sccp_fee: float = 0.0001
    src_fee: float = 0.00005
    stt_sell: float = 0.001
    spread_mult: float = 1.0      # quoted spread as a multiple of one tick (0 turns spreads off)
    extra_bps: float = 0.0        # extra slippage per side, in basis points

    def fee_rate(self, side):
        base = (self.commission + self.pse_fee) * (1 + self.vat) + self.sccp_fee + self.src_fee
        return base + (self.stt_sell if side == "sell" else 0.0)

    def half_spread(self, price):
        price = np.asarray(price, dtype=float)
        return self.spread_mult * tick_size(price) / (2 * price) + self.extra_bps / 1e4

    def net_return(self, gross_log, p_in, p_out):
        """Simple net return of a long trade: buy at ask plus fees, sell at bid minus fees and tax."""
        h_in, h_out = self.half_spread(p_in), self.half_spread(p_out)
        factor = (np.exp(gross_log) * (1 - h_out) / (1 + h_in)
                  * (1 - self.fee_rate("sell")) / (1 + self.fee_rate("buy")))
        return factor - 1.0
