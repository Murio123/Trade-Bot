"""Current-market spot facts for the product side (S4A).

This package **describes what is observably true right now** and does nothing
else. It contains no model, no score, no probability and no combination of
fields, because S3 measured six ranking rules against `CLEAN_2X(180d)` and
none of them ranked better than the eligible universe itself
(`reports/s3/S3_RESULT.md`, verdict `S3_NO_SIGNAL`). A screener built on top
of that result may prioritise a user's attention; it may not claim to predict.

Direction of dependency, deliberately one-way:

    tools/spot_snapshot.py   (research side)  -> writes the snapshot file
    spot_market/             (product side)   -> reads it, and only reads it
    bot/spot_screener.py     (adapter)        -> the one bot module that may
                                                 import this package

Nothing here imports `spot`, `analyzer`, `database`, `signal_engine`,
`pipeline`, `risk`, `scheduler` or `bot`, so the research package stays
research and the runtime stays free of it. Nothing here reaches the network,
opens a position, or reads futures state of any kind.
"""
from spot_market.snapshot import (BTC_SYMBOL, CoinFacts, MarketSnapshot,
                                  SnapshotUnavailable, load_snapshot)
from spot_market.views import VIEWS, VIEWS_BY_ID, View, shortlist

__all__ = ["BTC_SYMBOL", "CoinFacts", "MarketSnapshot", "SnapshotUnavailable",
           "load_snapshot", "VIEWS", "VIEWS_BY_ID", "View", "shortlist"]
