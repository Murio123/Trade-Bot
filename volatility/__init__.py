"""C4.4: report-only 48h volatility RANKING.

What this package is allowed to say: how the next 48 hours are expected to
rank against history — LOW / NORMAL / HIGH plus a percentile.
Nothing else. It carries no direction, no entry, no stop, no target, no
size, and it is not a trading signal. C4.3c rejected the underlying model as
a point predictor of volatility MAGNITUDE and that rejection stands.

The primary ranker is deliberately NOT the Ridge model. C4.3d/C4.3e measured
Ridge's honest ranking edge over an inverted trailing mean at about +0.13
Spearman — real (block-bootstrap CI lower bound ~+0.05, permutation
p <= 0.0036) but worth only ~1.5pp of quartile accuracy, and confirming it
forward would need ~2830 matured forecasts. So the trailing baseline ships
as v1 because it needs no features at all, and Ridge rides along as a
recorded shadow until the ledger can settle the question.

Import direction: nothing here may import tools.forecast_platform. Feature
computation belongs to the offline producer (tools/volatility_forecast_run.py);
the bot only ever reads the ledger this package writes.
"""
