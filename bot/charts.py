"""Chart rendering: candlestick PNGs for signals and levels.

Design follows the dataviz method: dark surface, a CVD-validated palette
(up/down pair ΔE 26 protan — safe), thin marks, recessive grid, selective
direct labels on reference lines (never color alone), one price axis with a
separate volume panel. Rendering is pure CPU — callers run it via
asyncio.to_thread.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config

# Validated palette (scripts/validate_palette.js, dark surface): ALL PASS.
SURFACE = "#1a1a19"
PANEL = "#1f1f1e"
UP = "#26A69A"          # bullish candles / long levels / bullish zones
DOWN = "#EF5350"        # bearish candles / stop / bearish zones
FVG_C = "#9A7BD8"       # FVG zones
ACCENT = "#B8862F"      # equilibrium / VWAP accents
GRID = "#33333a"
INK = "#d7d7d3"         # primary text
INK_MUTED = "#8a8a85"


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def _new_figure(df, with_volume: bool = True):
    if with_volume:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(11, 6.5), dpi=110, sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
        axv = None
    fig.patch.set_facecolor(SURFACE)
    for a in filter(None, (ax, axv)):
        a.set_facecolor(PANEL)
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
        a.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(GRID)
    return fig, ax, axv


def _draw_candles(ax, axv, df) -> None:
    x = range(len(df))
    for i, (_, c) in enumerate(df.iterrows()):
        o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        color = UP if cl >= o else DOWN
        # thin wick + slim body (thin marks per the mark spec)
        ax.vlines(i, l, h, color=color, linewidth=0.7, alpha=0.9)
        body_low, body_high = min(o, cl), max(o, cl)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64,
                               max(body_high - body_low, (h - l) * 0.001 or 1e-9),
                               facecolor=color, edgecolor="none", alpha=0.95))
        if axv is not None:
            axv.vlines(i, 0, float(c["volume"]), color=color, linewidth=1.6, alpha=0.45)
    ax.set_xlim(-1, len(df) + 14)  # right margin for direct labels

    # Sparse date ticks instead of bar indices.
    if "open_time" in df.columns:
        n = len(df)
        ticks = list(range(0, n, max(n // 6, 1)))
        bottom = axv if axv is not None else ax
        bottom.set_xticks(ticks)
        bottom.set_xticklabels(
            [df["open_time"].iloc[t].strftime("%d.%m %H:%M") for t in ticks],
            fontsize=7)


def _hline(ax, y: float, color: str, label: str, n: int,
           style: str = "--", lw: float = 1.1) -> None:
    ax.axhline(y, color=color, linestyle=style, linewidth=lw, alpha=0.85)
    ax.annotate(f"{label} {_fmt(y)}", xy=(n + 0.5, y), xytext=(4, 0),
                textcoords="offset points", color=color, fontsize=7.5,
                va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                          edgecolor="none", alpha=0.9))


def _zone(ax, low: float, high: float, n: int, color: str, label: str) -> None:
    ax.add_patch(Rectangle((-1, low), n + 15, high - low,
                           facecolor=color, edgecolor="none", alpha=0.13))
    ax.annotate(label, xy=(1, (low + high) / 2), color=color, fontsize=7,
                va="center", alpha=0.9)


def _overlay_zones(ax, ctx: dict[str, Any], n: int) -> None:
    ob = ctx.get("order_blocks", {}) or {}
    bull_ob, bear_ob = ob.get("bullish_ob"), ob.get("bearish_ob")
    if bull_ob:
        tf = (bull_ob.get("tf") or "").upper()
        _zone(ax, bull_ob["low"], bull_ob["high"], n, UP, f"OB {tf}")
    if bear_ob:
        tf = (bear_ob.get("tf") or "").upper()
        _zone(ax, bear_ob["low"], bear_ob["high"], n, DOWN, f"OB {tf}")

    fvg = ctx.get("fvg", {}) or {}
    for key in ("bullish_fvg", "bearish_fvg"):
        z = fvg.get(key)
        if z:
            _zone(ax, z["low"], z["high"], n, FVG_C, "FVG")

    eq = ctx.get("equilibrium", {}) or {}
    if eq.get("eq"):
        _hline(ax, eq["eq"], ACCENT, "EQ", n, style=":", lw=0.9)


def _finish(fig, ax, title: str) -> str:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    path = os.path.join(tempfile.gettempdir(),
                        f"chart_{abs(hash(title)) % 10**8}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def render_signal_chart(ctx: dict[str, Any], signal: dict[str, Any],
                        candles: int = 90) -> str:
    """Candles + zones + entry/SL/TP levels for a delivered signal."""
    df = ctx["df_signal"].iloc[-candles:].reset_index(drop=True)
    n = len(df)
    fig, ax, axv = _new_figure(df)
    _draw_candles(ax, axv, df)
    _overlay_zones(ax, ctx, n)

    entry = signal.get("entry_price")
    if entry:
        _hline(ax, entry, INK, "ВХОД", n, style="-", lw=1.2)
    if signal.get("stop_loss"):
        _hline(ax, signal["stop_loss"], DOWN, "SL", n)
    if signal.get("target_1"):
        _hline(ax, signal["target_1"], UP, "TP1", n)
    if signal.get("target_2"):
        _hline(ax, signal["target_2"], UP, "TP2", n)

    d = "ЛОНГ" if signal.get("direction") == "long" else "ШОРТ"
    title = (f"{config.SYMBOL_DISPLAY} {signal.get('style_label', '')} {d} | "
             f"{signal.get('timeframe', '').upper()} | score {signal.get('score')}")
    return _finish(fig, ax, title)


def render_levels_chart(ctx: dict[str, Any], candles: int = 110) -> str:
    """Candles + zones + nearest S/R and volume-profile levels."""
    df = ctx["df_signal"].iloc[-candles:].reset_index(drop=True)
    n = len(df)
    fig, ax, axv = _new_figure(df)
    _draw_candles(ax, axv, df)
    _overlay_zones(ax, ctx, n)

    vp = ctx.get("volume_profile", {}) or {}
    if vp.get("poc"):
        _hline(ax, vp["poc"], ACCENT, "POC", n, style="-.", lw=0.9)

    from bot.formatting import _nearest_sr
    support, resistance = _nearest_sr(ctx, ctx.get("price"))
    if support:
        _hline(ax, support, UP, "S", n)
    if resistance:
        _hline(ax, resistance, DOWN, "R", n)

    title = (f"{config.SYMBOL_DISPLAY} уровни | {ctx.get('timeframe', '').upper()} | "
             f"цена {_fmt(ctx.get('price', 0))}")
    return _finish(fig, ax, title)
