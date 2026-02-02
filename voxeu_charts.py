#!/usr/bin/env python3
"""
VoxEU article charts: AI hardware depreciation and hyperscaler capex.

Pulls quarterly capex from SEC EDGAR for Microsoft, Alphabet, Amazon, Meta,
constructs compute capital stock via Perpetual Inventory Method under three
depreciation assumptions, and produces two publication-quality figures.

Outputs:
  voxeu_fig1.png / .pdf  - Hyperscaler capital expenditure, 2018-2025
  voxeu_fig2.png / .pdf  - Decomposing AI investment: replacement vs net additions
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import YearLocator, DateFormatter
import pandas as pd

# ============================================================
# 0.  Configuration
# ============================================================

OUTDIR = Path(__file__).resolve().parent

FIRMS = {
    "Microsoft": {
        "cik": "0000789019",
        "tags": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    },
    "Alphabet": {
        "cik": "0001652044",
        "tags": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    },
    "Amazon": {
        "cik": "0001018724",
        # Amazon switched to PaymentsToAcquireProductiveAssets for recent years
        "tags": ["PaymentsToAcquireProductiveAssets",
                 "PaymentsToAcquirePropertyPlantAndEquipment"],
    },
    "Meta": {
        "cik": "0001326801",
        "tags": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot jonathan.rice@example.com)",
    "Accept": "application/json",
}

START_YEAR = 2018

# Quarterly depreciation rates
DELTA_HIGH = 0.095    # ~33% annual
DELTA_MED  = 0.06     # ~22% annual
DELTA_LOW  = 0.03     # ~11% annual

# ============================================================
# 1.  Pull capex from SEC EDGAR
# ============================================================

def fetch_company_facts(cik: str) -> dict:
    """Fetch the full companyfacts JSON from EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _end_to_q_label(end_date: str) -> str:
    """Map an end date string like '2023-06-30' to a quarter label like '2023Q2'."""
    month = int(end_date[5:7])
    year = int(end_date[:4])
    if month <= 3:
        return f"{year}Q1"
    elif month <= 6:
        return f"{year}Q2"
    elif month <= 9:
        return f"{year}Q3"
    else:
        return f"{year}Q4"


def extract_quarterly_capex(facts: dict, firm_name: str, tags: list) -> pd.Series:
    """
    Extract quarterly capex using EDGAR data.

    Three-pass strategy:
    1. Frame-based quarterly entries (CYyyyyQn) -- gold standard, single-quarter.
    2. Cumulative differencing: group entries sharing a common fiscal-year start
       date, sort by end date, and difference to get individual quarters.
    3. Calendar-year annual frame entries (CYyyyy): only use when the entry
       genuinely spans Jan 1 - Dec 31 to derive missing quarters.
    """
    usgaap = facts.get("facts", {}).get("us-gaap", {})

    # Try each tag in order; use the first one with recent data
    entries = None
    used_tag = None
    for tag in tags:
        if tag not in usgaap:
            continue
        candidate = usgaap[tag].get("units", {}).get("USD", [])
        has_recent = any(
            e.get("frame", "").startswith("CY201") or
            e.get("frame", "").startswith("CY202")
            for e in candidate
        )
        if has_recent:
            entries = candidate
            used_tag = tag
            break

    if entries is None:
        print(f"  WARNING: No suitable capex tag found for {firm_name}")
        return pd.Series(dtype=float)

    print(f"  {firm_name}: using tag '{used_tag}' ({len(entries)} entries)")

    quarterly_data = {}  # key: "2023Q1" -> value in USD

    # ---- Pass 1: Frame-based quarterly entries (authoritative) ----
    frame_quarters = {}
    frame_annual = {}      # year -> (val, start, end)
    for e in entries:
        frame = e.get("frame", "")
        if not frame:
            continue
        m = re.match(r"CY(\d{4})Q(\d)", frame)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            if y >= START_YEAR:
                frame_quarters[f"{y}Q{q}"] = abs(e["val"])
        m2 = re.match(r"CY(\d{4})$", frame)
        if m2:
            y = int(m2.group(1))
            if y >= START_YEAR - 1:
                frame_annual[y] = (abs(e["val"]),
                                   e.get("start", ""),
                                   e.get("end", ""))

    quarterly_data.update(frame_quarters)

    # ---- Pass 2: Cumulative differencing by fiscal-year start ----
    # Group all entries by their start date
    from datetime import datetime
    by_start = defaultdict(list)
    for e in entries:
        start = e.get("start", "")
        end = e.get("end", "")
        if not start or not end:
            continue
        try:
            int(end[:4])
        except (ValueError, IndexError):
            continue
        by_start[start].append(e)

    # For each start date with 2+ entries, difference cumulatives
    for start_date, group in by_start.items():
        if len(group) < 2:
            continue
        # De-duplicate by end date (keep entry with largest val = longest cumulation)
        by_end = {}
        for e in group:
            end = e["end"]
            val = abs(e["val"])
            if end not in by_end or val > by_end[end]:
                by_end[end] = val

        sorted_ends = sorted(by_end.items())  # sorted by end date
        prev_cum = 0
        for end_date, cum_val in sorted_ends:
            q_val = cum_val - prev_cum
            prev_cum = cum_val
            if q_val <= 0:
                continue
            q_label = _end_to_q_label(end_date)
            yr = int(q_label[:4])
            if yr < START_YEAR:
                continue
            # Only use if we don't already have a frame-based value
            if q_label not in quarterly_data:
                quarterly_data[q_label] = q_val

    # ---- Pass 3: Fill gaps from calendar-year annual frames ----
    for year, (annual_val, ann_start, ann_end) in frame_annual.items():
        if year < START_YEAR:
            continue
        # Only use genuine calendar-year annuals (Jan 1 - Dec 31)
        is_cal_year = (ann_start.endswith("-01-01") and ann_end.endswith("-12-31")
                       and ann_start[:4] == ann_end[:4] == str(year))
        if not is_cal_year:
            continue

        q_labels = [f"{year}Q{q}" for q in range(1, 5)]
        known = {ql: quarterly_data[ql] for ql in q_labels if ql in quarterly_data}
        missing = [ql for ql in q_labels if ql not in quarterly_data]

        if len(missing) == 1 and len(known) == 3:
            derived = annual_val - sum(known.values())
            if derived > 0:
                quarterly_data[missing[0]] = derived
        elif len(missing) > 1 and len(known) > 0:
            remaining = annual_val - sum(known.values())
            if remaining > 0:
                per_q = remaining / len(missing)
                for ql in missing:
                    quarterly_data[ql] = per_q
        elif len(missing) == 4:
            # No quarterly data at all; split annual evenly
            per_q = annual_val / 4
            for ql in missing:
                quarterly_data[ql] = per_q

    # Convert to pandas Series
    if not quarterly_data:
        return pd.Series(dtype=float)

    series = pd.Series(quarterly_data, dtype=float)
    series.index = pd.PeriodIndex(series.index, freq="Q")
    series = series.sort_index()
    series.name = firm_name

    return series


def build_capex_dataframe() -> pd.DataFrame:
    """Fetch and assemble quarterly capex for all four hyperscalers."""
    all_series = {}
    for name, info in FIRMS.items():
        cik = info["cik"]
        tags = info["tags"]
        print(f"Fetching {name} (CIK {cik})...")
        try:
            facts = fetch_company_facts(cik)
            s = extract_quarterly_capex(facts, name, tags)
            if not s.empty:
                all_series[name] = s / 1e9  # Convert to billions
                print(f"  -> {len(s)} quarterly observations")
            else:
                print(f"  -> No data extracted")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            import traceback; traceback.print_exc()
        time.sleep(0.2)

    capex = pd.DataFrame(all_series)
    capex.index.name = "quarter"
    capex = capex.sort_index()

    # Filter to START_YEAR onwards
    capex = capex[capex.index >= pd.Period(f"{START_YEAR}Q1", freq="Q")]

    # Drop trailing quarters where any firm is missing (not yet filed)
    while not capex.empty and capex.iloc[-1].isna().any():
        dropped = capex.index[-1]
        capex = capex.iloc[:-1]
        missing = [c for c in capex.columns if pd.isna(capex.iloc[-1][c])] if not capex.empty else []
        print(f"  Trimmed {dropped} (incomplete data)")

    return capex


# ============================================================
# 2.  Perpetual Inventory Method
# ============================================================

def perpetual_inventory(capex_total: pd.Series, delta: float) -> pd.DataFrame:
    """
    Given a quarterly combined capex series and quarterly depreciation rate,
    compute capital stock, replacement, and net additions.
    """
    capex = capex_total.values
    n = len(capex)
    stock = np.zeros(n)
    replacement = np.zeros(n)
    net_add = np.zeros(n)

    # Initialize: steady-state assumption
    stock[0] = capex[0] / delta
    replacement[0] = delta * stock[0]
    net_add[0] = capex[0] - replacement[0]

    for t in range(1, n):
        replacement[t] = delta * stock[t - 1]
        net_add[t] = capex[t] - replacement[t]
        stock[t] = (1 - delta) * stock[t - 1] + capex[t]

    return pd.DataFrame({
        "capex": capex,
        "stock": stock,
        "replacement": replacement,
        "net_additions": net_add,
    }, index=capex_total.index)


# ============================================================
# 3.  Charts
# ============================================================

# Firm colours - distinctive, print-friendly palette
FIRM_COLORS = {
    "Microsoft": "#2176AE",
    "Alphabet":  "#57B8FF",
    "Amazon":    "#F77F00",
    "Meta":      "#D62828",
}


def setup_style():
    """Set clean, publication-quality defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def fig1_capex_by_firm(capex: pd.DataFrame):
    """
    Chart A: Stacked area of quarterly capex by firm.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # Convert period index to timestamps for plotting
    dates = capex.index.to_timestamp()
    firms_ordered = ["Amazon", "Microsoft", "Alphabet", "Meta"]
    cols = [c for c in firms_ordered if c in capex.columns]
    data = capex[cols].fillna(0)

    ax.stackplot(
        dates,
        *[data[c].values for c in cols],
        labels=cols,
        colors=[FIRM_COLORS[c] for c in cols],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Quarterly capex (\\$bn)")
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(0)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(OUTDIR / f"voxeu_fig1.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved voxeu_fig1.png and .pdf")


def fig2_replacement_vs_net(
    capex_total: pd.Series,
    pim_high: pd.DataFrame,
    pim_med: pd.DataFrame,
    pim_low: pd.DataFrame,
):
    """
    Chart B: Stacked area -- replacement vs net additions (high-delta),
    with secondary y-axis showing implied stock under all three scenarios.
    """
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))

    dates = pim_high.index.to_timestamp()

    # Stacked area: replacement (bottom) + net additions (top)
    replacement = pim_high["replacement"].values
    net_add = pim_high["net_additions"].values

    # When net additions are negative, all spending is replacement
    net_add_pos = np.maximum(net_add, 0)
    replacement_shown = pim_high["capex"].values - net_add_pos

    ax1.fill_between(dates, 0, replacement_shown,
                     color="#B0B0B0", alpha=0.8, label="Replacement investment")
    ax1.fill_between(dates, replacement_shown, replacement_shown + net_add_pos,
                     color="#2176AE", alpha=0.8, label="Net additions to stock")

    ax1.set_ylabel("Quarterly investment (\\$bn)")
    ax1.set_ylim(0)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_major_formatter(DateFormatter("%Y"))

    # Secondary y-axis: capital stock
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)

    ax2.plot(dates, pim_high["stock"].values, "-", color="#D62828",
             linewidth=1.8, label=f"Stock ($\\delta_q$=0.095)")
    ax2.plot(dates, pim_med["stock"].values, "--", color="#F77F00",
             linewidth=1.5, label=f"Stock ($\\delta_q$=0.06)")
    ax2.plot(dates, pim_low["stock"].values, ":", color="#2D6A4F",
             linewidth=1.5, label=f"Stock ($\\delta_q$=0.03)")

    ax2.set_ylabel("Implied capital stock (\\$bn)")
    ax2.set_ylim(0)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=1)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(OUTDIR / f"voxeu_fig2.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved voxeu_fig2.png and .pdf")


# ============================================================
# 4.  Summary statistics
# ============================================================

def print_summary(capex: pd.DataFrame, pim_high: pd.DataFrame,
                  pim_med: pd.DataFrame, pim_low: pd.DataFrame):
    """Print key summary statistics."""
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS")
    print("=" * 65)

    # Annual total capex
    capex_annual = capex.copy()
    capex_annual["Total"] = capex_annual.sum(axis=1)
    capex_annual["year"] = capex_annual.index.to_timestamp().year
    yearly = capex_annual.groupby("year").sum()

    print("\n--- Total combined capex by year ($bn) ---")
    for yr in sorted(yearly.index):
        row = yearly.loc[yr]
        firm_str = "  ".join(
            f"{c}: {row[c]:6.1f}" for c in capex.columns if c in row.index
        )
        print(f"  {yr}:  Total={row['Total']:6.1f}   ({firm_str})")

    # Replacement fraction (high-delta)
    print(f"\n--- Replacement fraction of spending (high delta=0.095) ---")
    pim = pim_high.copy()
    pim["quarter"] = pim.index
    pim["repl_frac"] = pim["replacement"] / pim["capex"]
    tail = pim.tail(8)
    for _, row in tail.iterrows():
        print(f"  {row['quarter']}:  capex={row['capex']:.1f}  "
              f"repl={row['replacement']:.1f}  net={row['net_additions']:.1f}  "
              f"repl_share={row['repl_frac']:.1%}")

    # Stock levels under each scenario (latest quarter)
    latest_q = pim_high.index[-1]
    print(f"\n--- Implied capital stock at {latest_q} ($bn) ---")
    print(f"  High delta (0.095): {pim_high['stock'].iloc[-1]:8.1f}")
    print(f"  Med  delta (0.060): {pim_med['stock'].iloc[-1]:8.1f}")
    print(f"  Low  delta (0.030): {pim_low['stock'].iloc[-1]:8.1f}")
    print("=" * 65)


# ============================================================
# Main
# ============================================================

def main():
    setup_style()

    # Step 1: Fetch capex
    print("=" * 65)
    print("STEP 1: Fetching quarterly capex from SEC EDGAR")
    print("=" * 65)
    capex = build_capex_dataframe()
    print(f"\nCapex DataFrame shape: {capex.shape}")
    print(capex.to_string())

    # Combined capex series
    capex_total = capex.sum(axis=1)
    capex_total.name = "total_capex"

    # Drop any rows where total is 0 or NaN (shouldn't happen but safety)
    capex_total = capex_total[capex_total > 0]

    # Step 2: Perpetual inventory
    print("\n" + "=" * 65)
    print("STEP 2: Perpetual Inventory Method")
    print("=" * 65)
    pim_high = perpetual_inventory(capex_total, DELTA_HIGH)
    pim_med  = perpetual_inventory(capex_total, DELTA_MED)
    pim_low  = perpetual_inventory(capex_total, DELTA_LOW)

    # Step 3: Charts
    print("\n" + "=" * 65)
    print("STEP 3: Generating charts")
    print("=" * 65)
    fig1_capex_by_firm(capex)
    fig2_replacement_vs_net(capex_total, pim_high, pim_med, pim_low)

    # Step 4: Summary statistics
    print_summary(capex, pim_high, pim_med, pim_low)


if __name__ == "__main__":
    main()
