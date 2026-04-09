"""
rollex_utils.py — Central loader for Master Database Rollex parquets.

Usage in any dashboard or notebook:
    import sys
    sys.path.append(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Non Fundamental\Master Database\Rollex")
    from rollex_utils import load_rollex, load_all_rollex, AVAILABLE

    kc  = load_rollex("KC")       # single commodity
    all = load_all_rollex()       # dict of all 8

Columns returned per commodity:
    c1, c2              — raw settlement prices (front / second month)
    A, B                — regime flags (A=following c1, B=following c2)
    switch              — 1 on expiry/roll date
    c1_ret, c2_ret      — daily returns
    rollex_ret          — roll-adjusted daily return
    rollex_px           — roll-adjusted price level
    active_ric          — physical contract active on that date  e.g. "KCK18"
    active_label        — human-readable label                   e.g. "May'18"
    active_fnd          — First Notice Date of active contract
    active_ltd          — Last Trading Date of active contract
"""

import pandas as pd
from pathlib import Path

_DB          = Path(__file__).parent / "Database"
AVAILABLE    = ["KC", "RC", "CC", "LCC", "SB", "CT", "LSU", "OJ"]
_STALE_HOURS = 24


def load_rollex(comm: str) -> pd.DataFrame:
    """
    Load roll-adjusted series for one commodity.

    Parameters
    ----------
    comm : str  — one of KC, RC, CC, LCC, SB, CT, LSU, OJ

    Returns
    -------
    pd.DataFrame  (date-indexed)
    """
    comm = comm.upper()
    if comm not in AVAILABLE:
        raise ValueError(f"'{comm}' not recognised. Available: {AVAILABLE}")

    path = _DB / f"rollex_{comm}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"rollex_{comm}.parquet not found.\n"
            f"Run Code/rollex_builder.py first."
        )

    age_hours = (
        pd.Timestamp.now() -
        pd.Timestamp(path.stat().st_mtime, unit="s")
    ).total_seconds() / 3600

    if age_hours > _STALE_HOURS:
        print(f"[rollex_utils] WARNING: rollex_{comm}.parquet is "
              f"{age_hours:.0f}h old — consider running rollex_builder.py")

    return pd.read_parquet(path)


def load_all_rollex() -> dict:
    """
    Load all available commodities.

    Returns
    -------
    dict  e.g. {"KC": df_kc, "CC": df_cc, ...}
    Skips any commodity whose parquet doesn't exist yet (prints a warning).
    """
    result = {}
    for comm in AVAILABLE:
        try:
            result[comm] = load_rollex(comm)
        except FileNotFoundError as e:
            print(f"[rollex_utils] Skipping {comm}: {e}")
    return result
