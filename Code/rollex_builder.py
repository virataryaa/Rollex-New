"""
rollex_builder.py — Roll-Adjusted Time Series built from Master Database
========================================================================
- c1/c2 settlement prices fetched from LSEG
- Expiry dates derived from commodity_contract_engine (no parquet dependency)
- Active contract tagged per row (regime-aware)

Usage:
    python rollex_builder.py           # incremental update
    python rollex_builder.py --full    # full rebuild from START_DATE
"""

import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import lseg.data as rd
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).resolve().parent
ROLLEX_DIR  = CODE_DIR.parent
MASTER_CODE = CODE_DIR.parents[1] / "Code"   # Master Database/Code
REPO_DIR    = ROLLEX_DIR                      # Rollex folder IS the git repo
DB_DIR      = ROLLEX_DIR / "Database"         # parquets live here

sys.path.insert(0, str(MASTER_CODE))
from commodity_contract_engine import generate_contract_table

DB_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIG — edit here
# ============================================================
OFFSET     = 30          # trading days before LTD to switch c1 -> c2
START_DATE = "2005-01-01"
START_YEAR = 2004        # contract table generation start year

COMMODITIES = {
    "KC":  {"c1": "KCc1",  "c2": "KCc2",  "engine_key": "KC"},
    "RC":  {"c1": "LRCc1", "c2": "LRCc2", "engine_key": "RC"},
    "CC":  {"c1": "CCc1",  "c2": "CCc2",  "engine_key": "CC"},
    "LCC": {"c1": "LCCc1", "c2": "LCCc2", "engine_key": "LCC"},
    "SB":  {"c1": "SBc1",  "c2": "SBc2",  "engine_key": "SB"},
    "CT":  {"c1": "CTc1",  "c2": "CTc2",  "engine_key": "CT"},
    "LSU": {"c1": "LSUc1", "c2": "LSUc2", "engine_key": "LSU"},
    "OJ":  {"c1": "OJc1",  "c2": "OJc2",  "engine_key": "OJ"},
}
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Full rebuild from START_DATE")
args = parser.parse_args()
INCREMENTAL = not args.full

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_expiry_dates(engine_key: str) -> pd.DatetimeIndex:
    """Generate LTD dates from commodity_contract_engine — no parquet dependency."""
    end_year = pd.Timestamp.today().year + 1
    ct = generate_contract_table(engine_key, START_YEAR, end_year)
    ltds = pd.to_datetime(ct["LTD"]).dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(ltds)


def fetch_settlement(ric: str, start: str, end: str) -> pd.Series:
    """Fetch daily settlement prices from LSEG for a single RIC."""
    raw = rd.get_history(
        universe=[ric],
        fields=["TR.SETTLEMENTPRICE"],
        start=start, end=end,
        interval="daily"
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=ric)
    s = raw.iloc[:, 0]
    s.index = pd.to_datetime(s.index).normalize()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = pd.to_numeric(s, errors="coerce")
    return s.rename(ric)


def build_tags(dates: list, regime_a: list, engine_key: str) -> pd.DataFrame:
    """
    Tag each date with the active physical contract.

    Regime A (c1 regime) -> nearest non-expired contract
    Regime B (c2 regime) -> second nearest contract

    Uses commodity_contract_engine — no parquet dependency.
    """
    end_year = pd.Timestamp.today().year + 1
    ct = generate_contract_table(engine_key, START_YEAR, end_year)
    ct["LTD"] = pd.to_datetime(ct["LTD"]).dt.normalize()
    ct["FND"] = pd.to_datetime(ct["FND"]).dt.normalize()
    ct = ct.sort_values("LTD").reset_index(drop=True)

    ltd_arr = ct["LTD"].values  # numpy datetime64 array for searchsorted

    rows = []
    for d, is_A in zip(dates, regime_a):
        d_np = np.datetime64(pd.Timestamp(d).normalize())
        # index of first contract with LTD >= d
        c1_idx = int(np.searchsorted(ltd_arr, d_np, side="left"))
        target  = c1_idx if is_A else c1_idx + 1

        if target >= len(ct):
            target = len(ct) - 1

        contract = ct.iloc[target]
        ltd      = contract["LTD"]
        label    = f"{MONTH_NAMES[ltd.month]}'{str(ltd.year)[2:]}"

        rows.append({
            "active_ric":   contract["base_ric"],
            "active_label": label,
            "active_fnd":   contract["FND"],
            "active_ltd":   ltd,
        })

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


def build_rollex(comm: str, c1_s: pd.Series, c2_s: pd.Series,
                 expiry_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build roll-adjusted price series for one commodity."""

    df = pd.concat([c1_s.rename("c1"), c2_s.rename("c2")], axis=1)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[START_DATE:]
    df = df.dropna(subset=["c1", "c2"])
    df = df[(df["c1"] > 0) & (df["c2"] > 0)]

    if df.empty:
        print(f"  [{comm}] WARNING: no data after cleaning — skipped")
        return pd.DataFrame()

    dates   = df.index.tolist()
    exp_set = set(expiry_dates)
    n       = len(dates)

    switch_flag, regime_A, regime_B = [], [], []

    for i, d in enumerate(dates):
        # switch_flag = 1 on expiry date, or first trading day after if expiry falls on non-trading day
        is_exp = d in exp_set
        if not is_exp and i > 0:
            prev   = dates[i - 1]
            is_exp = any(prev < e <= d for e in exp_set)
        switch_flag.append(1 if is_exp else 0)

        # Regime B if any expiry falls within next OFFSET trading days
        window_end = dates[min(i + OFFSET - 1, n - 1)]
        has_expiry = any(d <= e <= window_end for e in exp_set)
        regime_A.append(0 if has_expiry else 1)
        regime_B.append(1 if has_expiry else 0)

    df["switch"] = switch_flag
    df["A"]      = regime_A
    df["B"]      = regime_B
    df["c1_ret"] = df["c1"].pct_change()
    df["c2_ret"] = df["c2"].pct_change()

    # Rollex return
    rollex_ret = [np.nan]
    for i in range(1, n):
        if switch_flag[i - 1] == 1:                       # yesterday = expiry -> roll bridge
            ret = df["c1"].iat[i] / df["c2"].iat[i - 1] - 1
        elif regime_A[i] == 1:
            ret = df["c1_ret"].iat[i]
        else:
            ret = df["c2_ret"].iat[i]
        rollex_ret.append(ret)

    df["rollex_ret"] = rollex_ret

    # Compound forward, anchor to latest actual settlement
    anchor = df["c1"].iat[-1] if regime_A[-1] == 1 else df["c2"].iat[-1]
    px = [1.0]
    for i in range(1, n):
        r = df["rollex_ret"].iat[i]
        px.append(px[-1] * (1 + r) if pd.notna(r) else px[-1])
    scale           = anchor / px[-1]
    df["rollex_px"] = [p * scale for p in px]

    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

print(f"Opening LSEG session...")
rd.open_session()

END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
results  = {}

for comm, cfg in COMMODITIES.items():
    print(f"\n{'='*55}")
    print(f"  {comm}  ({cfg['c1']} / {cfg['c2']})")
    print(f"{'='*55}")

    # Expiry dates from engine — no parquet dependency
    expiries = get_expiry_dates(cfg["engine_key"])
    print(f"  Expiries: {len(expiries)}  "
          f"({expiries.min().date()} -> {expiries.max().date()})")

    # Determine fetch window
    out_path = DB_DIR / f"rollex_{comm}.parquet"
    if INCREMENTAL and out_path.exists():
        existing    = pd.read_parquet(out_path)
        latest      = existing.index.max()
        fetch_start = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"  Mode: INCREMENTAL from {fetch_start}")
    else:
        existing    = None
        fetch_start = START_DATE
        print(f"  Mode: FULL from {fetch_start}")

    # Fetch c1 and c2
    c1_s = c2_s = None
    for label, ric in [("c1", cfg["c1"]), ("c2", cfg["c2"])]:
        try:
            s = fetch_settlement(ric, fetch_start, END_DATE).dropna()
            s = s[s > 0]
            if existing is not None:
                hist = existing[label].dropna()
                hist = hist[hist > 0]
                s = pd.concat([hist, s])
                s = s[~s.index.duplicated(keep="last")].sort_index()
            print(f"  {ric}: {len(s)} rows  "
                  f"({s.index.min().date()} -> {s.index.max().date()})")
            if label == "c1":
                c1_s = s
            else:
                c2_s = s
        except Exception as e:
            print(f"  ERROR fetching {ric}: {e}")
            if existing is not None:
                hist = existing[label].dropna()
                hist = hist[hist > 0]
                if not hist.empty:
                    print(f"  Falling back to existing {label} ({len(hist)} rows)")
                    if label == "c1":
                        c1_s = hist
                    else:
                        c2_s = hist

    if c1_s is None or c2_s is None or c1_s.empty or c2_s.empty:
        print(f"  Skipping {comm} — incomplete price data")
        continue

    # Build core Rollex
    df_out = build_rollex(comm, c1_s, c2_s, expiries)
    if df_out.empty:
        continue

    # Tag active contracts
    print(f"  Tagging active contracts...")
    tags   = build_tags(df_out.index.tolist(), df_out["A"].tolist(), cfg["engine_key"])
    df_out = pd.concat([df_out, tags], axis=1)

    # Save
    df_out.to_parquet(out_path)
    results[comm] = df_out
    print(f"  Saved -> {out_path.name}  |  {len(df_out)} rows  |  "
          f"Rollex Px: {df_out['rollex_px'].iat[-1]:.4f}  |  "
          f"Active: {df_out['active_label'].iat[-1]}")

rd.close_session()
print("\nLSEG session closed.")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  DONE — {len(results)}/{len(COMMODITIES)} commodities built")
print(f"{'='*55}")
for comm, df in results.items():
    print(f"  {comm:5s}  rows={len(df):5d}  "
          f"from={df.index.min().date()}  to={df.index.max().date()}  "
          f"rollex_px={df['rollex_px'].iat[-1]:.2f}  "
          f"active={df['active_label'].iat[-1]}")

# ── GIT PUSH TO GITHUB ────────────────────────────────────────────────────────
if results:
    print(f"\n{'='*55}")
    print("  Pushing to GitHub (Rollex-New)...")
    print(f"{'='*55}")
    try:
        def git(cmd: list):
            r = subprocess.run(["git", "-C", str(REPO_DIR)] + cmd,
                               capture_output=True, text=True)
            if r.stdout.strip():
                print(f"  {r.stdout.strip()}")
            if r.returncode != 0 and r.stderr.strip():
                print(f"  WARN: {r.stderr.strip()}")
            return r.returncode

        git(["add", "Database/"])
        git(["commit", "--amend", "--no-edit"])
        git(["push", "--force-with-lease"])
        print("  GitHub push complete.")
    except Exception as e:
        print(f"  GitHub push failed: {e}")
