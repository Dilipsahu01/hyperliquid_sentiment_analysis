#!/usr/bin/env python3
"""
btc_pipeline/macro_calendar.py

Pre-registered Macroeconomic Release Calendar (2023 - 2026).
Covers ALL scheduled US CPI, NFP, and FOMC releases.

Strict Timezone Handling:
All local times (08:30 AM / 14:00 PM Eastern Time) are defined in zoneinfo 'America/New_York'
and dynamically converted to UTC via .astimezone(datetime.timezone.utc), strictly ensuring
correct Daylight Saving Time (EDT: UTC-4, EST: UTC-5) alignment without manual offset bugs.
"""

import json
import zoneinfo
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = datetime.timezone.utc

# 1. CPI Releases (8:30 AM Eastern Time)
CPI_SCHEDULE = [
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13",
    "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
    "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-15", "2025-08-13", "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14", "2026-05-12", "2026-06-10",
    "2026-07-14", "2026-08-12", "2026-09-15", "2026-10-13", "2026-11-12"
]

# 2. NFP Releases (8:30 AM Eastern Time)
NFP_SCHEDULE = [
    # 2023
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07", "2023-05-05", "2023-06-02",
    "2023-07-07", "2023-08-04", "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    # 2024
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03", "2024-06-07",
    "2024-07-05", "2024-08-02", "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06",
    "2025-07-11", "2025-08-01", "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
    "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06"
]

# 3. FOMC Rate Decisions (14:00 PM Eastern Time Statement)
FOMC_SCHEDULE = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29", "2026-09-16", "2026-11-04"
]


def build_events_calendar() -> pd.DataFrame:
    events = []

    # Process CPI
    for d_str in CPI_SCHEDULE:
        y, m, d = map(int, d_str.split("-"))
        dt_ny = datetime.datetime(y, m, d, 8, 30, tzinfo=NY_TZ)
        dt_utc = dt_ny.astimezone(UTC_TZ)
        is_dst = bool(dt_ny.dst() and dt_ny.dst().total_seconds() > 0)
        
        # Placebo: exactly 24 hours prior (same time of day, non-event)
        dt_placebo_utc = dt_utc - datetime.timedelta(days=1)
        
        events.append({
            "event_id": f"CPI_{d_str}",
            "category": "CPI",
            "release_date_ny": d_str,
            "release_time_ny": "08:30:00",
            "tz_source": "America/New_York",
            "is_dst": is_dst,
            "dst_code": "EDT (UTC-4)" if is_dst else "EST (UTC-5)",
            "timestamp_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "utc_hour": dt_utc.hour,
            "utc_minute": dt_utc.minute,
            "placebo_timestamp_utc": dt_placebo_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "epoch_ms": int(dt_utc.timestamp() * 1000)
        })

    # Process NFP
    for d_str in NFP_SCHEDULE:
        y, m, d = map(int, d_str.split("-"))
        dt_ny = datetime.datetime(y, m, d, 8, 30, tzinfo=NY_TZ)
        dt_utc = dt_ny.astimezone(UTC_TZ)
        is_dst = bool(dt_ny.dst() and dt_ny.dst().total_seconds() > 0)
        
        # Placebo: exactly 7 days prior (same Friday time, non-NFP)
        dt_placebo_utc = dt_utc - datetime.timedelta(days=7)

        events.append({
            "event_id": f"NFP_{d_str}",
            "category": "NFP",
            "release_date_ny": d_str,
            "release_time_ny": "08:30:00",
            "tz_source": "America/New_York",
            "is_dst": is_dst,
            "dst_code": "EDT (UTC-4)" if is_dst else "EST (UTC-5)",
            "timestamp_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "utc_hour": dt_utc.hour,
            "utc_minute": dt_utc.minute,
            "placebo_timestamp_utc": dt_placebo_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "epoch_ms": int(dt_utc.timestamp() * 1000)
        })

    # Process FOMC
    for d_str in FOMC_SCHEDULE:
        y, m, d = map(int, d_str.split("-"))
        dt_ny = datetime.datetime(y, m, d, 14, 0, tzinfo=NY_TZ)
        dt_utc = dt_ny.astimezone(UTC_TZ)
        is_dst = bool(dt_ny.dst() and dt_ny.dst().total_seconds() > 0)

        # Placebo: exactly 24 hours prior (Tuesday 14:00 NY time)
        dt_placebo_utc = dt_utc - datetime.timedelta(days=1)

        events.append({
            "event_id": f"FOMC_{d_str}",
            "category": "FOMC",
            "release_date_ny": d_str,
            "release_time_ny": "14:00:00",
            "tz_source": "America/New_York",
            "is_dst": is_dst,
            "dst_code": "EDT (UTC-4)" if is_dst else "EST (UTC-5)",
            "timestamp_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "utc_hour": dt_utc.hour,
            "utc_minute": dt_utc.minute,
            "placebo_timestamp_utc": dt_placebo_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "epoch_ms": int(dt_utc.timestamp() * 1000)
        })

    df = pd.DataFrame(events).sort_values("epoch_ms").reset_index(drop=True)

    # Classify Inclusion / Exclusion Status
    # In-Sample Dataset Coverage: 2023-01-01 to 2026-11-25
    # Hyperliquid BTC perp launched in late 2023, reached liquidity threshold (>50 BTC/day) by 2024-Q1.
    def get_status(row):
        d_val = pd.to_datetime(row["release_date_ny"])
        if d_val < pd.Timestamp("2024-01-01"):
            return "Excluded", "Pre-dates Hyperliquid BTC perpetual institutional liquidity threshold (>50 BTC 24h vol)"
        elif d_val > pd.Timestamp("2026-11-25"):
            return "Excluded", "Post-dates evaluation sample cutoff (2026-11-25)"
        else:
            return "Included", "Covered in full 2024-2026 macro evaluation window"

    statuses = [get_status(r) for _, r in df.iterrows()]
    df["status"] = [s[0] for s in statuses]
    df["status_reason"] = [s[1] for s in statuses]

    # Save to data/
    csv_path = DATA_DIR / "macro_events_calendar.csv"
    json_path = DATA_DIR / "macro_events_calendar.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "metadata": {
            "total_preregistered_events": len(df),
            "date_range": f"{df['release_date_ny'].min()} to {df['release_date_ny'].max()}",
            "categories": df["category"].value_counts().to_dict(),
            "included_events_count": int(np.sum(df["status"] == "Included")),
            "excluded_events_count": int(np.sum(df["status"] == "Excluded")),
            "dst_breakdown": df["dst_code"].value_counts().to_dict()
        },
        "events": df.to_dict(orient="records")
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("PRE-REGISTERED MACROECONOMIC EVENT CALENDAR (2023 - 2026)")
    print("=" * 80)
    print(f"Total Pre-registered Releases: {len(df)}")
    print(f"  • CPI Releases:  {len(CPI_SCHEDULE)}")
    print(f"  • NFP Releases:  {len(NFP_SCHEDULE)}")
    print(f"  • FOMC Meetings: {len(FOMC_SCHEDULE)}")
    print(f"\nInclusion / Exclusion Breakdown:")
    print(f"  • Included (2024 - 2026 Active Liquidity): {summary['metadata']['included_events_count']}")
    print(f"  • Excluded (2023 Pre-liquidity):           {summary['metadata']['excluded_events_count']}")
    print(f"\nDaylight Saving Time (DST) Correct Alignment:")
    print(f"  • Summer (EDT, 12:30 UTC CPI/NFP, 18:00 UTC FOMC): {summary['metadata']['dst_breakdown'].get('EDT (UTC-4)', 0)}")
    print(f"  • Winter (EST, 13:30 UTC CPI/NFP, 19:00 UTC FOMC): {summary['metadata']['dst_breakdown'].get('EST (UTC-5)', 0)}")
    print(f"\n[OK] Saved calendar CSV to:  {csv_path}")
    print(f"[OK] Saved calendar JSON to: {json_path}")
    return df


if __name__ == "__main__":
    build_events_calendar()
