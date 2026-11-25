"""Region/date vocab for metadata-conditioned restoration fine-tunes.

Region and tpq/taq come straight off the raw PHI record (`main_region`, `tpq`, `taq`),
which are strings with real values ("-400"), signed-zero variants ("-0"), and assorted
junk (sentinels "999"/"-999", "NULL", "null34", "-", ""). REGIONS is the fixed, ordered
list of the 14 main_region values actually present in raw/iphi.jsonl (checked directly
against the corpus, not guessed) plus "unknown" for anything else (papyri records, which
never carry this metadata, and any inscription missing/blank main_region).
"""
from __future__ import annotations

import re

REGIONS = [
    "Asia Minor",
    "Aegean Islands, incl. Crete",
    "Attica",
    "Central Greece",
    "Peloponnesos",
    "Egypt, Nubia and Cyrenaïca",
    "Northern Greece",
    "Greater Syria and the East",
    "Sicily, Italy, and the West",
    "Thrace and the Lower Danube",
    "North Shore of the Black Sea",
    "Cyprus",
    "Upper Danube",
    "North Africa",
    "unknown",
]
REGION_TO_ID = {r: i for i, r in enumerate(REGIONS)}
UNK_REGION = REGION_TO_ID["unknown"]
N_REGION = len(REGIONS)

# Century buckets span 800 BC .. 600 AD (Ithaca/PHI's actual coverage), 100-year bins,
# plus one UNK bucket for missing/unparsable dates.
CENTURY_LO, CENTURY_HI = -800, 600
N_CENTURY_BINS = (CENTURY_HI - CENTURY_LO) // 100 + 1   # 14 bins
UNK_CENTURY = N_CENTURY_BINS
N_CENTURY = N_CENTURY_BINS + 1

_YEAR_RE = re.compile(r"^-?\d+$")


def region_to_id(region) -> int:
    return REGION_TO_ID.get(region, UNK_REGION)


def parse_year(v) -> int | None:
    """Raw tpq/taq field -> int year or None. Rejects sentinels/junk: "", "-", "NULL",
    "null34"-style garbage, and the ±999 out-of-range sentinel PHI uses for "no data"."""
    if v is None:
        return None
    s = str(v).strip()
    if not _YEAR_RE.match(s):
        return None
    y = int(s)
    if y in (999, -999) or y == 0 and s == "-0":
        return None
    return y


def year_to_century_id(year: int | None) -> int:
    if year is None:
        return UNK_CENTURY
    y = max(CENTURY_LO, min(CENTURY_HI, year))
    return (y - CENTURY_LO) // 100


def record_century_id(tpq, taq) -> int:
    """Best available point estimate for a record's date: midpoint of tpq/taq when both
    parse, else whichever one parses, else UNK."""
    a, b = parse_year(tpq), parse_year(taq)
    if a is not None and b is not None:
        return year_to_century_id((a + b) // 2)
    if a is not None:
        return year_to_century_id(a)
    if b is not None:
        return year_to_century_id(b)
    return UNK_CENTURY
