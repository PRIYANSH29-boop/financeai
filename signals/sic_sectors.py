"""
SIC → sector mapping — Phase 20 fallback (source B).

The wide universe (`data/universe_midlarge.csv`) carries no sector, so the pie engine's
≤5-names/sector and ≤30%/sector caps collapse into one '?' bucket. Source A (yfinance
`.info` sector) labels most names; for the rest we map the SEC 4-digit **SIC code** (from
EDGAR, keyed by the CIK already in the universe file) to the SAME 11-bucket taxonomy
yfinance uses, so A and B labels are consistent within a book.

Taxonomy = the 11 Yahoo/`.info` sectors. This is a COARSE, hand-checked range map whose only
job is to prevent correlated clustering in the caps — not GICS-grade classification. It is
applied to TODAY's SIC; historical reclassification is ignored (documented caveat).

Design: `sic_to_sector` is total — every 4-digit SIC (0000–9999) resolves to exactly one of
the 11 sectors, with a documented default, so the completeness test can never find a hole.
"""

from __future__ import annotations

# The 11 canonical sectors (Yahoo `.info` taxonomy). Source A and B both emit only these.
SECTORS = (
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Basic Materials", "Energy",
    "Utilities", "Real Estate", "Communication Services",
)

DEFAULT_SECTOR = "Industrials"   # documented catch-all for unmapped/edge SICs

# Specific 4-digit overrides that a coarse range map would misfile. Hand-checked against the
# SEC SIC list; these are the codes where the major group's bucket is wrong for the sub-code.
_OVERRIDES = {
    # drugs & biologics sit inside the 28xx chemicals group -> Healthcare
    **{s: "Healthcare" for s in range(2830, 2837)},
    2835: "Healthcare", 2836: "Healthcare", 8000: "Healthcare",
    # prepackaged software & computer services inside 73xx business services -> Technology
    **{s: "Technology" for s in range(7370, 7380)},
    3571: "Technology", 3572: "Technology", 3576: "Technology", 3577: "Technology",
    # medical instruments inside the 38xx instruments group -> Healthcare
    **{s: "Healthcare" for s in range(3840, 3852)},
    # motor vehicles inside 37xx transport-equipment -> Consumer Cyclical (aerospace stays Industrials)
    **{s: "Consumer Cyclical" for s in range(3710, 3717)},
    # REITs & real-estate operators inside 67xx holding offices -> Real Estate
    6798: "Real Estate", 6500: "Real Estate",
    # grocery & drug retail inside 5xxx retail -> Consumer Defensive
    5411: "Consumer Defensive", 5412: "Consumer Defensive", 5912: "Consumer Defensive",
    # advertising & motion pictures -> Communication Services
    7310: "Communication Services", 7311: "Communication Services",
    **{s: "Communication Services" for s in range(7812, 7842)},
}

# Coarse major-group ranges (inclusive lo, inclusive hi, sector). Order does not matter;
# overrides above win. Chosen to cover the full 0000–9999 space with no gaps.
_RANGES = (
    (100, 999, "Consumer Defensive"),      # agriculture, forestry, fishing
    (1000, 1299, "Basic Materials"),       # metal & coal mining
    (1300, 1399, "Energy"),                # oil & gas extraction
    (1400, 1499, "Basic Materials"),       # nonmetallic minerals
    (1500, 1799, "Industrials"),           # construction
    (2000, 2199, "Consumer Defensive"),    # food & tobacco
    (2200, 2399, "Consumer Cyclical"),     # textiles & apparel
    (2400, 2499, "Basic Materials"),       # lumber & wood
    (2500, 2599, "Consumer Cyclical"),     # furniture
    (2600, 2699, "Basic Materials"),       # paper
    (2700, 2799, "Communication Services"),# printing & publishing
    (2800, 2999, "Basic Materials"),       # chemicals (drugs overridden) ...
    (2900, 2999, "Energy"),                # petroleum refining (narrower range below wins)
    (3000, 3199, "Consumer Cyclical"),     # rubber, plastics, leather
    (3200, 3399, "Basic Materials"),       # stone/clay/glass, primary metals
    (3400, 3599, "Industrials"),           # fabricated metal & machinery (computers overridden)
    (3600, 3699, "Technology"),            # electronic & electrical equipment
    (3700, 3799, "Industrials"),           # transport equipment (autos overridden)
    (3800, 3899, "Technology"),            # instruments (medical overridden)
    (3900, 3999, "Consumer Cyclical"),     # misc manufacturing (jewelry, toys)
    (4000, 4799, "Industrials"),           # transportation services
    (4800, 4899, "Communication Services"),# communications (telephone, broadcasting)
    (4900, 4949, "Utilities"),             # electric, gas, water
    (4950, 4999, "Industrials"),           # sanitary / refuse
    (5000, 5199, "Industrials"),           # wholesale distribution
    (5200, 5999, "Consumer Cyclical"),     # retail (grocery/drug overridden)
    (6000, 6499, "Financial Services"),    # banks, brokers, insurance
    (6500, 6599, "Real Estate"),           # real estate
    (6600, 6799, "Financial Services"),    # holding & investment offices (REIT overridden)
    (7000, 7099, "Consumer Cyclical"),     # hotels
    (7100, 7399, "Industrials"),           # business services (software/advertising overridden)
    (7400, 7799, "Consumer Cyclical"),     # auto & misc repair/services
    (7800, 7999, "Communication Services"),# motion pictures & recreation (media core)
    (8000, 8099, "Healthcare"),            # health services
    (8100, 8999, "Industrials"),           # legal, education, engineering, research, mgmt
    (9000, 9999, "Industrials"),           # public administration / nonclassifiable
)


def sic_to_sector(sic) -> str:
    """Map a 4-digit SIC code to one of the 11 canonical sectors. Total: any input resolves
    (invalid/None -> DEFAULT_SECTOR). Specific-code overrides beat the coarse major-group range."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return DEFAULT_SECTOR
    if code in _OVERRIDES:
        return _OVERRIDES[code]
    # narrowest matching range wins (petroleum 2900-2999 over chemicals 2800-2999)
    best = None
    for lo, hi, sector in _RANGES:
        if lo <= code <= hi:
            span = hi - lo
            if best is None or span < best[0]:
                best = (span, sector)
    return best[1] if best else DEFAULT_SECTOR
