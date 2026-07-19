"""
forecast_rain_prob_parser.py

Segment-based HKO weather description parser.
Breaks compound descriptions by sentence terminators (。),
detects time qualifiers (初時/下午/稍後/…), maps weather keywords
to rain-intensity levels (0-5), and produces time-bucket features.

Time-bucket design:
  - morning (06-12): rain before peak-heating hours → low tmax impact
  - afternoon (12-18): rain during tmax window → high tmax impact
  - overall: worst-case rain level across all segments

Weight design:
  - 初時/早上 → 0.4 (rain passes before tmax)
  - (no qualifier) → 0.7 (anytime rain)
  - 下午 → 1.0 (direct tmax impact)
  - 稍後 → 0.85 (second half, likely afternoon)
  - 日間 → 0.7 (daytime coverage)
  - 晚間 → 0.3 (tmax already decided)

Example:
  Input:  「初時部分地區驟雨較多及有狂風雷暴。下午短暫時間有陽光。」
  Output: {'forecast_rain_prob_morning': 5.0,
           'forecast_rain_prob_afternoon': 2.0,
           'forecast_rain_prob_overall': 5.0,
           'forecast_rain_prob_missing': 0}
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

# ── Time qualifier → (bucket_id, tmax_weight) ──────────────────────────
# bucket_id: 1=morning, 2=daytime, 3=afternoon, 4=later, 5=evening
TIME_KEYWORDS: Dict[str, tuple[int, float]] = {
    "初時": (1, 0.4),
    "早上": (1, 0.4),
    "上午": (1, 0.4),
    "下午": (3, 1.0),
    "稍後": (4, 0.85),
    "日間": (2, 0.7),
    "晚間": (5, 0.3),
    "今晚": (5, 0.3),
    "明早": (1, 0.4),
}

# ── Weather keywords → rain intensity level (0-5) ──────────────────────
# Level 1: clear/sunny (no rain impact)
# Level 2: partly cloudy / brief sun
# Level 3: cloudy / mist / drizzle
# Level 4: showers / rain
# Level 5: thunderstorm / heavy rain

# Ordered from most specific → least specific per level
WEATHER_PATTERNS: list[tuple[re.Pattern, int, float]] = [
    # Level 5 — thunderstorm / heavy rain
    (re.compile(r"狂風雷暴|雷[雨暴]|大[到至]?雨|暴雨|特大暴雨"), 5, 1.0),
    # Level 4 — showers / rain
    (re.compile(r"驟雨|陣雨|有幾陣雨|間中有雨|零散驟雨|微雨|雨"), 4, 1.0),
    # Level 3 — cloudy / mist / drizzle
    (re.compile(r"多雲|毛毛雨|薄霧|煙霞|有霧"), 3, 0.8),
    # Level 2 — partly cloudy
    (re.compile(r"部分時間有陽光|短暫時間有陽光|天色明朗|部分天色明朗"), 2, 0.5),
    # Level 1 — sunny / clear
    (re.compile(r"天晴乾燥|陽光充沛|天晴|大致天晴|大致晴朗"), 1, 0.1),
]

# ── Segment delimiters ─────────────────────────────────────────────────
# Chinese sentence terminators: 。；!
_SEGMENT_DELIM = re.compile(r"[。；!]+")


def _detect_time_bucket(segment: str) -> tuple[Optional[str], float]:
    """Detect the time qualifier in a segment.

    Returns:
        (bucket_key, tmax_weight)
        bucket_key: 'morning', 'afternoon', 'overall', or None for 'overall'
        weight: multiplier applied to rain level
    """
    for kw, (bucket_id, weight) in TIME_KEYWORDS.items():
        if kw in segment:
            if bucket_id == 1:
                return ("morning", weight)
            elif bucket_id == 3:
                return ("afternoon", weight)
            elif bucket_id == 4:
                # 稍後 — context-dependent: could be afternoon or evening
                # Default to afternoon (conservative for tmax)
                return ("afternoon", weight)
            elif bucket_id == 5:
                return ("evening", weight)
            elif bucket_id == 2:
                return ("overall", weight)
    return (None, 0.7)  # no qualifier → overall


def _detect_rain_level(segment: str) -> int:
    """Return the highest matching rain intensity level (0-5) in a segment."""
    best_level = 0
    for pattern, level, _ in WEATHER_PATTERNS:
        if pattern.search(segment):
            if level > best_level:
                best_level = level
    return best_level


def parse_hko_description(desc: object) -> Dict[str, float]:
    """Parse a single HKO weather description into time-bucket features.

    Args:
        desc: Chinese weather description string from forecast_rain_prob column.

    Returns:
        dict with keys:
            forecast_rain_prob_morning:   max rain level (×weight) in morning segments (0-5)
            forecast_rain_prob_afternoon: max rain level (×weight) in afternoon segments (0-5)
            forecast_rain_prob_overall:   max rain level across ALL segments (0-5)
            forecast_rain_prob_missing:   1 if desc is NaN/empty, else 0
    """
    # Handle missing
    if desc is None or (isinstance(desc, float) and pd.isna(desc)):
        return {
            "forecast_rain_prob_morning": 0.0,
            "forecast_rain_prob_afternoon": 0.0,
            "forecast_rain_prob_overall": 0.0,
            "forecast_rain_prob_missing": 1.0,
        }

    desc = str(desc).strip()
    if not desc:
        return {
            "forecast_rain_prob_morning": 0.0,
            "forecast_rain_prob_afternoon": 0.0,
            "forecast_rain_prob_overall": 0.0,
            "forecast_rain_prob_missing": 1.0,
        }

    # Split into segments
    segments = [s.strip() for s in _SEGMENT_DELIM.split(desc) if s.strip()]

    # Per-bucket raw rain levels (before weight)
    buckets: Dict[str, list[float]] = {"morning": [], "afternoon": [], "overall": [], "evening": []}

    for seg in segments:
        bucket_key, weight = _detect_time_bucket(seg)
        rain_level = _detect_rain_level(seg)
        if rain_level == 0:
            rain_level = 1  # no rain terms found → assume clear (level 1)
        weighted = float(rain_level) * weight
        key = bucket_key if bucket_key else "overall"
        buckets[key].append(weighted)

    # Aggregate per bucket
    morning_max = max(buckets["morning"]) if buckets["morning"] else 0.0
    afternoon_max = max(buckets["afternoon"]) if buckets["afternoon"] else 0.0

    # Overall = worst weighted across ALL segments
    all_vals = (
        buckets["morning"] + buckets["afternoon"] + buckets["overall"] + buckets["evening"]
    )
    overall_max = max(all_vals) if all_vals else 0.0

    return {
        "forecast_rain_prob_morning": round(morning_max, 2),
        "forecast_rain_prob_afternoon": round(afternoon_max, 2),
        "forecast_rain_prob_overall": round(overall_max, 2),
        "forecast_rain_prob_missing": 0.0,
    }


def parse_hko_description_series(series: pd.Series) -> pd.DataFrame:
    """Apply parse_hko_description to a Series of weather descriptions.

    Returns a DataFrame with the 4 output columns.
    """
    results = series.apply(parse_hko_description)
    return pd.DataFrame(results.tolist())


# ── Column auto-detection ─────────────────────────────────────────────

_KNOWN_PROB_LABELS = frozenset({"低", "中低", "中", "中高", "高", "官方天氣稿"})

_WEATHER_KEYWORDS = frozenset({
    "天晴", "陽光", "驟雨", "雷暴", "多雲", "薄霧", "煙霞", "微雨",
    "毛毛雨", "陣雨", "大雨", "暴雨", "天色", "有霧", "晴朗",
})


def _is_weather_description(val: object) -> bool:
    """Check if a value looks like a weather description (not a rain-prob label)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if s in _KNOWN_PROB_LABELS:
        return False
    # If it contains known weather keywords, it's a description
    for kw in _WEATHER_KEYWORDS:
        if kw in s:
            return True
    # If it's long enough (contains Chinese sentence structure), it's a description
    if len(s) >= 6 and "，" in s:
        return True
    return False


def resolve_weather_description(
    rain_prob_col: pd.Series,
    weather_desc_col: pd.Series,
) -> pd.Series:
    """Resolve which column holds the weather description.

    Handles HKO schema where the column meaning swapped over time:
    - Older rows: forecast_rain_prob has description, forecast_weather_desc has label
    - Newer rows: forecast_weather_desc has description, forecast_rain_prob has label

    Returns a Series of weather descriptions (NaN where neither column has one).
    """
    result = pd.Series(index=rain_prob_col.index, dtype=object)

    for idx in rain_prob_col.index:
        rp = rain_prob_col.iloc[idx]
        wd = weather_desc_col.iloc[idx]

        if _is_weather_description(rp):
            result.iloc[idx] = str(rp).strip()
        elif _is_weather_description(wd):
            result.iloc[idx] = str(wd).strip()
        else:
            # Fallback: use whichever is not NaN and not a label
            for v in [rp, wd]:
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    s = str(v).strip()
                    if s and s not in _KNOWN_PROB_LABELS:
                        result.iloc[idx] = s
                        break
            else:
                result.iloc[idx] = None

    return result


def resolve_and_parse(
    rain_prob_col: pd.Series,
    weather_desc_col: pd.Series,
) -> pd.DataFrame:
    """Resolve the weather description column and parse it in one step.

    Returns a DataFrame with the 4 parsed columns.
    """
    descriptions = resolve_weather_description(rain_prob_col, weather_desc_col)
    return parse_hko_description_series(descriptions)


# ── Validation helpers ──────────────────────────────────────────────────

_RAIN_PROB_LEVEL_MAP: Dict[str, int] = {
    "低": 1,
    "中低": 2,
    "中": 3,
    "中高": 4,
    "高": 5,
}


def map_weather_desc_to_ordinal(val: object) -> Optional[int]:
    """Map forecast_weather_desc rain probability label to ordinal 1-5.

    Returns 0 for non-probability values (e.g. '官方天氣稿').
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return _RAIN_PROB_LEVEL_MAP.get(str(val).strip(), 0)
    except Exception:
        return 0


def validate_parser(forecast_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Validate parser output against ground-truth rain probability labels.

    Uses auto-detection to find weather descriptions and parses them.
    Compares parsed overall score against label ordinal from forecast_weather_desc/forecast_rain_prob.

    Args:
        forecast_df: DataFrame with columns 'forecast_rain_prob', 'forecast_weather_desc'.

    Returns:
        dict of validation metrics.
    """
    df = forecast_df.copy()

    # Resolve descriptions
    descriptions = resolve_weather_description(
        df["forecast_rain_prob"], df["forecast_weather_desc"]
    )
    parsed = parse_hko_description_series(descriptions)
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    # Ground truth: check BOTH columns for probability labels
    df["gt_ordinal"] = df["forecast_weather_desc"].apply(map_weather_desc_to_ordinal)
    # If forecast_weather_desc is not a label, try forecast_rain_prob
    mask = df["gt_ordinal"] == 0
    df.loc[mask, "gt_ordinal"] = df.loc[mask, "forecast_rain_prob"].apply(map_weather_desc_to_ordinal)

    gt = df[df["gt_ordinal"] > 0].copy()
    if verbose:
        print(f"  Rows with weather description: {descriptions.notna().sum()} / {len(df)}")
        print(f"  Ground-truth rows with prob label: {len(gt)} / {len(df)}")

    if len(gt) == 0:
        return {"n_gt": 0, "error": "No ground-truth rows found"}

    # Correlation
    corr_overall = gt["forecast_rain_prob_overall"].corr(gt["gt_ordinal"])

    # Accuracy: round parsed overall to nearest integer, check if matches ordinal
    gt["parsed_level"] = gt["forecast_rain_prob_overall"].round().clip(1, 5).astype(int)
    exact_match = (gt["parsed_level"] == gt["gt_ordinal"]).mean()
    within_one = (gt["parsed_level"] - gt["gt_ordinal"]).abs().le(1).mean()

    if verbose:
        print(f"  Correlation (overall vs ground-truth): {corr_overall:.4f}")
        print(f"  Exact match rate: {exact_match:.1%}")
        print(f"  Within-1 accuracy: {within_one:.1%}")

        # Show confusion matrix
        cm = pd.crosstab(gt["gt_ordinal"], gt["parsed_level"], margins=True)
        print(f"\n  Confusion matrix (gt vs parsed):")
        print(cm.to_string())

    return {
        "n_gt": len(gt),
        "n_descriptions": int(descriptions.notna().sum()),
        "correlation": round(float(corr_overall), 4) if not pd.isna(corr_overall) else 0.0,
        "exact_match_rate": round(float(exact_match), 4),
        "within_one_rate": round(float(within_one), 4),
    }
