"""Temporal parsing and relevance scoring for memory retrieval."""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s][^\s,;?]+)?\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\s+(19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime
    source: str


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def parse_datetime(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _utc(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return _utc(datetime.fromisoformat(text))
    except ValueError:
        pass
    match = _ISO_DATE_RE.search(text)
    if match:
        try:
            return datetime(*(int(part) for part in match.groups()), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def extract_time_range(query: str, now: Optional[datetime] = None) -> Optional[TimeRange]:
    """Extract a conservative absolute/relative interval from a query."""
    now = _utc(now or datetime.now(timezone.utc))
    text = query.lower()

    match = _ISO_DATE_RE.search(query)
    if match:
        try:
            start = datetime(*(int(part) for part in match.groups()), tzinfo=timezone.utc)
            return TimeRange(start, start + timedelta(days=1), match.group(0))
        except ValueError:
            return None

    month = _MONTH_RE.search(query)
    if month:
        month_number = _MONTHS[month.group(1).lower()]
        year = int(month.group(2))
        start = datetime(year, month_number, 1, tzinfo=timezone.utc)
        end = datetime(year + (month_number == 12), 1 if month_number == 12 else month_number + 1, 1, tzinfo=timezone.utc)
        return TimeRange(start, end, month.group(0))

    year = _YEAR_RE.search(query)
    if year:
        value = int(year.group(1))
        return TimeRange(
            datetime(value, 1, 1, tzinfo=timezone.utc),
            datetime(value + 1, 1, 1, tzinfo=timezone.utc),
            year.group(0),
        )

    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "yesterday" in text:
        return TimeRange(day_start - timedelta(days=1), day_start, "yesterday")
    if "today" in text:
        return TimeRange(day_start, day_start + timedelta(days=1), "today")
    if "last week" in text:
        return TimeRange(now - timedelta(days=7), now, "last week")
    if "last month" in text:
        return TimeRange(now - timedelta(days=30), now, "last month")
    if "last year" in text:
        return TimeRange(now - timedelta(days=365), now, "last year")
    return None


def temporal_relevance(
    value,
    target: Optional[TimeRange],
    half_life_days: float = 30.0,
) -> float:
    """Score an event timestamp against a query interval in [0, 1]."""
    if target is None:
        return 0.0
    event = parse_datetime(value)
    if event is None:
        return 0.0
    if target.start <= event < target.end:
        return 1.0
    distance = min(abs((event - target.start).total_seconds()), abs((event - target.end).total_seconds())) / 86400.0
    return math.exp(-math.log(2.0) * distance / max(half_life_days, 1e-6))


def validity_relevance(node: dict, target: Optional[TimeRange], now=None) -> float:
    """Prefer the fact version valid at the queried time (or now)."""
    valid_from = parse_datetime(node.get("valid_from"))
    valid_until = parse_datetime(node.get("valid_until"))
    if target is not None:
        # Half-open interval overlap: [valid_from, valid_until) intersects
        # [target.start, target.end). This correctly handles a fact that became
        # valid part-way through a month/year query interval.
        if valid_until is not None and valid_until <= target.start:
            return 0.1
        if valid_from is not None and valid_from >= target.end:
            return 0.1
        return 1.0

    reference = parse_datetime(now) or datetime.now(timezone.utc)
    if valid_from is not None and reference < valid_from:
        return 0.1
    if valid_until is not None and reference >= valid_until:
        return 0.1
    return 1.0
