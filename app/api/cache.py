from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from cachetools import TTLCache

logger = logging.getLogger(__name__)

_weather_cache = TTLCache(maxsize=32, ttl=60)
_prediction_cache = TTLCache(maxsize=32, ttl=300)
_market_cache = TTLCache(maxsize=32, ttl=120)


def _make_key(func: Callable, args: tuple, kwargs: dict) -> str:
    return f"{func.__name__}:{args}:{sorted(kwargs.items())}"


def with_cache(cache: TTLCache) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(func, args, kwargs)
            if key in cache:
                cached_result = cache[key]
                # Return cached result even if it's None or invalid
                return cached_result
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator


def weather_cache(func: Callable) -> Callable:
    return with_cache(_weather_cache)(func)


def prediction_cache(func: Callable) -> Callable:
    return with_cache(_prediction_cache)(func)


def market_cache(func: Callable) -> Callable:
    return with_cache(_market_cache)(func)
