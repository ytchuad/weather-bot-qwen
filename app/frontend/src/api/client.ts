const BASE = "/api"

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path}: ${res.status}`)
  return res.json()
}

import type {
  PredictionResult,
  WeatherNow,
  SuggestResponse,
  BucketDef,
  EventMarket,
  EventSearchResponse,
} from "../types"

export function fetchPredictions(
  date: string,
  isMinTemp = false,
): Promise<PredictionResult> {
  return get(
    `/predictions?date=${encodeURIComponent(date)}&is_min_temp=${isMinTemp}`,
  )
}

export function fetchWeatherNow(date?: string): Promise<WeatherNow> {
  const q = date ? `?date=${encodeURIComponent(date)}` : ""
  return get(`/weather/now${q}`)
}

export function fetchWeatherIntraday(date: string) {
  return get(`/weather/intraday?date=${encodeURIComponent(date)}`)
}

export function fetchWeatherRain(date: string) {
  return get(`/weather/rain?date=${encodeURIComponent(date)}`)
}

export function fetchBuckets(type = "tmax"): Promise<BucketDef> {
  return get(`/markets/buckets?type=${type}`)
}

export function fetchEvents(date?: string): Promise<EventSearchResponse> {
  const q = date ? `?date=${encodeURIComponent(date)}` : ""
  return get(`/markets/events${q}`)
}

export function fetchEvent(slug: string, isMinTemp = false): Promise<EventMarket> {
  return get(`/markets/event/${slug}?is_min_temp=${isMinTemp}`)
}

export function suggestStrategy(
  date: string,
  capital = 10000,
  kellyFraction = 0.25,
  isMinTemp = false,
): Promise<SuggestResponse> {
  return post("/strategies/suggest", {
    date,
    capital,
    kelly_fraction: kellyFraction,
    is_min_temp: isMinTemp,
  })
}

export function runBacktest(
  initialCapital = 10000,
  kellyFraction = 0.25,
): Promise<{ task_id: string }> {
  return post(
    `/backtest/run?initial_capital=${initialCapital}&kelly_fraction=${kellyFraction}`,
    {},
  )
}

export function getBacktestStatus(taskId: string) {
  return get(`/backtest/status/${taskId}`)
}

export function getBacktestResult(taskId: string) {
  return get(`/backtest/result/${taskId}`)
}

export function checkHealth() {
  return get("/health")
}
