const BASE = "/api"

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
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
  Strategy,
  StrategyCreate,
  StrategyUpdate,
  PortfolioStats,
  StrategyChartData,
  ModelsComparisonData,
  BucketProbsData,
  MinuteHistoryResponse,
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

export function fetchTodayEvent(date?: string, isMinTemp = false): Promise<{ event: EventSearchResponse["events"][number] }> {
  const q = date ? `?date=${encodeURIComponent(date)}&is_min_temp=${isMinTemp}` : ""
  return get(`/markets/today-event${q}`)
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

// Strategy endpoints
export function getPortfolio(): Promise<PortfolioStats> {
  return get("/strategies/portfolio")
}

export function getStrategies(): Promise<{ strategies: Strategy[] }> {
  return get("/strategies")
}

export function createStrategy(req: StrategyCreate): Promise<{ status: string; strategy: Strategy }> {
  return post("/strategies", req)
}

export function updateStrategy(
  id: string,
  req: StrategyUpdate,
): Promise<{ status: string; strategy: Strategy }> {
  return fetch(`${BASE}/strategies/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((res) => {
    if (!res.ok) throw new Error(`PATCH strategies/${id}: ${res.status}`)
    return res.json()
  })
}

export function resetStrategy(id: string): Promise<{ status: string; strategy: Strategy }> {
  return post(`/strategies/${id}/reset`, {})
}

export function deleteStrategy(id: string): Promise<{ status: string; id: string }> {
  return fetch(`${BASE}/strategies/${id}`, { method: "DELETE" }).then((res) => {
    if (!res.ok) throw new Error(`DELETE strategies/${id}: ${res.status}`)
    return res.json()
  })
}

// Run all enabled strategies
export function runAllStrategies(): Promise<{ results: unknown[]; total: number }> {
  return post("/strategies/run-all", {})
}

// Models comparison chart (Hub page)
export function fetchModelsComparison(date: string): Promise<ModelsComparisonData> {
  return get(`/charts/models-comparison?date=${encodeURIComponent(date)}`, { cache: "no-store" })
}

// Bucket probability time-series chart
export function fetchBucketProbs(date: string, bucket?: string): Promise<BucketProbsData> {
  let q = `date=${encodeURIComponent(date)}`
  if (bucket) q += `&bucket=${encodeURIComponent(bucket)}`
  return get(`/charts/bucket-probs?${q}`)
}

export function fetchMinuteHistory(date: string, limit = 1000, signal?: AbortSignal): Promise<MinuteHistoryResponse> {
  return get(`/history/minute?date=${encodeURIComponent(date)}&limit=${limit}`, {
    cache: "no-store",
    signal,
  })
}

// Strategy chart
export function fetchStrategyChart(
  sid: string,
  date: string,
  slug?: string,
): Promise<StrategyChartData> {
  let q = `date=${encodeURIComponent(date)}`
  if (slug) q += `&slug=${encodeURIComponent(slug)}`
  return get(`/strategies/${sid}/chart?${q}`)
}


