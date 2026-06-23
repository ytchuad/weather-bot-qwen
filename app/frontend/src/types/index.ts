export interface PredictionResult {
  date: string
  models: Record<string, ModelPrediction>
}

export interface ModelPrediction {
  mean: number
  std: number
  source: string
  probs: Record<string, number> | null
  degraded?: boolean
}

export interface WeatherNow {
  date?: string
  temp: number | null
  humidity: number | null
  max_today: number | null
  min_today: number | null
  forecast: number | null
  aws_temp: number | null
  source: string
  fetched_at: string
  rain_60m: number | null
  rain_120m: number | null
  rain_accumulated_today: number | null
  rain_nowcast: number | null
}

export interface RainfallData {
  date: string
  rain_60m: number
  rain_120m: number
  rain_data_ok: boolean
  rainfall_60m_missing_flag: number
  rainfall_120m_missing_flag: number
}

export interface Suggestion {
  bucket: string
  market_price: number
  model_prob: number
  edge: number
  kelly_fraction: number
  action: "buy_yes" | "buy_no" | "pass"
}

export interface SuggestResponse {
  date: string
  suggestions: Suggestion[]
}

export interface BucketDef {
  type: string
  buckets: string[]
}

export interface EventMarket {
  slug: string
  title: string
  markets: Record<string, unknown>[]
  prices?: Record<string, number>
}

export interface EventSearchResponse {
  events: Array<{ slug: string; title: string }>
}

// Strategy Types
export interface Strategy {
  id: string
  label: string
  model: string
  capital: number
  initial_capital: number
  market_template: string
  status: string
  scheduler_on: boolean
  last_run: string | null
  params: {
    bias: number
    std_mult: number
    kelly_fraction: number
  }
  from_strategy_key?: string | null
  gate_config_override?: Record<string, unknown> | null
}

export interface StrategyCreate {
  id: string
  label: string
  model?: string
  capital?: number
  initial_capital?: number
  market_template?: string
  from_strategy_key?: string | null
}

export interface StrategyUpdate {
  status?: string
  scheduler_on?: boolean
  capital?: number
  params?: Record<string, unknown>
}

export interface PortfolioStats {
  total_capital: number
  total_pnl: number
  total_return_pct: number
  active_strategies: number
  count: number
}

export interface StrategyTrade {
  timestamp?: string
  entry_time?: string
  model_key?: string
  selected_model?: string
  slug?: string
  bucket: string
  action?: string
  side?: string
  side_after?: string
  qty_after?: number
  target_price?: number
  market_prices?: string
  entry_reason?: string
  exit_reason?: string
  entry_price?: number
  exit_price?: number | null
  quantity?: number
  pnl?: number | null
  reason?: string | null
  reason_code?: string
}
