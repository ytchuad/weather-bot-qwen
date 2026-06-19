export interface PredictionResult {
  date: string
  models: Record<string, ModelPrediction>
}

export interface ModelPrediction {
  mean: number
  std: number
  source: string
  probs: Record<string, number> | null
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
