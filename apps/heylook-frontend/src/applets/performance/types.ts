export type TimeRange = '1h' | '6h' | '24h' | '7d'

export interface TimingOperation {
  operation: string
  avg_time_ms: number
  count: number
  percentage: number
}

export interface ResourceTimepoint {
  timestamp: string
  memory_gb: number
  gpu_percent: number
  tokens_per_second: number
  requests: number
}

export interface ModelBottleneck {
  model: string
  avg_total_ms: number
  breakdown: {
    queue: number
    model_load: number
    image_processing: number
    token_generation: number
    first_token: number
  }
  request_count: number
}

export interface PerformanceTrend {
  hour: string
  response_time_ms: number
  tokens_per_second: number
  requests: number
  errors: number
  response_time_change: number
  tps_change: number
}

export interface PerformanceProfile {
  time_range: string
  timing_breakdown: TimingOperation[]
  resource_timeline: ResourceTimepoint[]
  bottlenecks: ModelBottleneck[]
  trends: PerformanceTrend[]
}
