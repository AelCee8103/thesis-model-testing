export type Dimension = "valence" | "arousal";
export type Architecture = "lstm" | "bilstm";

export interface ModelInfo {
  dataset: string;
  dataset_description: string;
  language: string;
  dimension: Dimension;
  architecture: Architecture;
  run_tag: string;
  scale_min: number;
  scale_max: number;
  region_depth: number;
  checkpoint_exists: boolean;
}

export interface PredictRequest {
  text: string;
  dataset: string;
  dimension: Dimension;
  architecture: Architecture;
  run_tag: string;
}

export interface PredictResponse {
  score: number;
  score_raw: number;
  scale_min: number;
  scale_max: number;
  dataset: string;
  dimension: Dimension;
  architecture: Architecture;
  run_tag: string;
  region_depth: number;
  region_capacity: number;
  n_regions: number;
  n_tokens_total: number;
  regions: string[][];
  warnings: string[];
}

export interface ApiError {
  detail: string | { msg: string }[];
}

export interface EvaluateRequest {
  truth: number[];
  baseline_pred: number[];
  proposed_pred: number[];
  dimension: Dimension;
}

export interface MetricSet {
  mae: number;
  pearson_r: number;
}

export interface MannWhitneyResult {
  u_statistic: number;
  p_value: number;
  significant: boolean;
}

export interface EvaluateResponse {
  n: number;
  dimension: Dimension;
  baseline: MetricSet;
  proposed: MetricSet;
  mann_whitney_u: MannWhitneyResult;
}
