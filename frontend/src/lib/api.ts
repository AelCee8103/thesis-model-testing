import type {
  EvaluateRequest,
  EvaluateResponse,
  ModelInfo,
  PredictRequest,
  PredictResponse,
} from "@/types/api";

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail.map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d)).join("; ");
    }
    return res.statusText || `HTTP ${res.status}`;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function fetchHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE_URL}/models`, { signal });
  if (!res.ok) {
    throw new ApiError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as ModelInfo[];
}

export async function predict(body: PredictRequest, signal?: AbortSignal): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as PredictResponse;
}

export async function evaluate(
  body: EvaluateRequest,
  signal?: AbortSignal,
): Promise<EvaluateResponse> {
  const res = await fetch(`${BASE_URL}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as EvaluateResponse;
}

export const API_BASE_URL = BASE_URL;
