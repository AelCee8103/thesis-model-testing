import { useCallback, useRef, useState } from "react";
import { predict, ApiError } from "@/lib/api";
import type { PredictRequest, PredictResponse } from "@/types/api";

export interface UsePredictResult {
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
  run: (body: PredictRequest) => Promise<void>;
  reset: () => void;
}

export function usePredict(): UsePredictResult {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const run = useCallback(async (body: PredictRequest) => {
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    setLoading(true);
    setError(null);
    try {
      const data = await predict(body, controller.signal);
      if (!controller.signal.aborted) {
        setResult(data);
      }
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      if (err instanceof ApiError) setError(err.detail);
      else if (err instanceof Error) setError(err.message);
      else setError("Prediction failed");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    inflight.current?.abort();
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, run, reset };
}
