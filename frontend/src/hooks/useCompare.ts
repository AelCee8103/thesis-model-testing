import { useCallback, useRef, useState } from "react";
import { predict, ApiError } from "@/lib/api";
import type { PredictRequest, PredictResponse } from "@/types/api";

export interface CompareResult {
  baseline: PredictResponse;
  proposed: PredictResponse;
}

export type CompareRequest = Omit<PredictRequest, "architecture">;

export interface UseCompareResult {
  result: CompareResult | null;
  loading: boolean;
  error: string | null;
  run: (body: CompareRequest) => Promise<void>;
  reset: () => void;
}

export function useCompare(): UseCompareResult {
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const run = useCallback(async (body: CompareRequest) => {
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    setLoading(true);
    setError(null);
    try {
      const [baseline, proposed] = await Promise.all([
        predict({ ...body, architecture: "lstm" }, controller.signal),
        predict({ ...body, architecture: "bilstm" }, controller.signal),
      ]);
      if (!controller.signal.aborted) {
        setResult({ baseline, proposed });
      }
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      if (err instanceof ApiError) setError(err.detail);
      else if (err instanceof Error) setError(err.message);
      else setError("Comparison failed");
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
