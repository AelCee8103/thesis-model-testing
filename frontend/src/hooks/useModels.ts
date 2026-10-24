import { useEffect, useState } from "react";
import { fetchModels, ApiError } from "@/lib/api";
import type { ModelInfo } from "@/types/api";

export interface UseModelsResult {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useModels(): UseModelsResult {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchModels(controller.signal)
      .then((data) => setModels(data))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        if (err instanceof ApiError) setError(err.detail);
        else if (err instanceof Error) setError(err.message);
        else setError("Failed to fetch models");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [nonce]);

  return { models, loading, error, refetch: () => setNonce((n) => n + 1) };
}
