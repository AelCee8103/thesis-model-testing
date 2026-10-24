import { useCallback, useRef, useState } from "react";

import { predict, evaluate, ApiError } from "@/lib/api";
import { runPool, type PoolOutcome } from "@/lib/pool";
import type { EvalRow } from "@/lib/csv";
import type { Dimension, EvaluateResponse, PredictResponse } from "@/types/api";

const CONCURRENCY = 5;
const RUN_TAG = "paper";

type PredPair = { baseline: PredictResponse; proposed: PredictResponse };

export type RowStatus = "pending" | "running" | "done" | "error";

export interface BatchRowResult {
  index: number;
  text: string;
  truth: number;
  status: RowStatus;
  baseline?: PredictResponse;
  proposed?: PredictResponse;
  delta?: number;
  baselineAbsErr?: number;
  proposedAbsErr?: number;
  error?: string;
}

export interface BatchConfig {
  dataset: string;
  dimension: Dimension;
}

export interface UseBatchCompareResult {
  rows: BatchRowResult[];
  progress: { done: number; total: number };
  running: boolean;
  evaluating: boolean;
  error: string | null;
  evalError: string | null;
  metrics: EvaluateResponse | null;
  run: (rows: EvalRow[], config: BatchConfig) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

export function useBatchCompare(): UseBatchCompareResult {
  const [rows, setRows] = useState<BatchRowResult[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<EvaluateResponse | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    inflight.current?.abort();
    setRunning(false);
    setEvaluating(false);
  }, []);

  const reset = useCallback(() => {
    inflight.current?.abort();
    setRows([]);
    setProgress({ done: 0, total: 0 });
    setRunning(false);
    setEvaluating(false);
    setError(null);
    setEvalError(null);
    setMetrics(null);
  }, []);

  const run = useCallback(async (evalRows: EvalRow[], config: BatchConfig) => {
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;

    setRows(
      evalRows.map((r, index) => ({ index, text: r.text, truth: r.truth, status: "pending" })),
    );
    setProgress({ done: 0, total: evalRows.length });
    setError(null);
    setEvalError(null);
    setMetrics(null);
    setRunning(true);
    setEvaluating(false);

    const patch = (index: number, next: Partial<BatchRowResult>) =>
      setRows((prev) => prev.map((r) => (r.index === index ? { ...r, ...next } : r)));

    let done = 0;
    let outcomes: PoolOutcome<PredPair>[];
    try {
      outcomes = await runPool<EvalRow, PredPair>(
        evalRows,
        CONCURRENCY,
        async (row, index) => {
          patch(index, { status: "running" });
          const body = {
            text: row.text,
            dataset: config.dataset,
            dimension: config.dimension,
            run_tag: RUN_TAG,
          };
          const [baseline, proposed] = await Promise.all([
            predict({ ...body, architecture: "lstm" }, controller.signal),
            predict({ ...body, architecture: "bilstm" }, controller.signal),
          ]);
          return { baseline, proposed };
        },
        {
          signal: controller.signal,
          onSettled: (index, outcome) => {
            if (controller.signal.aborted) return;
            const { truth } = evalRows[index];
            if (outcome.ok) {
              const { baseline, proposed } = outcome.value;
              patch(index, {
                status: "done",
                baseline,
                proposed,
                delta: proposed.score - baseline.score,
                baselineAbsErr: Math.abs(baseline.score - truth),
                proposedAbsErr: Math.abs(proposed.score - truth),
              });
            } else {
              const e = outcome.error;
              const msg =
                e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Request failed";
              patch(index, { status: "error", error: msg });
            }
            done += 1;
            setProgress({ done, total: evalRows.length });
          },
        },
      );
    } catch (e: unknown) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "Batch failed");
        setRunning(false);
      }
      return;
    }

    if (controller.signal.aborted) return;
    setRunning(false);

    // Build aligned arrays from successfully scored rows, then ask the backend for metrics.
    const truth: number[] = [];
    const baselinePred: number[] = [];
    const proposedPred: number[] = [];
    outcomes.forEach((outcome, index) => {
      if (outcome?.ok) {
        truth.push(evalRows[index].truth);
        baselinePred.push(outcome.value.baseline.score);
        proposedPred.push(outcome.value.proposed.score);
      }
    });

    if (truth.length < 2) {
      setEvalError("Need at least 2 successfully scored rows to compute metrics.");
      return;
    }

    setEvaluating(true);
    try {
      const result = await evaluate(
        {
          truth,
          baseline_pred: baselinePred,
          proposed_pred: proposedPred,
          dimension: config.dimension,
        },
        controller.signal,
      );
      if (!controller.signal.aborted) setMetrics(result);
    } catch (e: unknown) {
      if (controller.signal.aborted) return;
      const msg =
        e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Evaluation failed";
      setEvalError(msg);
    } finally {
      if (!controller.signal.aborted) setEvaluating(false);
    }
  }, []);

  return {
    rows,
    progress,
    running,
    evaluating,
    error,
    evalError,
    metrics,
    run,
    cancel,
    reset,
  };
}
