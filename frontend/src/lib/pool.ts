// Bounded-concurrency async pool. Each item is processed by `worker`; one item's
// failure is captured as an outcome and never rejects the whole batch.

export type PoolOutcome<R> = { ok: true; value: R } | { ok: false; error: unknown };

export async function runPool<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
  opts?: {
    signal?: AbortSignal;
    onSettled?: (index: number, outcome: PoolOutcome<R>) => void;
  },
): Promise<PoolOutcome<R>[]> {
  const results: PoolOutcome<R>[] = new Array(items.length);
  const lanes = Math.max(1, Math.min(limit, items.length));
  let cursor = 0;

  async function lane(): Promise<void> {
    while (true) {
      if (opts?.signal?.aborted) return;
      const index = cursor++;
      if (index >= items.length) return;

      let outcome: PoolOutcome<R>;
      try {
        outcome = { ok: true, value: await worker(items[index], index) };
      } catch (error) {
        outcome = { ok: false, error };
      }
      results[index] = outcome;
      opts?.onSettled?.(index, outcome);
    }
  }

  await Promise.all(Array.from({ length: lanes }, () => lane()));
  return results;
}
