import { AlertCircle, ArrowDown, ArrowRight, ArrowUp, Gauge, Hash, Layers, Sparkles } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { ScoreGauge } from "@/components/score-gauge";
import { WarningsList } from "@/components/warnings-list";
import { RegionsView } from "@/components/regions-view";
import { cn } from "@/lib/utils";
import type { Dimension, PredictResponse } from "@/types/api";
import type { CompareResult } from "@/hooks/useCompare";

interface ResultPanelProps {
  result: CompareResult | null;
  loading: boolean;
  error: string | null;
}

function MetaTile({
  icon: Icon,
  label,
  value,
  mono,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-border/60 bg-muted/30 p-3 transition-colors hover:bg-muted/50">
      <Icon className="mt-0.5 size-4 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          {label}
        </div>
        <div className={mono ? "font-mono text-sm text-foreground" : "text-sm text-foreground"}>
          {value}
        </div>
      </div>
    </div>
  );
}

function interpretScore(
  score: number,
  min: number,
  max: number,
  dimension: Dimension,
): { label: string; tone: "low" | "mid-low" | "mid" | "mid-high" | "high" } {
  const range = Math.max(1e-9, max - min);
  const pct = (Math.max(min, Math.min(max, score)) - min) / range;
  if (dimension === "valence") {
    if (pct >= 0.8) return { label: "Strongly positive", tone: "high" };
    if (pct >= 0.6) return { label: "Mildly positive", tone: "mid-high" };
    if (pct > 0.4) return { label: "Neutral", tone: "mid" };
    if (pct > 0.2) return { label: "Mildly negative", tone: "mid-low" };
    return { label: "Strongly negative", tone: "low" };
  }
  if (pct >= 0.8) return { label: "Highly aroused", tone: "high" };
  if (pct >= 0.6) return { label: "Activated", tone: "mid-high" };
  if (pct > 0.4) return { label: "Moderate", tone: "mid" };
  if (pct > 0.2) return { label: "Calm", tone: "mid-low" };
  return { label: "Very calm", tone: "low" };
}

const TONE_CLASSES: Record<string, string> = {
  high: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 ring-emerald-500/20",
  "mid-high": "bg-emerald-500/5 text-emerald-700/80 dark:text-emerald-300/80 ring-emerald-500/15",
  mid: "bg-muted text-muted-foreground ring-border",
  "mid-low": "bg-amber-500/5 text-amber-700/80 dark:text-amber-300/80 ring-amber-500/15",
  low: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
};

function ModelColumn({
  result,
  variant,
  delta,
}: {
  result: PredictResponse;
  variant: "baseline" | "proposed";
  delta?: number;
}) {
  const interp = interpretScore(result.score, result.scale_min, result.scale_max, result.dimension);
  const isProposed = variant === "proposed";
  const showDelta = delta !== undefined && Math.abs(delta) > 1e-4;

  return (
    <div
      className={cn(
        "rounded-lg border p-4 space-y-3",
        isProposed ? "border-primary/40 bg-primary/[0.03]" : "border-border bg-muted/20",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          {isProposed ? "New Model" : "Baseline"}
        </span>
        <Badge
          variant={isProposed ? "default" : "secondary"}
          className="text-[10px] uppercase tracking-wider"
        >
          {result.architecture === "bilstm" ? "BiLSTM" : "LSTM"} · {result.run_tag}
        </Badge>
      </div>

      <div className="flex items-baseline gap-2">
        <span className="font-serif text-4xl font-semibold tabular-nums text-foreground sm:text-5xl">
          {result.score.toFixed(2)}
        </span>
        {showDelta && isProposed && (
          <span
            className={cn(
              "inline-flex items-center text-xs font-medium tabular-nums",
              delta! > 0
                ? "text-emerald-600 dark:text-emerald-400"
                : "text-rose-600 dark:text-rose-400",
            )}
          >
            {delta! > 0 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />}
            {delta! > 0 ? "+" : ""}
            {delta!.toFixed(2)}
          </span>
        )}
      </div>

      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
          TONE_CLASSES[interp.tone],
        )}
      >
        {interp.label}
      </span>

      <ScoreGauge
        score={result.score}
        min={result.scale_min}
        max={result.scale_max}
        dimension={result.dimension}
      />

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>raw</span>
        <span className="font-mono tabular-nums text-foreground/80">{result.score_raw.toFixed(4)}</span>
      </div>
    </div>
  );
}

function dedupe(strings: string[]): string[] {
  return Array.from(new Set(strings));
}

export function ResultPanel({ result, loading, error }: ResultPanelProps) {
  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-xl">Comparison</CardTitle>
        <CardDescription>
          Side-by-side prediction from the baseline LSTM and the proposed BiLSTM on the same input — same dataset, same regions, same scale.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>Request failed</AlertTitle>
            <AlertDescription className="break-words">{error}</AlertDescription>
          </Alert>
        )}

        {loading && !result && (
          <div className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <Skeleton className="h-40" />
              <Skeleton className="h-40" />
            </div>
            <Skeleton className="h-5 w-full" />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          </div>
        )}

        {!loading && !result && !error && (
          <div className="flex h-60 flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/30 text-center">
            <Sparkles className="mb-3 size-6 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">Waiting for input</p>
            <p className="mt-1 max-w-[28ch] text-xs text-muted-foreground">
              Submit a sentence above to compare the baseline LSTM and the proposed BiLSTM on the same input.
            </p>
          </div>
        )}

        {result && (() => {
          const { baseline, proposed } = result;
          const delta = proposed.score - baseline.score;
          const showDelta = Math.abs(delta) > 1e-4;
          const verdict =
            !showDelta
              ? { tone: "mid", label: "Models agree" }
              : delta > 0
                ? { tone: "high", label: "BiLSTM scores higher" }
                : { tone: "low", label: "LSTM scores higher" };

          return (
            <>
              <div className="rounded-lg border border-border/60 bg-muted/20 px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
                    Predicted {baseline.dimension} · on [{baseline.scale_min}, {baseline.scale_max}]
                  </div>
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
                      TONE_CLASSES[verdict.tone],
                    )}
                  >
                    {verdict.label}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap items-baseline gap-3 font-mono tabular-nums">
                  <span className="text-base text-muted-foreground">{baseline.score.toFixed(2)}</span>
                  <ArrowRight className="size-4 text-muted-foreground" />
                  <span className="text-base text-foreground">{proposed.score.toFixed(2)}</span>
                  {showDelta && (
                    <span
                      className={cn(
                        "ml-1 inline-flex items-center text-xs font-semibold",
                        delta > 0
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-rose-600 dark:text-rose-400",
                      )}
                    >
                      Δ {delta > 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <ModelColumn result={baseline} variant="baseline" />
                <ModelColumn result={proposed} variant="proposed" delta={delta} />
              </div>

              <Separator />

              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <MetaTile icon={Layers} label="Dataset" value={baseline.dataset} mono />
                <MetaTile icon={Gauge} label="Dimension" value={baseline.dimension} />
                <MetaTile icon={Hash} label="Region depth" value={String(baseline.region_depth)} mono />
                <MetaTile
                  icon={Hash}
                  label="Regions / capacity"
                  value={`${baseline.n_regions} / ${baseline.region_capacity}`}
                  mono
                />
                <MetaTile icon={Hash} label="Tokens total" value={String(baseline.n_tokens_total)} mono />
                <MetaTile
                  icon={Sparkles}
                  label="Raw output (LSTM → BiLSTM)"
                  value={`${baseline.score_raw.toFixed(4)} → ${proposed.score_raw.toFixed(4)}`}
                  mono
                />
              </div>

              <RegionsView
                regions={baseline.regions}
                depth={baseline.region_depth}
                capacity={baseline.region_capacity}
              />

              <WarningsList warnings={dedupe([...baseline.warnings, ...proposed.warnings])} />
            </>
          );
        })()}
      </CardContent>
    </Card>
  );
}
