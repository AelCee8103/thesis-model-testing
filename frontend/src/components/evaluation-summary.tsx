import { Activity, BarChart3, Sigma, Target } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { EvaluateResponse, MetricSet } from "@/types/api";

interface EvaluationSummaryProps {
  metrics: EvaluateResponse | null;
  evaluating: boolean;
  doneCount: number;
  errorCount: number;
}

function fmt(n: number, digits = 3): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function fmtP(p: number): string {
  if (!Number.isFinite(p)) return "—";
  return p < 0.0001 ? p.toExponential(2) : p.toFixed(4);
}

function MetricColumn({
  label,
  archLabel,
  metrics,
  variant,
  bestMae,
  bestR,
}: {
  label: string;
  archLabel: string;
  metrics: MetricSet;
  variant: "baseline" | "proposed";
  bestMae: boolean;
  bestR: boolean;
}) {
  const isProposed = variant === "proposed";
  return (
    <div
      className={cn(
        "space-y-3 rounded-lg border p-4",
        isProposed ? "border-primary/40 bg-primary/[0.03]" : "border-border bg-muted/20",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          {label}
        </span>
        <Badge
          variant={isProposed ? "default" : "secondary"}
          className="text-[10px] uppercase tracking-wider"
        >
          {archLabel}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-0.5">
          <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            <Target className="size-3" /> MAE
          </div>
          <div
            className={cn(
              "font-serif text-2xl font-semibold tabular-nums",
              bestMae ? "text-emerald-600 dark:text-emerald-400" : "text-foreground",
            )}
          >
            {fmt(metrics.mae)}
          </div>
          <div className="text-[10px] text-muted-foreground">lower is better</div>
        </div>
        <div className="space-y-0.5">
          <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            <Activity className="size-3" /> Pearson r
          </div>
          <div
            className={cn(
              "font-serif text-2xl font-semibold tabular-nums",
              bestR ? "text-emerald-600 dark:text-emerald-400" : "text-foreground",
            )}
          >
            {fmt(metrics.pearson_r)}
          </div>
          <div className="text-[10px] text-muted-foreground">higher is better</div>
        </div>
      </div>
    </div>
  );
}

export function EvaluationSummary({
  metrics,
  evaluating,
  doneCount,
  errorCount,
}: EvaluationSummaryProps) {
  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-xl">Evaluation</CardTitle>
        <CardDescription>
          Accuracy vs ground truth — MAE and Pearson r per model, plus a Mann-Whitney U test on
          the absolute errors.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {evaluating && (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Skeleton className="h-28" />
              <Skeleton className="h-28" />
            </div>
            <Skeleton className="h-20" />
          </div>
        )}

        {!evaluating && !metrics && (
          <div className="flex h-40 flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/30 text-center">
            <BarChart3 className="mb-3 size-6 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">No metrics yet</p>
            <p className="mt-1 max-w-[34ch] text-xs text-muted-foreground">
              Upload a labelled CSV and run the evaluation to compute MAE, Pearson r, and the
              significance test.
            </p>
          </div>
        )}

        {!evaluating && metrics && (() => {
          const { baseline, proposed, mann_whitney_u: mwu, n } = metrics;
          const baselineBetterMae = baseline.mae <= proposed.mae;
          const proposedBetterR = proposed.pearson_r >= baseline.pearson_r;
          const betterLabel = baselineBetterMae ? "LSTM (baseline)" : "BiLSTM (proposed)";

          return (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
                  n = {n}
                </Badge>
                {errorCount > 0 && (
                  <Badge variant="outline" className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    {errorCount} row(s) excluded (errors)
                  </Badge>
                )}
                <span className="text-[11px] text-muted-foreground">
                  scored {doneCount} row(s)
                </span>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <MetricColumn
                  label="Baseline"
                  archLabel="LSTM"
                  metrics={baseline}
                  variant="baseline"
                  bestMae={baselineBetterMae}
                  bestR={!proposedBetterR}
                />
                <MetricColumn
                  label="New Model"
                  archLabel="BiLSTM"
                  metrics={proposed}
                  variant="proposed"
                  bestMae={!baselineBetterMae}
                  bestR={proposedBetterR}
                />
              </div>

              <Separator />

              <div className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
                    <Sigma className="size-3.5" /> Mann-Whitney U · absolute errors
                  </div>
                  <Badge
                    className={cn(
                      "text-[10px] uppercase tracking-wider",
                      mwu.significant
                        ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                        : "bg-muted text-muted-foreground",
                    )}
                  >
                    {mwu.significant ? "Significant" : "Not significant"}
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                      U statistic
                    </div>
                    <div className="font-mono text-sm tabular-nums text-foreground">
                      {fmt(mwu.u_statistic, 1)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                      p-value
                    </div>
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                      <span className="font-mono text-sm tabular-nums text-foreground">
                        {fmtP(mwu.p_value)}
                      </span>
                      {mwu.significant && (
                        <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
                          {betterLabel} performed better
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </>
          );
        })()}
      </CardContent>
    </Card>
  );
}
