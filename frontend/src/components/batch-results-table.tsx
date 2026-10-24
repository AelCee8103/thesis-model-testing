import { Download, Loader2, Table2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { toCsv } from "@/lib/csv";
import type { Dimension } from "@/types/api";
import type { BatchRowResult } from "@/hooks/useBatchCompare";

interface BatchResultsTableProps {
  rows: BatchRowResult[];
  dataset: string;
  dimension: Dimension;
}

function StatusBadge({ row }: { row: BatchRowResult }) {
  switch (row.status) {
    case "done":
      return (
        <Badge variant="secondary" className="text-[10px] uppercase tracking-wider">
          done
        </Badge>
      );
    case "error":
      return (
        <Badge variant="destructive" className="text-[10px] uppercase tracking-wider">
          error
        </Badge>
      );
    case "running":
      return (
        <Badge variant="outline" className="gap-1 text-[10px] uppercase tracking-wider">
          <Loader2 className="size-3 animate-spin" /> running
        </Badge>
      );
    default:
      return (
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider text-muted-foreground">
          pending
        </Badge>
      );
  }
}

function num(v: number | undefined, digits = 2): string {
  return v === undefined ? "—" : v.toFixed(digits);
}

export function BatchResultsTable({ rows, dataset, dimension }: BatchResultsTableProps) {
  const hasDone = rows.some((r) => r.status === "done");

  const download = () => {
    const header = [
      "row",
      "text",
      "truth",
      "baseline_score",
      "baseline_abs_err",
      "proposed_score",
      "proposed_abs_err",
      "delta",
      "status",
      "warnings",
    ];
    const body = rows.map((r) => [
      r.index + 1,
      r.text,
      r.truth,
      r.baseline?.score ?? "",
      r.baselineAbsErr ?? "",
      r.proposed?.score ?? "",
      r.proposedAbsErr ?? "",
      r.delta ?? "",
      r.status === "error" && r.error ? `error: ${r.error}` : r.status,
      [...(r.baseline?.warnings ?? []), ...(r.proposed?.warnings ?? [])].join(" | "),
    ]);
    const csv = "﻿" + toCsv([header, ...body]); // BOM for Excel
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `eval_${dataset}_${dimension}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader className="space-y-1">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-xl">Per-row results</CardTitle>
            <CardDescription>
              Truth, baseline LSTM vs New Model BiLSTM, and absolute error for each row.
            </CardDescription>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={download}
            disabled={!hasDone}
            className="gap-2"
          >
            <Download className="size-4" /> Download results CSV
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <div className="flex h-48 flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/30 text-center">
            <Table2 className="mb-3 size-6 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">No results yet</p>
            <p className="mt-1 max-w-[30ch] text-xs text-muted-foreground">
              Upload a labelled CSV and run the evaluation to score every row.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border/60">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-muted/40 text-left text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Text</th>
                  <th className="px-3 py-2 text-right font-medium">Truth</th>
                  <th className="px-3 py-2 text-right font-medium">Baseline</th>
                  <th className="px-3 py-2 text-right font-medium">|err|</th>
                  <th className="px-3 py-2 text-right font-medium">New Model</th>
                  <th className="px-3 py-2 text-right font-medium">|err|</th>
                  <th className="px-3 py-2 text-right font-medium">Δ</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const hasDelta = r.delta !== undefined && Math.abs(r.delta) > 1e-4;
                  return (
                    <tr key={r.index} className="border-t border-border/40 align-top">
                      <td className="px-3 py-2 text-muted-foreground tabular-nums">{r.index + 1}</td>
                      <td className="px-3 py-2">
                        <span className="line-clamp-2 max-w-[36ch] text-foreground/90">{r.text}</span>
                        {r.status === "error" && r.error && (
                          <span className="mt-0.5 block text-[11px] text-destructive">{r.error}</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-foreground/70">
                        {num(r.truth)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-foreground/90">
                        {num(r.baseline?.score)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                        {num(r.baselineAbsErr)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-foreground/90">
                        {num(r.proposed?.score)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                        {num(r.proposedAbsErr)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {r.delta === undefined ? (
                          <span className="text-muted-foreground">—</span>
                        ) : !hasDelta ? (
                          <span className="text-muted-foreground">0.00</span>
                        ) : (
                          <span
                            className={cn(
                              r.delta > 0
                                ? "text-emerald-600 dark:text-emerald-400"
                                : "text-rose-600 dark:text-rose-400",
                            )}
                          >
                            {r.delta > 0 ? "+" : ""}
                            {r.delta.toFixed(2)}
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2">
                        <StatusBadge row={r} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
