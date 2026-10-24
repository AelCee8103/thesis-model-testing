import { useMemo, useRef } from "react";
import { AlertTriangle, FileUp, Loader2, Play, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { Dimension, ModelInfo } from "@/types/api";

export interface BatchUploadFormProps {
  models: ModelInfo[];
  modelsLoading: boolean;
  dataset: string;
  onDatasetChange: (dataset: string) => void;
  dimension: Dimension | null;
  fileName: string | null;
  rowCount: number;
  skipped: number;
  invalidCount: number;
  encoding: string | null;
  parseError: string | null;
  datasetError: string | null;
  scaleWarning: string | null;
  onFile: (file: File) => void;
  onRun: () => void;
  onCancel: () => void;
  running: boolean;
  progress: { done: number; total: number };
}

export function BatchUploadForm({
  models,
  modelsLoading,
  dataset,
  onDatasetChange,
  dimension,
  fileName,
  rowCount,
  skipped,
  invalidCount,
  encoding,
  parseError,
  datasetError,
  scaleWarning,
  onFile,
  onRun,
  onCancel,
  running,
  progress,
}: BatchUploadFormProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const datasets = useMemo(() => {
    const seen = new Map<string, ModelInfo>();
    for (const m of models) if (!seen.has(m.dataset)) seen.set(m.dataset, m);
    return Array.from(seen.values());
  }, [models]);

  const pct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;
  const showProgress = running || progress.done > 0;
  const runDisabled = rowCount === 0 || modelsLoading || datasetError !== null;

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-xl">Evaluate CSV</CardTitle>
        <CardDescription>
          Upload a CSV with a <code className="font-mono text-xs">text</code> column and a{" "}
          <code className="font-mono text-xs">true_valence</code> or{" "}
          <code className="font-mono text-xs">true_arousal</code> column. The app re-predicts both
          models and scores them against the truth. Max 2,000 rows.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* File picker */}
        <div className="space-y-2">
          <Label>CSV file</Label>
          <div className="flex flex-wrap items-center gap-3">
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFile(file);
                e.target.value = "";
              }}
            />
            <Button
              type="button"
              variant="outline"
              onClick={() => inputRef.current?.click()}
              disabled={running}
              className="gap-2"
            >
              <FileUp className="size-4" />
              Choose CSV
            </Button>
            {fileName ? (
              <span className="truncate text-sm text-foreground/80">{fileName}</span>
            ) : (
              <span className="text-sm text-muted-foreground">No file selected</span>
            )}
          </div>
          {fileName && !parseError && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <Badge variant="secondary" className="text-[10px] uppercase tracking-wider">
                {rowCount} rows
              </Badge>
              {dimension && (
                <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
                  {dimension} · from true_{dimension}
                </Badge>
              )}
              {skipped > 0 && (
                <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
                  {skipped} skipped
                </Badge>
              )}
              {invalidCount > 0 && (
                <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
                  {invalidCount} too long
                </Badge>
              )}
              {encoding && encoding !== "utf-8" && (
                <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
                  decoded {encoding}
                </Badge>
              )}
            </div>
          )}
        </div>

        {parseError && (
          <Alert variant="destructive">
            <AlertTitle>Could not read CSV</AlertTitle>
            <AlertDescription className="break-words">{parseError}</AlertDescription>
          </Alert>
        )}

        {/* Setup */}
        <div className="space-y-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Setup
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={dataset} onValueChange={onDatasetChange} disabled={modelsLoading || running}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={modelsLoading ? "Loading…" : "Choose dataset"} />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Available datasets</SelectLabel>
                    {datasets.map((m) => (
                      <SelectItem key={m.dataset} value={m.dataset}>
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">{m.dataset_description}</span>
                          <Badge variant="secondary" className="shrink-0 text-[10px] uppercase">
                            {m.language}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Dimension (from file)</Label>
              <div className="flex h-9 items-center rounded-md border border-dashed border-border/70 bg-muted/30 px-3 text-sm">
                {dimension ? (
                  <span className="capitalize text-foreground">{dimension}</span>
                ) : (
                  <span className="text-muted-foreground">auto-detected on upload</span>
                )}
              </div>
            </div>

            <div className="space-y-2 sm:col-span-2">
              <Label className="text-muted-foreground">Architectures (compared automatically)</Label>
              <div className="flex flex-wrap items-center gap-2 rounded-md border border-dashed border-border/70 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                <Badge variant="secondary" className="text-[10px] uppercase tracking-wider">
                  LSTM · baseline
                </Badge>
                <Badge className="text-[10px] uppercase tracking-wider">BiLSTM · new model</Badge>
                <span className="ml-1">paper run, both per row</span>
              </div>
            </div>
          </div>
        </div>

        {datasetError && (
          <Alert variant="destructive">
            <AlertTitle>Dataset / dimension mismatch</AlertTitle>
            <AlertDescription className="break-words">{datasetError}</AlertDescription>
          </Alert>
        )}

        {scaleWarning && !datasetError && (
          <Alert>
            <AlertTriangle className="size-4" />
            <AlertTitle>Check the scale</AlertTitle>
            <AlertDescription className="break-words">{scaleWarning}</AlertDescription>
          </Alert>
        )}

        {/* Progress */}
        {showProgress && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-muted-foreground tabular-nums">
              <span>{running ? "Scoring…" : "Done"}</span>
              <span>
                {progress.done} / {progress.total}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}

        {/* Actions */}
        {running ? (
          <Button type="button" variant="destructive" onClick={onCancel} size="lg" className="w-full gap-2">
            <X className="size-4" /> Cancel
          </Button>
        ) : (
          <Button type="button" onClick={onRun} disabled={runDisabled} size="lg" className="w-full gap-2">
            {modelsLoading ? (
              <>
                <Loader2 className="size-4 animate-spin" /> Loading models…
              </>
            ) : (
              <>
                <Play className="size-4" /> Run {rowCount > 0 ? `${rowCount} rows` : "evaluation"}
              </>
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
