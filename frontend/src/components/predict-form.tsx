import { useMemo } from "react";
import { Sparkles, Loader2, WandSparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
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
import type { Dimension, ModelInfo } from "@/types/api";

export interface PredictFormState {
  text: string;
  dataset: string;
  dimension: Dimension;
}

interface PredictFormProps {
  models: ModelInfo[];
  modelsLoading: boolean;
  value: PredictFormState;
  onChange: (next: PredictFormState) => void;
  onSubmit: () => void;
  submitting: boolean;
}

const MAX_CHARS = 10_000;

function uniq<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}

const DATASET_EXAMPLES: Record<string, string> = {
  emobank_writer: "The team worked overtime for weeks, and when the launch finally succeeded everyone cried with joy.",
  emobank_reader: "Reports of the quiet celebration spread through the newsroom.",
  fb: "Had the best time with friends tonight!! 🎉",
  sst: "A dull, overlong, and poorly acted film.",
  cvat: "這部電影真的太感人了，看完後心情久久不能平靜。",
};

export function PredictForm({
  models,
  modelsLoading,
  value,
  onChange,
  onSubmit,
  submitting,
}: PredictFormProps) {
  const datasets = useMemo(() => {
    const seen = new Map<string, ModelInfo>();
    for (const m of models) if (!seen.has(m.dataset)) seen.set(m.dataset, m);
    return Array.from(seen.values());
  }, [models]);

  const dimensionsForDataset = useMemo<Dimension[]>(() => {
    return uniq(models.filter((m) => m.dataset === value.dataset).map((m) => m.dimension));
  }, [models, value.dataset]);

  const charCount = value.text.length;
  const tooLong = charCount > MAX_CHARS;
  const empty = value.text.trim().length === 0;
  const disabled = submitting || empty || tooLong || modelsLoading || datasets.length === 0;

  const handleDatasetChange = (next: string) => {
    const dims = uniq(models.filter((m) => m.dataset === next).map((m) => m.dimension));
    const dimension: Dimension = dims.includes(value.dimension)
      ? value.dimension
      : dims[0] ?? "valence";
    onChange({ ...value, dataset: next, dimension });
  };

  const handleDimensionChange = (next: Dimension) => {
    onChange({ ...value, dimension: next });
  };

  const activeInfo = models.find(
    (m) => m.dataset === value.dataset && m.dimension === value.dimension,
  );

  const exampleText = DATASET_EXAMPLES[value.dataset] ?? DATASET_EXAMPLES.emobank_writer;

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-xl">Input</CardTitle>
        <CardDescription>
          Pick a dataset and dimension, paste text, and run both architectures side by side.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Text */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="text">Text</Label>
            <button
              type="button"
              onClick={() => onChange({ ...value, text: exampleText })}
              className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent/40 hover:text-foreground"
            >
              <WandSparkles className="size-3" />
              Try a sample
            </button>
          </div>
          <Textarea
            id="text"
            rows={8}
            placeholder={exampleText}
            value={value.text}
            onChange={(e) => onChange({ ...value, text: e.target.value })}
            className="resize-none font-serif text-base leading-relaxed"
          />
          <div className="flex items-center justify-end text-[11px] text-muted-foreground tabular-nums">
            <span className={tooLong ? "text-destructive font-medium" : undefined}>
              {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          </div>
        </div>

        {/* Setup */}
        <div className="space-y-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Setup
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={value.dataset} onValueChange={handleDatasetChange} disabled={modelsLoading}>
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
              <Label>Dimension</Label>
              <Select
                value={value.dimension}
                onValueChange={(v) => handleDimensionChange(v as Dimension)}
                disabled={modelsLoading || dimensionsForDataset.length === 0}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {dimensionsForDataset.map((d) => (
                    <SelectItem key={d} value={d}>
                      {d === "valence" ? "Valence (pleasantness)" : "Arousal (activation)"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 sm:col-span-2">
              <Label className="text-muted-foreground">Architectures (compared automatically)</Label>
              <div className="flex flex-wrap items-center gap-2 rounded-md border border-dashed border-border/70 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                <Badge variant="secondary" className="text-[10px] uppercase tracking-wider">
                  LSTM · baseline
                </Badge>
                <Badge className="text-[10px] uppercase tracking-wider">
                  BiLSTM · new model
                </Badge>
                <span className="ml-1">paper run, both run in parallel</span>
              </div>
            </div>
          </div>
        </div>

        {/* Active model summary */}
        {activeInfo && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
            <span className="text-foreground/80">{activeInfo.dataset_description}</span>
            <span className="text-border/80">•</span>
            <span>scale</span>
            <span className="font-mono text-foreground/80 tabular-nums">
              {activeInfo.scale_min}&ndash;{activeInfo.scale_max}
            </span>
            <span className="text-border/80">•</span>
            <span>region depth</span>
            <span className="font-mono text-foreground/80 tabular-nums">{activeInfo.region_depth}</span>
          </div>
        )}

        <Button type="button" onClick={onSubmit} disabled={disabled} size="lg" className="w-full gap-2">
          {submitting ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Running both models…
            </>
          ) : (
            <>
              <Sparkles className="size-4" /> Compare baseline vs proposed
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
