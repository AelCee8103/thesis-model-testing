import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Languages, Layers3, Network } from "lucide-react";

import { Header } from "@/components/header";
import { PredictForm } from "@/components/predict-form";
import type { PredictFormState } from "@/components/predict-form";
import { ResultPanel } from "@/components/result-panel";
import { BatchUploadForm } from "@/components/batch-upload-form";
import { BatchResultsTable } from "@/components/batch-results-table";
import { EvaluationSummary } from "@/components/evaluation-summary";
import { ModelsCatalog } from "@/components/models-catalog";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";

import { useModels } from "@/hooks/useModels";
import { useCompare } from "@/hooks/useCompare";
import { useBatchCompare } from "@/hooks/useBatchCompare";
import { decodeBytes, parseEvalRows } from "@/lib/csv";
import type { EvalParse, EvalRow } from "@/lib/csv";
import type { Dimension, ModelInfo } from "@/types/api";

type Mode = "single" | "evaluate";

const MAX_EVAL_ROWS = 2000;
const MAX_CHARS = 10_000;

interface CsvState {
  fileName: string | null;
  rows: EvalRow[];
  skipped: number;
  invalidCount: number;
  dimension: Dimension | null;
  parseError: string | null;
  encoding: string | null;
}

const RUN_TAG = "paper";

function pickInitialState(models: ModelInfo[]): PredictFormState {
  if (!models.length) {
    return { text: "", dataset: "emobank_writer", dimension: "valence" };
  }
  const preferred = models.find(
    (m) => m.dataset === "emobank_writer" && m.dimension === "valence",
  );
  const chosen = preferred ?? models[0]!;
  return {
    text: "",
    dataset: chosen.dataset,
    dimension: chosen.dimension as Dimension,
  };
}

function App() {
  const { models, loading: modelsLoading, error: modelsError } = useModels();
  const { result, loading: predicting, error: predictError, run } = useCompare();
  const batch = useBatchCompare();

  const [mode, setMode] = useState<Mode>("single");
  const [form, setForm] = useState<PredictFormState>(() => pickInitialState([]));
  const [formSeeded, setFormSeeded] = useState(false);
  const [batchConfig, setBatchConfig] = useState<{ dataset: string; dimension: Dimension }>(() => {
    const init = pickInitialState([]);
    return { dataset: init.dataset, dimension: init.dimension };
  });
  const [csv, setCsv] = useState<CsvState>({
    fileName: null,
    rows: [],
    skipped: 0,
    invalidCount: 0,
    dimension: null,
    parseError: null,
    encoding: null,
  });

  useEffect(() => {
    if (!formSeeded && models.length > 0) {
      const init = pickInitialState(models);
      setForm(init);
      setBatchConfig({ dataset: init.dataset, dimension: init.dimension });
      setFormSeeded(true);
    }
  }, [models, formSeeded]);

  useEffect(() => {
    if (modelsError) {
      toast.error("Could not load models", { description: modelsError });
    }
  }, [modelsError]);

  useEffect(() => {
    if (predictError) {
      toast.error("Comparison failed", { description: predictError });
    }
  }, [predictError]);

  useEffect(() => {
    if (batch.error) {
      toast.error("Run failed", { description: batch.error });
    }
  }, [batch.error]);

  useEffect(() => {
    if (batch.evalError) {
      toast.error("Could not compute metrics", { description: batch.evalError });
    }
  }, [batch.evalError]);

  const modelsSorted = useMemo(() => {
    return [...models].sort((a, b) => {
      return (
        a.dataset.localeCompare(b.dataset) ||
        a.dimension.localeCompare(b.dimension) ||
        a.architecture.localeCompare(b.architecture)
      );
    });
  }, [models]);

  const summary = useMemo(() => {
    const datasets = new Set(models.map((m) => m.dataset)).size;
    const languages = new Set(models.map((m) => m.language)).size;
    const architectures = new Set(models.map((m) => m.architecture)).size;
    return { datasets, languages, architectures };
  }, [models]);

  const onSubmit = () => {
    run({
      text: form.text,
      dataset: form.dataset,
      dimension: form.dimension,
      run_tag: RUN_TAG,
    });
  };

  const onFile = async (file: File) => {
    let encoding: string;
    let parse: EvalParse;
    try {
      const decoded = decodeBytes(await file.arrayBuffer());
      encoding = decoded.encoding;
      parse = parseEvalRows(decoded.text);
    } catch {
      setCsv({
        fileName: file.name,
        rows: [],
        skipped: 0,
        invalidCount: 0,
        dimension: null,
        parseError: "Could not parse CSV",
        encoding: null,
      });
      return;
    }

    if (parse.error) {
      setCsv({
        fileName: file.name,
        rows: [],
        skipped: parse.skipped,
        invalidCount: 0,
        dimension: null,
        parseError: parse.error,
        encoding,
      });
      return;
    }

    const valid = parse.rows.filter((r) => r.text.length <= MAX_CHARS);
    const invalidCount = parse.rows.length - valid.length;

    let capped = valid;
    if (valid.length > MAX_EVAL_ROWS) {
      capped = valid.slice(0, MAX_EVAL_ROWS);
      toast.warning(`Capped to ${MAX_EVAL_ROWS} rows`, {
        description: `File had ${valid.length} rows; only the first ${MAX_EVAL_ROWS} will be scored.`,
      });
    }

    if (capped.length === 0) {
      toast.error("No usable rows", {
        description: "Need rows with non-empty text and a numeric truth value.",
      });
    }

    setCsv({
      fileName: file.name,
      rows: capped,
      skipped: parse.skipped,
      invalidCount,
      dimension: parse.dimension,
      parseError: null,
      encoding,
    });

    if (parse.dimension) {
      const detected = parse.dimension;
      setBatchConfig((prev) => ({ ...prev, dimension: detected }));
    }
  };

  const onDatasetChange = (dataset: string) =>
    setBatchConfig((prev) => ({ ...prev, dataset }));

  const onRunBatch = () => batch.run(csv.rows, batchConfig);

  const { datasetError, scaleWarning } = useMemo(() => {
    if (!csv.dimension || csv.rows.length === 0) {
      return { datasetError: null as string | null, scaleWarning: null as string | null };
    }
    const info = models.find(
      (m) => m.dataset === batchConfig.dataset && m.dimension === csv.dimension,
    );
    if (!info) {
      return {
        datasetError: `Dataset “${batchConfig.dataset}” has no ${csv.dimension} model. Choose a dataset that supports ${csv.dimension}.`,
        scaleWarning: null as string | null,
      };
    }
    const outOfRange = csv.rows.filter(
      (r) => r.truth < info.scale_min || r.truth > info.scale_max,
    ).length;
    return {
      datasetError: null as string | null,
      scaleWarning:
        outOfRange > 0
          ? `${outOfRange} truth value(s) fall outside this dataset's scale [${info.scale_min}, ${info.scale_max}]. MAE is only meaningful when truth and predictions share a scale.`
          : null,
    };
  }, [models, batchConfig.dataset, csv.dimension, csv.rows]);

  const doneCount = batch.rows.filter((r) => r.status === "done").length;
  const errorCount = batch.rows.filter((r) => r.status === "error").length;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main className="mx-auto max-w-6xl space-y-8 px-6 py-10">
        {/* Hero / context */}
        <section className="space-y-4">
          <div className="space-y-2">
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Dimensional sentiment analysis
            </p>
            <h2 className="font-serif text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
              Score any sentence on a continuous valence &amp; arousal scale.
            </h2>
            <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground sm:text-base">
              Pick a dataset and architecture, type or paste text, and get the model's prediction on
              the dataset's native rating scale &mdash; with the parsed regions the model actually
              attended to.
            </p>
          </div>

          {!modelsLoading && models.length > 0 && (
            <dl className="flex flex-wrap gap-2 pt-1">
              <SummaryPill icon={Layers3} label="Datasets" value={summary.datasets} />
              <SummaryPill icon={Languages} label="Languages" value={summary.languages} />
              <SummaryPill icon={Network} label="Architectures" value={summary.architectures} />
            </dl>
          )}
        </section>

        {/* Mode toggle */}
        <div className="inline-flex rounded-lg border border-border/60 bg-muted/30 p-1">
          <Button
            type="button"
            size="sm"
            variant={mode === "single" ? "default" : "ghost"}
            onClick={() => setMode("single")}
          >
            Single
          </Button>
          <Button
            type="button"
            size="sm"
            variant={mode === "evaluate" ? "default" : "ghost"}
            onClick={() => setMode("evaluate")}
          >
            Evaluate CSV
          </Button>
        </div>

        {/* Workspace */}
        <div className="space-y-6">
          {mode === "single" ? (
            <>
              <PredictForm
                models={modelsSorted}
                modelsLoading={modelsLoading}
                value={form}
                onChange={setForm}
                onSubmit={onSubmit}
                submitting={predicting}
              />
              <ResultPanel result={result} loading={predicting} error={predictError} />
            </>
          ) : (
            <>
              <BatchUploadForm
                models={modelsSorted}
                modelsLoading={modelsLoading}
                dataset={batchConfig.dataset}
                onDatasetChange={onDatasetChange}
                dimension={csv.dimension}
                fileName={csv.fileName}
                rowCount={csv.rows.length}
                skipped={csv.skipped}
                invalidCount={csv.invalidCount}
                encoding={csv.encoding}
                parseError={csv.parseError}
                datasetError={datasetError}
                scaleWarning={scaleWarning}
                onFile={onFile}
                onRun={onRunBatch}
                onCancel={batch.cancel}
                running={batch.running}
                progress={batch.progress}
              />
              <EvaluationSummary
                metrics={batch.metrics}
                evaluating={batch.evaluating}
                doneCount={doneCount}
                errorCount={errorCount}
              />
              <BatchResultsTable
                rows={batch.rows}
                dataset={batchConfig.dataset}
                dimension={batchConfig.dimension}
              />
            </>
          )}
        </div>

        <ModelsCatalog models={modelsSorted} />

        <footer className="border-t border-border/60 pt-6 text-center text-xs text-muted-foreground">
          Research prototype &middot; Regional CNN-BiLSTM &middot; not for clinical or diagnostic use
        </footer>
      </main>
      <Toaster position="bottom-right" />
    </div>
  );
}

function SummaryPill({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/60 px-3 py-1 text-xs">
      <Icon className="size-3.5 text-muted-foreground" />
      <span className="font-medium tabular-nums text-foreground">{value}</span>
      <span className="text-muted-foreground">{label.toLowerCase()}</span>
    </div>
  );
}

export default App;
