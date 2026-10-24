import { cn } from "@/lib/utils";

interface ScoreGaugeProps {
  score: number;
  min: number;
  max: number;
  dimension: "valence" | "arousal";
}

export function ScoreGauge({ score, min, max, dimension }: ScoreGaugeProps) {
  const range = Math.max(1e-9, max - min);
  const clamped = Math.max(min, Math.min(max, score));
  const pct = ((clamped - min) / range) * 100;
  // Clamp the callout position so the bubble doesn't slip past the gauge edges.
  const calloutPct = Math.max(4, Math.min(96, pct));

  const fillVar = dimension === "valence" ? "var(--chart-2)" : "var(--chart-1)";

  // Tick positions for neat axis (5 ticks)
  const ticks = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);

  return (
    <div className="space-y-2">
      {/* Floating callout */}
      <div className="relative h-5">
        <div
          className="absolute -translate-x-1/2 transition-[left] duration-500 ease-out"
          style={{ left: `${calloutPct}%` }}
        >
          <span className="inline-block rounded-md border border-border/60 bg-card px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-foreground shadow-xs">
            {clamped.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Bar */}
      <div className="relative h-5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%`, background: fillVar }}
          aria-hidden
        />
        <div
          className="absolute top-0 h-full w-[2px] bg-foreground/80 transition-[left] duration-500 ease-out"
          style={{ left: `calc(${pct}% - 1px)` }}
          aria-hidden
        />
      </div>

      {/* Ticks */}
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        {ticks.map((t, i) => (
          <span key={i} className={cn(i === 0 || i === ticks.length - 1 ? "" : "opacity-60")}>
            {t.toFixed(t >= 1 ? 1 : 2)}
          </span>
        ))}
      </div>
    </div>
  );
}
