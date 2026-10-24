import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface RegionsViewProps {
  regions: string[][];
  depth: number;
  capacity: number;
}

const COLLAPSE_THRESHOLD = 6;

export function RegionsView({ regions, depth, capacity }: RegionsViewProps) {
  const [open, setOpen] = useState(false);

  if (!regions.length) return null;

  const collapsible = regions.length > COLLAPSE_THRESHOLD;
  const visible = collapsible && !open ? regions.slice(0, COLLAPSE_THRESHOLD) : regions;
  const hidden = regions.length - visible.length;

  return (
    <div className="space-y-3 rounded-lg border border-border/60 bg-muted/30 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-0.5">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Parsed regions &middot; depth {depth}
          </div>
          <p className="max-w-[60ch] text-xs text-muted-foreground">
            Each card is one chunk of the constituency parse at the dataset's optimal depth.
            The model accepts up to {capacity} regions; the remainder is zero-padded.
          </p>
        </div>
        <Badge variant="secondary" className="font-mono tabular-nums">
          {regions.length} / {capacity}
        </Badge>
      </div>

      <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((tokens, i) => (
          <li
            key={i}
            className="flex items-start gap-2 rounded-md border border-border/60 bg-card px-3 py-2 shadow-xs"
          >
            <span className="mt-0.5 font-mono text-[10px] font-medium text-muted-foreground tabular-nums">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div className="flex flex-wrap gap-1">
              {tokens.map((tok, j) => (
                <span
                  key={j}
                  className="rounded-sm bg-accent px-1.5 py-0.5 font-mono text-[11px] text-accent-foreground"
                >
                  {tok}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ol>

      {collapsible && (
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="w-full gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Show fewer" : `Show ${hidden} more region${hidden === 1 ? "" : "s"}`}
          <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
        </Button>
      )}
    </div>
  );
}
