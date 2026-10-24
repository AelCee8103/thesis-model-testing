import { Activity } from "lucide-react";
import { HealthDot } from "@/components/health-dot";
import { ThemeToggle } from "@/components/theme-toggle";

export function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-border/60 bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <div className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm ring-1 ring-primary/20">
            <Activity className="size-5" />
          </span>
          <div className="leading-tight">
            <h1 className="font-serif text-base font-semibold text-foreground sm:text-lg">
              Regional CNN-BiLSTM
            </h1>
            <p className="text-[11px] text-muted-foreground sm:text-xs">
              Dimensional sentiment &middot; valence &amp; arousal
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
