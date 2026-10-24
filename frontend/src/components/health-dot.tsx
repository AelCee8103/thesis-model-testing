import { useHealth } from "@/hooks/useHealth";
import { cn } from "@/lib/utils";
import { API_BASE_URL } from "@/lib/api";

export function HealthDot() {
  const ok = useHealth();
  const color =
    ok === null ? "bg-muted-foreground/50 animate-pulse" :
    ok ? "bg-emerald-500" :
    "bg-destructive";
  const label =
    ok === null ? "Connecting…" :
    ok ? "Backend online" :
    "Backend unreachable";
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground" title={API_BASE_URL}>
      <span className={cn("inline-block size-2 rounded-full", color)} aria-hidden />
      <span className="hidden sm:inline">{label}</span>
    </div>
  );
}
