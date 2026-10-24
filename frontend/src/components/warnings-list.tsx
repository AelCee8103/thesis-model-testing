import { AlertTriangle, Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

function severity(text: string): "info" | "warn" {
  const t = text.toLowerCase();
  if (t.includes("out-of-vocabulary") || t.includes("clamped") || t.includes("truncated")) return "warn";
  return "info";
}

export function WarningsList({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <div className="space-y-2">
      {warnings.map((w, i) => {
        const s = severity(w);
        const Icon = s === "warn" ? AlertTriangle : Info;
        return (
          <Alert key={i} variant={s === "warn" ? "destructive" : "default"} className="bg-card/60">
            <Icon className="size-4" />
            <AlertTitle className="capitalize">{s === "warn" ? "Warning" : "Notice"}</AlertTitle>
            <AlertDescription>{w}</AlertDescription>
          </Alert>
        );
      })}
    </div>
  );
}
