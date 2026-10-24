import { useState } from "react";
import { ChevronDown, Database } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ModelInfo } from "@/types/api";

export function ModelsCatalog({ models }: { models: ModelInfo[] }) {
  const [open, setOpen] = useState(false);

  if (!models.length) return null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 py-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="size-4 text-muted-foreground" />
          Available models
          <Badge variant="secondary" className="ml-1">
            {models.length}
          </Badge>
        </CardTitle>
        <Button
          size="sm"
          variant="ghost"
          className="gap-1 text-xs"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Hide" : "Show"}
          <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} />
        </Button>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="overflow-hidden rounded-md border border-border/60">
            <table className="w-full text-sm">
              <thead className="bg-muted/60 text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Dataset</th>
                  <th className="px-3 py-2 text-left font-medium">Dim</th>
                  <th className="px-3 py-2 text-left font-medium">Arch</th>
                  <th className="px-3 py-2 text-left font-medium">Tag</th>
                  <th className="px-3 py-2 text-left font-medium">Scale</th>
                  <th className="px-3 py-2 text-left font-medium">Depth</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i} className="border-t border-border/40 font-mono text-xs">
                    <td className="px-3 py-2 text-foreground">{m.dataset}</td>
                    <td className="px-3 py-2">{m.dimension}</td>
                    <td className="px-3 py-2">{m.architecture}</td>
                    <td className="px-3 py-2">{m.run_tag}</td>
                    <td className="px-3 py-2">
                      {m.scale_min} – {m.scale_max}
                    </td>
                    <td className="px-3 py-2">{m.region_depth}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
