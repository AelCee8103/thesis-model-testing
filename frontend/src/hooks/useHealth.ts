import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";

export function useHealth(intervalMs: number = 15000): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    const tick = () => {
      fetchHealth(controller.signal).then((v) => {
        if (active) setOk(v);
      });
    };

    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      active = false;
      controller.abort();
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return ok;
}
