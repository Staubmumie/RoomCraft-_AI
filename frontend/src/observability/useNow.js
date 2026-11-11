import { useEffect, useState } from 'react';

// Current time, refreshed every `intervalMs` while `active`; used for live timers.
export function useNow(active, intervalMs = 100) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [active, intervalMs]);
  return now;
}
