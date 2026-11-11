import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api.js';
import { deriveRun, TERMINAL_EVENTS } from './runState.js';

// Starts a design workflow run and follows its events over SSE.
// `start` resolves with the workflow result or rejects with its error.
export function useDesignRun() {
  const [runId, setRunId] = useState(null);
  const [events, setEvents] = useState([]);
  const [connectionError, setConnectionError] = useState('');
  const [cancelling, setCancelling] = useState(false);
  // Server time -> browser time, so live timers don't depend on both clocks matching.
  const [clockOffset, setClockOffset] = useState(0);
  const source = useRef(null);
  const lastSeq = useRef(0);

  const close = useCallback(() => {
    source.current?.close();
    source.current = null;
  }, []);

  useEffect(() => close, [close]);

  const start = useCallback(async (payload) => {
    close();
    setEvents([]);
    setConnectionError('');
    setCancelling(false);
    lastSeq.current = 0;
    const { runId: id } = await api.startDesignRun(payload);
    setRunId(id);

    return new Promise((resolve, reject) => {
      let offset = null;
      // Downloads are on the server once asset_download completed, even if the run ends without a result.
      let downloaded = false;
      const fail = (message) => {
        const error = new Error(message);
        error.downloaded = downloaded;
        reject(error);
      };
      // EventSource reconnects on its own and sends Last-Event-ID, so the backend only
      // replays missed events; the run itself is never started twice.
      const es = new EventSource(api.designRunEventsUrl(id));
      source.current = es;

      es.onmessage = (message) => {
        const event = JSON.parse(message.data);
        if (event.seq <= lastSeq.current) return;
        lastSeq.current = event.seq;
        const measured = Date.now() - event.ts;
        if (offset === null || measured < offset) {
          offset = measured;
          setClockOffset(measured);
        }
        setConnectionError('');
        setEvents((list) => [...list, event]);
        if (event.type === 'node.completed' && event.node === 'asset_download'
          && event.payload?.output?.download_result?.added?.length) downloaded = true;

        if (TERMINAL_EVENTS.has(event.type)) {
          es.close();
          setCancelling(false);
          if (event.type === 'workflow.completed') resolve(event.payload.result);
          else if (event.type === 'workflow.cancelled') fail('The run was aborted. The room was not changed.');
          else fail(event.payload.error || 'The workflow failed.');
        }
      };

      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED) {
          setCancelling(false);
          fail('The connection to the workflow was lost.');
        } else {
          setConnectionError('Connection interrupted – reconnecting…');
        }
      };
    });
  }, [close]);

  // Aborts the run for good; the terminal event arrives on the open stream and ends `start`.
  const cancel = useCallback(async () => {
    if (!runId || cancelling) return;
    setCancelling(true);
    try {
      await api.cancelDesignRun(runId);
    } catch {
      setCancelling(false);
    }
  }, [runId, cancelling]);

  // Answers a pending approval; the same run continues and its events keep arriving on the open stream.
  const respond = useCallback((decision) => api.answerApproval(runId, decision), [runId]);

  const run = useMemo(() => deriveRun(events), [events]);
  // The run can be aborted while it works or waits for the user.
  const active = run.status === 'running' || run.status === 'waiting';
  return { runId, events, run, clockOffset, connectionError, start, cancel, cancelling, active, respond };
}
