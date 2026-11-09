import { useCallback, useRef, useState } from 'react';
import { api } from '../api/client';
import type { EvaluateResult, Region } from '../content/types';
import { cueIncorrect, cueListen } from './sounds';
import { playClip, stopClip } from './useHoverAudio';
import { useRecorder, type RecordingHandle } from './useRecorder';

export type RecitePhase =
  | 'idle'
  | 'playing' // reference audio playing
  | 'listening' // mic open, waiting for the learner
  | 'evaluating' // (sync mode) waiting for the verdict
  | 'correct'
  | 'incorrect' // brief flash before the reference replays
  | 'moveon' // 3 failed attempts — continue to the next item
  | 'finishing' // practice flow done, background checks still draining
  | 'done' // practice run finished
  | 'error';

/** Verdict of a background (pipelined) check, keyed by region id. */
export interface AsyncVerdict {
  status: 'pending' | 'correct' | 'failed' | 'error';
  result?: EvaluateResult;
}

export interface ReciteState {
  phase: RecitePhase;
  region: Region | null;
  attempt: number;
  result: EvaluateResult | null;
  error: string | null;
  /** Practice-mode progress; null for a single-item recite. */
  queueIndex: number | null;
  queueTotal: number | null;
  /** Background-check verdicts for the current practice run. */
  verdicts: Record<string, AsyncVerdict>;
  /** True while re-doing the words that failed the flow round. */
  reviewing: boolean;
}

type ItemOutcome = 'correct' | 'failed' | 'skipped' | 'cancelled' | 'error';

const MAX_ATTEMPTS = 3;
const LISTEN_MS = 7000;

const IDLE: ReciteState = {
  phase: 'idle',
  region: null,
  attempt: 0,
  result: null,
  error: null,
  queueIndex: null,
  queueTotal: null,
  verdicts: {},
  reviewing: false,
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * The learning loop. Practice mode is *pipelined*: while the backend checks
 * word N (~1.5s), the learner is already listening to / reciting word N+1,
 * so checking time never blocks the flow. Words that fail come back for a
 * synchronous review round at the end.
 */
export function useReciteMachine() {
  const [state, setState] = useState<ReciteState>(IDLE);
  const recorder = useRecorder();
  const runId = useRef(0);
  const skipId = useRef(0);
  const activeRecording = useRef<RecordingHandle | null>(null);
  const playReferenceRef = useRef(true);

  const cancel = useCallback(() => {
    runId.current += 1;
    stopClip();
    activeRecording.current?.stop();
    setState(IDLE);
  }, []);

  /** Stops the mic early (silence detection usually does this automatically). */
  const stopListening = useCallback(() => {
    activeRecording.current?.stop();
  }, []);

  /** Abandon the current item; a practice run continues with the next one. */
  const skip = useCallback(() => {
    skipId.current += 1;
    stopClip();
    activeRecording.current?.stop();
  }, []);

  const setPlayReference = useCallback((v: boolean) => {
    playReferenceRef.current = v;
  }, []);

  /** One item, fully synchronous (single click + review round): teach on mistakes. */
  const runOne = useCallback(
    async (region: Region, rid: number, patchBase: Partial<ReciteState>): Promise<ItemOutcome> => {
      const sid = skipId.current;
      const aliveRun = () => runId.current === rid;
      const aliveItem = () => aliveRun() && skipId.current === sid;
      const outcomeIfDead = (): ItemOutcome => (aliveRun() ? 'skipped' : 'cancelled');
      const update = (patch: Partial<ReciteState>) => {
        if (aliveRun()) setState((s) => ({ ...s, ...patchBase, region, ...patch }));
      };

      try {
        for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
          // On later attempts the reference always replays — that's the teaching step.
          if (playReferenceRef.current || attempt > 1) {
            update({ phase: 'playing', attempt });
            await playClip(api.audioUrl(region.audio));
            if (!aliveItem()) return outcomeIfDead();
          }

          update({ phase: 'listening', attempt });
          cueListen();
          const recording = await recorder.start(LISTEN_MS);
          activeRecording.current = recording;
          const blob = await recording.blob;
          activeRecording.current = null;
          if (!aliveItem()) return outcomeIfDead();

          update({ phase: 'evaluating' });
          const result = await api.evaluate(region.id, blob);
          if (!aliveItem()) return outcomeIfDead();

          if (result.verdict === 'correct') {
            update({ phase: 'correct', result });
            await sleep(600);
            return aliveItem() ? 'correct' : outcomeIfDead();
          }

          update({ phase: 'incorrect', result });
          cueIncorrect();
          await sleep(800);
          if (!aliveItem()) return outcomeIfDead();
        }
        update({ phase: 'moveon' });
        await sleep(1500);
        return aliveItem() ? 'failed' : outcomeIfDead();
      } catch (err) {
        update({ phase: 'error', error: err instanceof Error ? err.message : String(err) });
        await sleep(2500);
        return 'error';
      }
    },
    [recorder],
  );

  const startRecite = useCallback(
    async (region: Region) => {
      if (!region.expected) return;
      runId.current += 1;
      const rid = runId.current;
      stopClip();
      const outcome = await runOne(region, rid, { queueIndex: null, queueTotal: null });
      if (outcome !== 'cancelled' && runId.current === rid) setState(IDLE);
    },
    [runOne],
  );

  const startPractice = useCallback(
    async (regions: Region[], pipelined = false) => {
      const queue = regions.filter((r) => r.expected);
      if (queue.length === 0) return;
      runId.current += 1;
      const rid = runId.current;
      const aliveRun = () => runId.current === rid;
      stopClip();
      setState({ ...IDLE, queueTotal: queue.length, queueIndex: 0 });

      // Synchronous mode (default): verdict before moving on, teaching inline.
      if (!pipelined) {
        for (let i = 0; i < queue.length; i++) {
          const outcome = await runOne(queue[i], rid, { queueIndex: i, queueTotal: queue.length });
          if (outcome === 'cancelled' || !aliveRun()) return;
          if (outcome === 'error') break;
        }
        if (aliveRun()) {
          setState((s) => ({ ...s, phase: 'done', region: null }));
          await sleep(2500);
          if (aliveRun()) setState(IDLE);
        }
        return;
      }

      const setVerdict = (id: string, v: AsyncVerdict) => {
        if (aliveRun()) setState((s) => ({ ...s, verdicts: { ...s.verdicts, [id]: v } }));
      };
      const checks: Promise<void>[] = [];

      // Flow round: record each word, fire the check, move on immediately.
      for (let i = 0; i < queue.length; i++) {
        const region = queue[i];
        const sid = skipId.current;
        const aliveItem = () => aliveRun() && skipId.current === sid;
        const update = (patch: Partial<ReciteState>) => {
          if (aliveRun()) setState((s) => ({ ...s, queueIndex: i, region, ...patch }));
        };

        try {
          if (playReferenceRef.current) {
            update({ phase: 'playing' });
            await playClip(api.audioUrl(region.audio));
            if (!aliveRun()) return;
            if (!aliveItem()) continue;
          }

          update({ phase: 'listening' });
          cueListen();
          const recording = await recorder.start(LISTEN_MS);
          activeRecording.current = recording;
          const blob = await recording.blob;
          activeRecording.current = null;
          if (!aliveRun()) return;
          if (!aliveItem()) continue;

          setVerdict(region.id, { status: 'pending' });
          checks.push(
            api
              .evaluate(region.id, blob)
              .then((result) => {
                if (!aliveRun()) return;
                if (result.verdict === 'correct') {
                  setVerdict(region.id, { status: 'correct', result });
                } else {
                  setVerdict(region.id, { status: 'failed', result });
                  cueIncorrect(); // the "that one needs another look" nudge
                }
              })
              .catch(() => setVerdict(region.id, { status: 'error' })),
          );
        } catch (err) {
          if (!aliveRun()) return;
          setVerdict(region.id, { status: 'error' });
          setState((s) => ({ ...s, error: err instanceof Error ? err.message : String(err) }));
        }
      }

      // Drain outstanding checks.
      if (!aliveRun()) return;
      setState((s) => ({ ...s, phase: 'finishing', region: null }));
      await Promise.allSettled(checks);
      if (!aliveRun()) return;

      // Review round: synchronously re-teach everything that didn't pass.
      let failed: Region[] = [];
      setState((s) => {
        failed = queue.filter((r) => s.verdicts[r.id]?.status === 'failed');
        return s;
      });
      await sleep(0); // let the state read above settle
      if (failed.length > 0) {
        for (let i = 0; i < failed.length; i++) {
          if (!aliveRun()) return;
          const outcome = await runOne(failed[i], rid, {
            queueIndex: i,
            queueTotal: failed.length,
            reviewing: true,
          });
          if (outcome === 'cancelled') return;
          if (outcome === 'correct') setVerdict(failed[i].id, { status: 'correct' });
        }
      }

      if (aliveRun()) {
        setState((s) => ({ ...s, phase: 'done', region: null, reviewing: false }));
        await sleep(2500);
        if (aliveRun()) setState(IDLE);
      }
    },
    [recorder, runOne],
  );

  return { state, startRecite, startPractice, stopListening, skip, cancel, setPlayReference };
}
