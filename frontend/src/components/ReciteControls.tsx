import type { ReciteState } from '../hooks/useReciteMachine';

interface Props {
  state: ReciteState;
  onStopListening: () => void;
  onSkip: () => void;
  onCancel: () => void;
}

const PHASE_TEXT: Record<string, string> = {
  playing: 'Listen…',
  listening: 'Your turn — recite now',
  evaluating: 'Checking…',
  correct: 'Correct! ماشاء الله',
  incorrect: 'Not quite — listen again',
  moveon: "Moving on — we'll come back to this one",
  finishing: 'Finishing the last checks…',
  done: 'Page complete! ماشاء الله',
  error: 'Something went wrong',
};

export function ReciteControls({ state, onStopListening, onSkip, onCancel }: Props) {
  if (state.phase === 'idle') return null;
  const inPractice = state.queueIndex !== null && state.queueTotal !== null;
  const pending = Object.values(state.verdicts).filter((v) => v.status === 'pending').length;
  const failed = Object.values(state.verdicts).filter((v) => v.status === 'failed').length;
  // Specific feedback from the backend replaces the generic phase text on a miss.
  const feedback = state.result?.feedback ?? null;
  const showFeedback =
    feedback !== null && (state.phase === 'incorrect' || state.phase === 'moveon' || state.phase === 'correct');
  const status = showFeedback && state.phase !== 'correct' ? feedback!.message : PHASE_TEXT[state.phase];
  return (
    <div className={`recite-bar phase-${state.phase}${showFeedback && feedback!.lowConfidence ? ' low-confidence' : ''}`}>
      {state.reviewing && <span className="recite-review-tag">review</span>}
      {inPractice && state.phase !== 'done' && state.phase !== 'finishing' && (
        <span className="recite-progress">
          {state.queueIndex! + 1}/{state.queueTotal}
        </span>
      )}
      <span className="recite-status">{status}</span>
      {showFeedback && (
        <span className="recite-feedback">
          <span dir="rtl" lang="ar">
            {feedback!.heard ? `heard «${feedback!.heard}» · expected «${feedback!.expected}»` : `expected «${feedback!.expected}»`}
            {feedback!.expectedLetter && feedback!.heardLetter && (
              <b>
                {' '}
                {feedback!.expectedLetter} ≠ {feedback!.heardLetter}
              </b>
            )}
          </span>
          {feedback!.maddHint && <span className="recite-madd">{feedback!.maddHint}</span>}
        </span>
      )}
      {state.attempt > 1 && state.reviewing && <span className="recite-attempt">attempt {state.attempt}/3</span>}
      {state.phase === 'listening' && (
        <>
          <span className="mic-dot" aria-hidden />
          <button onClick={onStopListening}>Done</button>
        </>
      )}
      {pending > 0 && <span className="recite-pending">checking {pending}…</span>}
      {failed > 0 && state.phase !== 'done' && !state.reviewing && (
        <span className="recite-failed-count">{failed} to review</span>
      )}
      {state.result && !showFeedback && state.reviewing && state.phase === 'correct' && (
        <span className="recite-heard" dir="rtl">
          heard: «{state.result.transcription || '—'}» ({state.result.score})
        </span>
      )}
      {state.phase === 'error' && <span className="recite-error">{state.error}</span>}
      {inPractice && state.phase !== 'done' && state.phase !== 'finishing' && <button onClick={onSkip}>Skip →</button>}
      <button className="recite-cancel" onClick={onCancel} title="Stop">
        ✕
      </button>
    </div>
  );
}
