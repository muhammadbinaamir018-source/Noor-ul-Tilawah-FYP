import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { PageImage } from '../components/PageImage';
import { RegionBox } from '../components/RegionBox';
import { ReciteControls } from '../components/ReciteControls';
import type { PageContent, Region } from '../content/types';
import { useHoverAudio } from '../hooks/useHoverAudio';
import { useReciteMachine } from '../hooks/useReciteMachine';

/** Reading order: rows top to bottom, right to left within a row. */
function readingOrder(regions: Region[]): Region[] {
  return [...regions].sort((a, b) => a.row - b.row || b.rect.x - a.rect.x);
}

export function QaidaLessonPage() {
  const { pageId = '' } = useParams();
  const [page, setPage] = useState<PageContent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playReference, setPlayReferenceUi] = useState(true);
  const [fastFlow, setFastFlow] = useState(false);
  const { state, startRecite, startPractice, stopListening, skip, cancel, setPlayReference } = useReciteMachine();
  const hoverEnabled = state.phase === 'idle';
  const { hoverStart, hoverEnd } = useHoverAudio(hoverEnabled);

  useEffect(() => {
    api.page(pageId).then(setPage).catch((e) => setError(String(e)));
    return cancel;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageId]);

  const activeId = state.region?.id;

  // Keep the item being recited in view during a practice run.
  useEffect(() => {
    if (!activeId) return;
    document
      .querySelector('.region-box.active')
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [activeId]);

  if (error) return <p className="error">{error}</p>;
  if (!page) return <p>Loading…</p>;

  const recitable = readingOrder(page.regions).filter((r) => r.expected);
  const flashClass =
    state.phase === 'correct' ? 'flash-correct' : state.phase === 'incorrect' || state.phase === 'moveon' ? 'flash-incorrect' : '';
  // The learner's eyes are on the Qaida text, so the state cue lives on the letter itself.
  const phaseBadge: Partial<Record<typeof state.phase, string>> = {
    playing: '🔊',
    listening: '🎙️',
    evaluating: '⏳',
    correct: '✓',
    incorrect: '✗',
    moveon: '→',
  };

  return (
    <div className="lesson-page">
      <header className="lesson-header">
        <Link to="/">← Lessons</Link>
        <h2>{page.id}</h2>
        {recitable.length > 0 && state.phase === 'idle' && (
          <>
            <button className="practice-btn" onClick={() => void startPractice(recitable, fastFlow)}>
              ▶ Practice page ({recitable.length})
            </button>
            <label className="play-ref-toggle">
              <input
                type="checkbox"
                checked={playReference}
                onChange={(e) => {
                  setPlayReferenceUi(e.target.checked);
                  setPlayReference(e.target.checked);
                }}
              />
              play reference first
            </label>
            <label className="play-ref-toggle" title="Check in the background and review mistakes at the end">
              <input type="checkbox" checked={fastFlow} onChange={(e) => setFastFlow(e.target.checked)} />
              fast flow
            </label>
          </>
        )}
        <span className="hint">Hover to hear · click to recite one · Practice runs the page</span>
      </header>

      <PageImage src={api.imageUrl(page.image)}>
        {() =>
          page.regions.map((r) => (
            <RegionBox
              key={r.id}
              rect={r.rect}
              className={[
                r.expected ? 'recitable' : 'play-only',
                r.id === activeId ? `active ${flashClass}` : '',
                state.verdicts[r.id] ? `check-${state.verdicts[r.id].status}` : '',
              ].join(' ')}
              onMouseEnter={() => hoverStart(api.audioUrl(r.audio))}
              onMouseLeave={hoverEnd}
              onClick={() => {
                if (state.phase !== 'idle') return;
                hoverEnd();
                if (r.expected) void startRecite(r);
              }}
            >
              {r.id === activeId && phaseBadge[state.phase] && (
                <span className={`phase-badge badge-${state.phase}`}>{phaseBadge[state.phase]}</span>
              )}
            </RegionBox>
          ))
        }
      </PageImage>

      {page.regions.length === 0 && (
        <p className="hint">
          This page has no hotspots yet — map them in the <Link to={`/editor/${page.id}`}>editor</Link>.
        </p>
      )}

      <ReciteControls state={state} onStopListening={stopListening} onSkip={skip} onCancel={cancel} />
    </div>
  );
}
