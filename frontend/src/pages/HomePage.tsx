import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { PageStatus, ProgressSummary } from '../content/types';
import { getLearnerId, setLearnerId } from '../learner';

export function HomePage() {
  const [status, setStatus] = useState<PageStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [learner, setLearner] = useState(getLearnerId);
  const [draft, setDraft] = useState(learner);
  const [progress, setProgress] = useState<ProgressSummary | null>(null);

  useEffect(() => {
    api.status().then(setStatus).catch((e) => setError(String(e)));
  }, []);

  const loadProgress = useCallback((id: string) => {
    api
      .progress(id)
      .then(setProgress)
      .catch(() => setProgress(null));
  }, []);
  useEffect(() => loadProgress(learner), [learner, loadProgress]);

  const commitLearner = () => {
    const id = setLearnerId(draft);
    setDraft(id);
    setLearner(id);
  };

  if (error) return <p className="error">{error}</p>;

  const arabicNum = (n: number) =>
    String(n).replace(/\d/g, (d) => '٠١٢٣٤٥٦٧٨٩'[Number(d)]);

  return (
    <div className="home">
      <h1>تلاوة</h1>
      <p className="subtitle">Learn to read the Quran, starting from the Noorani Qaida.</p>
      <p className="dev-links">
        <Link to="/logs">evaluation log</Link> · <Link to="/collect">collect recordings</Link>
      </p>

      <section className="progress-panel">
        <label className="learner-field">
          practising as
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitLearner}
            onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
            maxLength={40}
            aria-label="learner name"
          />
        </label>
        {progress && progress.attempts > 0 ? (
          <>
            <p className="progress-counts">
              {progress.itemsPassed} passed · {progress.itemsAttempted} tried · {progress.attempts} attempts
            </p>
            {progress.weakLetters.length > 0 && (
              <p className="weak-letters" dir="rtl" lang="ar">
                {progress.weakLetters.map((w) => (
                  <span key={w.letter} className="weak-letter" title={`${w.fails} of ${w.attempts} attempts failed`}>
                    {w.letter}
                  </span>
                ))}
              </p>
            )}
            {progress.revise.length > 0 && (
              <div className="revise">
                <h3>Revise these</h3>
                <ul>
                  {progress.revise.map((r) => (
                    <li key={r.regionId}>
                      <Link to={`/lesson/${r.pageId}`} dir="rtl" lang="ar" className="revise-text">
                        {r.text}
                      </Link>
                      <span className="revise-meta">
                        page {Number(r.pageId.slice(1))} · failed {r.fails}×
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <button
              className="progress-reset"
              onClick={() => void api.resetProgress(learner).then(() => loadProgress(learner))}
            >
              reset progress
            </button>
          </>
        ) : (
          <p className="progress-counts">No attempts yet: open a page and click a letter to recite.</p>
        )}
      </section>

      <div className="lesson-grid">
        {status.map((s, i) => (
          <div key={s.id} className={`lesson-card ${s.regions > 0 ? 'mapped' : 'unmapped'}`}>
            <Link to={`/lesson/${s.id}`} className="lesson-title">
              صفحة<span className="page-num">{arabicNum(i + 1)}</span>
            </Link>
            <div className="lesson-meta">
              {s.audioPage ? (
                <span>
                  {s.assignedClips}/{s.totalClips} clips · {s.regionsWithExpected} recitable
                </span>
              ) : (
                <span>no audio page</span>
              )}
              <Link to={`/editor/${s.id}`} className="edit-link">
                edit
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
