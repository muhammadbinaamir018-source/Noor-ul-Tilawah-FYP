import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { playClip } from '../hooks/useHoverAudio';

interface LogEntry {
  time: string;
  audio: string;
  regionId: string;
  verdict: 'correct' | 'incorrect' | 'unclear';
  transcription: string;
  normalizedTranscription: string;
  expectedText: string | null;
  bestVariant: string | null;
  variants: string[];
  mode: string;
  score: number;
  threshold: number;
  latencyMs: number;
}

export function EvalLogPage() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    fetch('/api/logs')
      .then((r) => r.json())
      .then(setEntries)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(refresh, [refresh]);

  const clear = async () => {
    await fetch('/api/logs', { method: 'DELETE' });
    refresh();
  };

  if (error) return <p className="error">{error}</p>;

  return (
    <div className="logs-page">
      <header className="editor-header">
        <Link to="/">← Lessons</Link>
        <h2>Evaluation log</h2>
        <span className="hint">Every check: what was expected, what the model heard, and the actual recording.</span>
        <span className="editor-progress">{entries.length} attempts</span>
        <button onClick={refresh}>Refresh</button>
        <button className="danger" onClick={clear}>
          Clear log
        </button>
      </header>

      {entries.length === 0 ? (
        <p className="hint">No evaluations logged yet — recite something on a lesson page first.</p>
      ) : (
        <table className="log-table">
          <thead>
            <tr>
              <th>time</th>
              <th>item</th>
              <th>expected</th>
              <th>heard</th>
              <th>matched variant</th>
              <th>score</th>
              <th>verdict</th>
              <th>ms</th>
              <th>recording</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.audio} className={`verdict-${e.verdict}`}>
                <td className="mono">{e.time.slice(11)}</td>
                <td className="mono">{e.regionId}</td>
                <td dir="rtl" className="ar">
                  {e.expectedText}
                  {e.variants?.length > 0 && <span className="variants"> +{e.variants.length} variants</span>}
                </td>
                <td dir="rtl" className="ar">
                  {e.transcription || '— (silence)'}
                </td>
                <td dir="rtl" className="ar">
                  {e.bestVariant}
                </td>
                <td className="mono">
                  {e.score} / {e.threshold}
                </td>
                <td>{e.verdict === 'correct' ? '✓ correct' : e.verdict === 'unclear' ? '? unclear' : '✗ incorrect'}</td>
                <td className="mono">{e.latencyMs}</td>
                <td>
                  <button onClick={() => void playClip(`/api/logs/audio/${e.audio}`)}>▶</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
