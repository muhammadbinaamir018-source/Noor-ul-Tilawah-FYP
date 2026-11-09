import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { CollectItem, Speaker } from '../content/types';
import { useRecorder } from '../hooks/useRecorder';

/**
 * /collect — record every recitable item of pages 1-12 for the evaluation
 * dataset. Consent → anonymous speaker id → one item at a time (cell from
 * the scan, expected text, reference audio) → record, listen back, save,
 * re-record or skip. The speaker id is kept in localStorage so a refresh
 * resumes where the volunteer left off.
 */

const STORAGE_KEY = 'tilawah.collect.speaker';
const MAX_MS: Record<CollectItem['mode'], number> = { 'letter-name': 6000, syllable: 6000, word: 8000, phrase: 12000 };
// The page scans are 1080 × 1608 px.
const SCAN_ASPECT = 1080 / 1608;

function loadSpeaker(): Speaker | null {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null');
  } catch {
    return null;
  }
}

/** The item's cell, cropped out of the page scan with background-position. */
function CellPreview({ item }: { item: CollectItem }) {
  const { x, y, w, h } = item.rect;
  const width = 260;
  const height = Math.max(70, Math.min(260, (width * h) / (w * SCAN_ASPECT)));
  return (
    <div
      className="collect-cell"
      style={{
        width,
        height,
        backgroundImage: `url(${api.imageUrl(item.image)})`,
        backgroundSize: `${100 / w}% ${100 / h}%`,
        backgroundPosition: `${w < 1 ? (x / (1 - w)) * 100 : 0}% ${h < 1 ? (y / (1 - h)) * 100 : 0}%`,
      }}
      title={item.id}
    />
  );
}

export function CollectPage() {
  const [speaker, setSpeaker] = useState<Speaker | null>(loadSpeaker);
  const [items, setItems] = useState<CollectItem[]>([]);
  const [done, setDone] = useState<Record<string, 'recorded' | 'skipped'>>({});
  const [index, setIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // consent form
  const [consent, setConsent] = useState(false);
  const [ageGroup, setAgeGroup] = useState<Speaker['ageGroup']>('adult');
  const [native, setNative] = useState(false);

  // recording
  const { start, release } = useRecorder();
  const [phase, setPhase] = useState<'idle' | 'recording' | 'review' | 'saving'>('idle');
  const [blob, setBlob] = useState<Blob | null>(null);
  const stopRef = useRef<() => void>(() => {});
  const blobUrl = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob]);
  useEffect(() => () => void (blobUrl && URL.revokeObjectURL(blobUrl)), [blobUrl]);
  useEffect(() => release, [release]);

  useEffect(() => {
    api.collectItems().then(setItems).catch((e) => setError(String(e)));
  }, []);

  // resume: first item this speaker has not recorded or skipped
  useEffect(() => {
    if (!speaker || items.length === 0) return;
    api
      .collectProgress(speaker.speakerId)
      .then((p) => {
        setDone(p.status);
        const next = items.findIndex((i) => !p.status[i.id]);
        setIndex(next === -1 ? items.length : next);
      })
      .catch(() => {
        // unknown speaker on this server (e.g. recordings folder was moved): start over
        localStorage.removeItem(STORAGE_KEY);
        setSpeaker(null);
      });
  }, [speaker, items]);

  const begin = async () => {
    try {
      const s = await api.newSpeaker({ consent, ageGroup, native });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
      setSpeaker(s);
    } catch (e) {
      setError(String(e));
    }
  };

  const item = index !== null && index < items.length ? items[index] : null;

  const record = async () => {
    if (!item) return;
    setBlob(null);
    setPhase('recording');
    try {
      const handle = await start(MAX_MS[item.mode], true);
      stopRef.current = handle.stop;
      const b = await handle.blob;
      setBlob(b);
      setPhase('review');
    } catch (e) {
      setError(String(e));
      setPhase('idle');
    }
  };

  const advance = (status: 'recorded' | 'skipped') => {
    if (!item) return;
    setDone((d) => ({ ...d, [item.id]: status }));
    setBlob(null);
    setPhase('idle');
    setIndex((i) => (i ?? 0) + 1);
  };

  const save = async () => {
    if (!item || !blob || !speaker) return;
    setPhase('saving');
    try {
      await api.saveRecording(speaker.speakerId, item.id, blob);
      advance('recorded');
    } catch (e) {
      setError(String(e));
      setPhase('review');
    }
  };

  const skip = async () => {
    if (!item || !speaker) return;
    try {
      await api.skipItem(speaker.speakerId, item.id);
      advance('skipped');
    } catch (e) {
      setError(String(e));
    }
  };

  const recorded = Object.values(done).filter((s) => s === 'recorded').length;

  if (error) return <p className="error">{error}</p>;

  if (!speaker) {
    return (
      <div className="collect-page">
        <header className="lesson-header">
          <Link to="/">← Lessons</Link>
          <h2>Help us collect recordings</h2>
        </header>
        <div className="collect-card">
          <p>
            You will read letters, syllables and words from the first 12 pages of the Noorani Qaida into the
            microphone. The recordings are used only to evaluate this project's recitation checker. No name,
            email or other identifying detail is stored: you get an anonymous id such as S03.
          </p>
          <label className="collect-field">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />I agree that my
            recordings may be used for this project's evaluation (required)
          </label>
          <label className="collect-field">
            Age group
            <select value={ageGroup} onChange={(e) => setAgeGroup(e.target.value as Speaker['ageGroup'])}>
              <option value="child">under 12</option>
              <option value="teen">12 – 17</option>
              <option value="adult">18 or over</option>
            </select>
          </label>
          <label className="collect-field">
            <input type="checkbox" checked={native} onChange={(e) => setNative(e.target.checked)} />
            Arabic is my first language
          </label>
          <button className="practice-btn" disabled={!consent} onClick={() => void begin()}>
            Start recording
          </button>
        </div>
      </div>
    );
  }

  if (index === null || items.length === 0) return <p>Loading…</p>;

  if (!item) {
    return (
      <div className="collect-page">
        <header className="lesson-header">
          <Link to="/">← Lessons</Link>
          <h2>All done — thank you, {speaker.speakerId}</h2>
        </header>
        <div className="collect-card">
          <p>
            {recorded} of {items.length} items recorded. The files are in <code>data/recordings/{speaker.speakerId}/</code>.
          </p>
          <p>
            <a href={api.collectExportUrl}>Download the labels template</a> for the qari (all speakers, eval
            harness format).
          </p>
          <button
            onClick={() => {
              localStorage.removeItem(STORAGE_KEY);
              setSpeaker(null);
              setIndex(null);
              setDone({});
            }}
          >
            New speaker
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="collect-page">
      <header className="lesson-header">
        <Link to="/">← Lessons</Link>
        <h2>Recording as {speaker.speakerId}</h2>
        <span className="recite-progress">
          {index + 1}/{items.length} · page {item.pageId.slice(1)}
        </span>
      </header>
      <div className="collect-card">
        <CellPreview item={item} />
        <div className="collect-text" dir="rtl" lang="ar">
          {item.text}
        </div>
        <p className="collect-mode">
          {item.mode === 'letter-name' ? 'say the letter name' : item.mode === 'phrase' ? 'read the phrase' : 'read it'}
          {' · '}
          <button className="collect-link" onClick={() => void new Audio(api.audioUrl(item.audio)).play()}>
            ▶ hear the reference
          </button>
        </p>

        <div className="collect-actions">
          {phase === 'idle' && (
            <>
              <button className="practice-btn" onClick={() => void record()}>
                🎙️ Record
              </button>
              <button onClick={() => void skip()}>Skip</button>
            </>
          )}
          {phase === 'recording' && (
            <>
              <span className="mic-dot" aria-hidden /> <span>Recording — stops by itself when you pause</span>
              <button onClick={() => stopRef.current()}>Done</button>
            </>
          )}
          {(phase === 'review' || phase === 'saving') && blobUrl && (
            <>
              <audio controls src={blobUrl} />
              <button className="practice-btn" disabled={phase === 'saving'} onClick={() => void save()}>
                Save & next
              </button>
              <button disabled={phase === 'saving'} onClick={() => void record()}>
                Re-record
              </button>
              <button disabled={phase === 'saving'} onClick={() => void skip()}>
                Skip
              </button>
            </>
          )}
        </div>
      </div>
      <p className="hint">
        Speak clearly, close to the microphone. Mistakes are fine: wrong recitations are as useful as right ones.
      </p>
    </div>
  );
}
