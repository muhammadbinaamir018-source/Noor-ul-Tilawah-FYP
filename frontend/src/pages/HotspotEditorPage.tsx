import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { PageImage } from '../components/PageImage';
import { RegionBox } from '../components/RegionBox';
import type { Expected, ExpectedMode, PageContent, Rect, Region } from '../content/types';
import { playClip } from '../hooks/useHoverAudio';

interface DragState {
  x0: number;
  y0: number;
  rect: Rect | null;
}

function nextRegionId(page: PageContent): string {
  const used = new Set(page.regions.map((r) => r.id));
  for (let i = 1; ; i++) {
    const id = `${page.id}-r${String(i).padStart(3, '0')}`;
    if (!used.has(id)) return id;
  }
}

export function HotspotEditorPage() {
  const { pageId = '' } = useParams();
  const [page, setPage] = useState<PageContent | null>(null);
  const [clips, setClips] = useState<string[]>([]);
  const [audioPages, setAudioPages] = useState<string[]>([]);
  const [currentRow, setCurrentRow] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [message, setMessage] = useState('');
  const drag = useRef<DragState | null>(null);
  const [dragRect, setDragRect] = useState<Rect | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.page(pageId).then((p) => {
      setPage(p);
      const maxRow = Math.max(0, ...p.regions.map((r) => r.row));
      setCurrentRow(Math.max(1, maxRow));
    });
    api.status().then((all) => {
      setAudioPages([...new Set(all.map((s) => s.audioPage).filter((a): a is string => !!a))]);
    });
  }, [pageId]);

  useEffect(() => {
    if (page?.audioPage) api.clips(page.audioPage).then(setClips);
    else setClips([]);
  }, [page?.audioPage]);

  const assigned = useMemo(() => new Set(page?.regions.map((r) => r.audio) ?? []), [page]);
  const nextClip = clips.find((c) => !assigned.has(c));
  const selected = page?.regions.find((r) => r.id === selectedId) ?? null;

  const mutate = useCallback((fn: (p: PageContent) => PageContent) => {
    setPage((p) => (p ? fn(p) : p));
    setDirty(true);
  }, []);

  const updateRegion = (id: string, patch: Partial<Region>) =>
    mutate((p) => ({
      ...p,
      regions: p.regions.map((r) => (r.id === id ? { ...r, ...patch } : r)),
    }));

  const updateExpected = (id: string, patch: Partial<Expected> | null) =>
    mutate((p) => ({
      ...p,
      regions: p.regions.map((r) => {
        if (r.id !== id) return r;
        if (patch === null) return { ...r, expected: null };
        const base: Expected = r.expected ?? { text: '', variants: [], mode: 'syllable' };
        return { ...r, expected: { ...base, ...patch } };
      }),
    }));

  // --- drag-to-draw ---------------------------------------------------------

  const toFraction = (e: React.MouseEvent): { x: number; y: number } | null => {
    const el = overlayRef.current;
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)),
      y: Math.min(1, Math.max(0, (e.clientY - b.top) / b.height)),
    };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const p = toFraction(e);
    if (!p) return;
    drag.current = { x0: p.x, y0: p.y, rect: null };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!drag.current) return;
    const p = toFraction(e);
    if (!p) return;
    const rect: Rect = {
      x: Math.min(drag.current.x0, p.x),
      y: Math.min(drag.current.y0, p.y),
      w: Math.abs(p.x - drag.current.x0),
      h: Math.abs(p.y - drag.current.y0),
    };
    drag.current.rect = rect;
    setDragRect(rect);
  };

  const onMouseUp = () => {
    const rect = drag.current?.rect;
    drag.current = null;
    setDragRect(null);
    if (!rect || !page || rect.w < 0.01 || rect.h < 0.01) return;
    const region: Region = {
      id: nextRegionId(page),
      rect,
      audio: nextClip ?? '',
      row: currentRow,
      expected: null,
    };
    mutate((p) => ({ ...p, regions: [...p.regions, region] }));
    setSelectedId(region.id);
    if (region.audio) void playClip(api.audioUrl(region.audio));
    else setMessage('No unassigned clips left for this audio page — pick one manually.');
  };

  // --------------------------------------------------------------------------

  const save = async () => {
    if (!page) return;
    try {
      await api.savePage(page);
      setDirty(false);
      setMessage(`Saved ${page.regions.length} regions.`);
    } catch (err) {
      setMessage(String(err));
    }
  };

  if (!page) return <p>Loading…</p>;

  return (
    <div className="editor-page">
      <header className="editor-header">
        <Link to="/">← Lessons</Link>
        <h2>Editor — {page.id}</h2>
        <label>
          Audio page:{' '}
          <select
            value={page.audioPage ?? ''}
            onChange={(e) => mutate((p) => ({ ...p, audioPage: e.target.value || null }))}
          >
            <option value="">none</option>
            {audioPages.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label>
          Row:{' '}
          <input
            type="number"
            min={1}
            value={currentRow}
            onChange={(e) => setCurrentRow(Number(e.target.value) || 1)}
            style={{ width: '3.5em' }}
          />
        </label>
        <button onClick={() => setCurrentRow((r) => r + 1)}>New row</button>
        <span className="editor-progress">
          {assigned.size}/{clips.length} clips assigned {nextClip ? `· next: ${nextClip}` : ''}
        </span>
        <button className="save-btn" onClick={save} disabled={!dirty}>
          {dirty ? 'Save*' : 'Saved'}
        </button>
      </header>
      {message && <p className="editor-message">{message}</p>}

      <div className="editor-body">
        <div className="editor-canvas">
          <PageImage
            src={api.imageUrl(page.image)}
            overlayRef={overlayRef}
            overlayProps={{
              onMouseDown,
              onMouseMove,
              onMouseUp,
              onMouseLeave: onMouseUp,
              style: { cursor: 'crosshair', pointerEvents: 'auto' },
            }}
          >
            {() => (
              <>
                {page.regions.map((r) => (
                  <RegionBox
                    key={r.id}
                    rect={r.rect}
                    className={`editor-region row-${r.row % 6} ${r.id === selectedId ? 'selected' : ''} ${r.expected ? 'has-expected' : ''}`}
                    title={`${r.id} · ${r.audio} · row ${r.row}`}
                    onClick={() => {
                      setSelectedId(r.id);
                      if (r.audio) void playClip(api.audioUrl(r.audio));
                    }}
                  >
                    <span className="region-label">{r.audio.replace(/^a\d{2}0*/, '').replace('.mp3', '')}</span>
                  </RegionBox>
                ))}
                {dragRect && <RegionBox rect={dragRect} className="drag-preview" />}
              </>
            )}
          </PageImage>
        </div>

        <aside className="editor-side">
          {selected ? (
            <div className="region-editor">
              <h3>{selected.id}</h3>
              <label>
                Clip:{' '}
                <select
                  value={selected.audio}
                  onChange={(e) => {
                    updateRegion(selected.id, { audio: e.target.value });
                    void playClip(api.audioUrl(e.target.value));
                  }}
                >
                  <option value="">—</option>
                  {clips.map((c) => (
                    <option key={c} value={c}>
                      {c} {assigned.has(c) && c !== selected.audio ? '(assigned)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <button onClick={() => selected.audio && playClip(api.audioUrl(selected.audio))}>▶ Play</button>
              <label>
                Row:{' '}
                <input
                  type="number"
                  min={1}
                  value={selected.row}
                  onChange={(e) => updateRegion(selected.id, { row: Number(e.target.value) || 1 })}
                  style={{ width: '3.5em' }}
                />
              </label>
              <label>
                Expected text (Arabic, enables recite):
                <input
                  dir="rtl"
                  lang="ar"
                  value={selected.expected?.text ?? ''}
                  onChange={(e) =>
                    e.target.value ? updateExpected(selected.id, { text: e.target.value }) : updateExpected(selected.id, null)
                  }
                  placeholder="اَ"
                />
              </label>
              {selected.expected && (
                <>
                  <label>
                    Mode:{' '}
                    <select
                      value={selected.expected.mode}
                      onChange={(e) => updateExpected(selected.id, { mode: e.target.value as ExpectedMode })}
                    >
                      <option value="letter-name">letter-name</option>
                      <option value="syllable">syllable</option>
                      <option value="word">word</option>
                      <option value="phrase">phrase</option>
                    </select>
                  </label>
                  <label>
                    Variants (comma-separated):
                    <input
                      dir="rtl"
                      lang="ar"
                      value={selected.expected.variants.join('، ')}
                      onChange={(e) =>
                        updateExpected(selected.id, {
                          variants: e.target.value
                            .split(/[,،]/)
                            .map((v) => v.trim())
                            .filter(Boolean),
                        })
                      }
                    />
                  </label>
                </>
              )}
              <button
                className="danger"
                onClick={() => {
                  mutate((p) => ({ ...p, regions: p.regions.filter((r) => r.id !== selected.id) }));
                  setSelectedId(null);
                }}
              >
                Delete region
              </button>
            </div>
          ) : (
            <p className="hint">
              Drag on the page to draw a hotspot — the next unassigned clip is auto-assigned and played. Click a box to
              edit it.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
