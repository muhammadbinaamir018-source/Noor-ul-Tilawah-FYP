import type { CollectItem, EvaluateResult, Manifest, PageContent, PageStatus, ProgressSummary, Speaker } from '../content/types';
import { getLearnerId } from '../learner';

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

export const api = {
  manifest: () => getJson<Manifest>('/api/content/manifest'),
  page: (id: string) => getJson<PageContent>(`/api/content/pages/${id}`),
  status: () => getJson<PageStatus[]>('/api/editor/status'),
  clips: (audioPage: string) => getJson<string[]>(`/api/editor/clips/${audioPage}`),

  audioUrl: (clip: string) => `/api/audio/${clip}`,
  imageUrl: (file: string) => `/api/content/image/${file}`,

  async savePage(page: PageContent): Promise<void> {
    const res = await fetch(`/api/editor/pages/${page.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(page),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status} ${await res.text()}`);
  },

  // /collect: recording collection for the evaluation dataset
  collectItems: () => getJson<CollectItem[]>('/api/collect/items'),
  newSpeaker: (body: { consent: boolean; ageGroup: Speaker['ageGroup']; native: boolean }) =>
    postJson<Speaker>('/api/collect/speakers', body),
  collectProgress: (speakerId: string) =>
    getJson<{ speakerId: string; status: Record<string, 'recorded' | 'skipped'> }>(`/api/collect/progress/${speakerId}`),
  skipItem: (speakerId: string, itemId: string) => postJson<unknown>('/api/collect/skip', { speakerId, itemId }),
  collectExportUrl: '/api/collect/export',
  async saveRecording(speakerId: string, itemId: string, audio: Blob): Promise<void> {
    const form = new FormData();
    form.append('audio', audio, `${itemId}.webm`);
    form.append('speaker_id', speakerId);
    form.append('item_id', itemId);
    const res = await fetch('/api/collect/recordings', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Save failed: ${res.status} ${await res.text()}`);
  },

  // per-learner progress (data/progress/<learner>.json)
  progress: (learnerId: string) => getJson<ProgressSummary>(`/api/progress/${learnerId}`),
  async resetProgress(learnerId: string): Promise<void> {
    const res = await fetch(`/api/progress/${learnerId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
  },

  async evaluate(regionId: string, audio: Blob): Promise<EvaluateResult> {
    const form = new FormData();
    form.append('audio', audio, 'recitation.webm');
    form.append('region_id', regionId);
    form.append('learner_id', getLearnerId()); // progress tracking; the state machine need not know
    const res = await fetch('/api/evaluate', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Evaluate failed: ${res.status} ${await res.text()}`);
    return res.json();
  },
};
