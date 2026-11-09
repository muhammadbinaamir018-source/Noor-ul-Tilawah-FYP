/**
 * Who is practising: a short name kept in localStorage (no accounts). It is
 * sent with every recitation check so the backend can keep per-learner
 * progress in data/progress/<name>.json.
 */
const KEY = 'tilawah.learner';
const DEFAULT = 'learner';
const SAFE = /^[A-Za-z0-9_-]{1,40}$/;

export function getLearnerId(): string {
  try {
    const v = localStorage.getItem(KEY);
    return v && SAFE.test(v) ? v : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

/** Keeps only letters, digits, - and _ (the backend uses it as a file name). */
export function setLearnerId(raw: string): string {
  const id = raw.trim().replace(/[^A-Za-z0-9_-]/g, '-').slice(0, 40) || DEFAULT;
  try {
    localStorage.setItem(KEY, id);
  } catch {
    // private mode: the name just does not persist
  }
  return id;
}
