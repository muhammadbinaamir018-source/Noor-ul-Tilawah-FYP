export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type ExpectedMode = 'letter-name' | 'syllable' | 'word' | 'phrase';

export interface Expected {
  text: string;
  variants: string[];
  mode: ExpectedMode;
}

export interface Region {
  id: string;
  rect: Rect;
  audio: string;
  row: number;
  expected?: Expected | null;
}

export interface PageContent {
  id: string;
  image: string;
  audioPage: string | null;
  regions: Region[];
}

export interface Lesson {
  id: string;
  number: number;
  title: string;
  image: string;
  audioPage: string | null;
  file: string;
}

export interface Manifest {
  version: number;
  lessons: Lesson[];
}

export interface PageStatus {
  id: string;
  title: string;
  audioPage: string | null;
  totalClips: number;
  assignedClips: number;
  regions: number;
  regionsWithExpected: number;
}

export type Verdict = 'correct' | 'incorrect' | 'unclear';

/** One item to record on /collect. */
export interface CollectItem {
  id: string;
  pageId: string;
  image: string;
  rect: Rect;
  text: string;
  mode: ExpectedMode;
  audio: string;
}

export interface Speaker {
  speakerId: string;
  ageGroup: 'child' | 'teen' | 'adult';
  native: boolean;
  created: string;
}

/** Vowel-length check from the audio; only when the backend has madd enabled. */
export interface MaddInfo {
  measured_s: number;
  reference_s: number;
  reference_is_default: boolean;
  counts: number;
  expected_counts: number;
  verdict: 'too-short' | '2-count' | '4-count' | 'too-long' | 'unclear';
  ok: boolean;
  confidence: number;
  hint: string;
  clipped: boolean;
}

export interface EvaluateResult {
  regionId: string;
  verdict: Verdict;
  transcription: string;
  normalizedTranscription: string;
  expectedText: string | null;
  bestVariant: string | null;
  score: number;
  threshold: number;
  latencyMs: number;
  madd?: MaddInfo | null;
  /** Vowel-mark agreement 0-100; only when the backend has the harakat signal enabled. */
  harakatScore?: number | null;
  /** Present when the verdict is not correct, or the madd check failed. */
  feedback?: Feedback | null;
}

/** /api/progress/{learner}: what the home page shows. */
export interface ProgressSummary {
  learnerId: string;
  itemsAttempted: number;
  itemsPassed: number;
  attempts: number;
  weakLetters: { letter: string; attempts: number; fails: number }[];
  revise: { regionId: string; pageId: string; text: string; fails: number; lastScore: number | null }[];
}

export interface Feedback {
  message: string;
  heard: string;
  expected: string;
  expectedLetter?: string | null;
  heardLetter?: string | null;
  lowConfidence: boolean;
  maddHint?: string | null;
}
