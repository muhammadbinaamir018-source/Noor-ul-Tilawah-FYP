// Tiny audible cues so the learner's eyes can stay on the Qaida text.
let ctx: AudioContext | null = null;

function tone(freq: number, ms: number, delayMs = 0, gain = 0.08) {
  ctx ??= new AudioContext();
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  const t0 = ctx.currentTime + delayMs / 1000;
  osc.frequency.value = freq;
  osc.type = 'sine';
  g.gain.setValueAtTime(gain, t0);
  g.gain.exponentialRampToValueAtTime(0.001, t0 + ms / 1000);
  osc.connect(g).connect(ctx.destination);
  osc.start(t0);
  osc.stop(t0 + ms / 1000);
}

/** "Your turn" — short rising beep when the mic opens. */
export function cueListen() {
  tone(660, 120);
  tone(880, 140, 110);
}

/** Soft low note on a mistake (the replayed reference does the real teaching). */
export function cueIncorrect() {
  tone(220, 220, 0, 0.06);
}
