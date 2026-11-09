import { useCallback, useEffect, useRef } from 'react';

// One shared element so rapid hover sweeps never stack overlapping sounds.
const sharedAudio = new Audio();

export function playClip(url: string): Promise<void> {
  return new Promise((resolve) => {
    sharedAudio.src = url;
    sharedAudio.onended = () => resolve();
    sharedAudio.onerror = () => resolve();
    sharedAudio.play().catch(() => resolve());
  });
}

export function stopClip(): void {
  sharedAudio.pause();
  sharedAudio.onended = null;
}

/** Debounced hover-to-play on the shared audio element. */
export function useHoverAudio(enabled: boolean, debounceMs = 250) {
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const hoverStart = useCallback(
    (url: string) => {
      if (!enabled) return;
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => {
        void playClip(url);
      }, debounceMs);
    },
    [enabled, debounceMs],
  );

  const hoverEnd = useCallback(() => {
    window.clearTimeout(timer.current);
    stopClip();
  }, []);

  return { hoverStart, hoverEnd };
}
