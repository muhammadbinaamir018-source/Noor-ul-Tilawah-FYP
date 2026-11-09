import { useCallback, useRef } from 'react';

export interface RecordingHandle {
  /** Resolves with the recorded blob when stopped (manually, by silence, or by the max-duration cap). */
  blob: Promise<Blob>;
  stop: () => void;
}

const SILENCE_RMS = 0.015;
const SILENCE_HOLD_MS = 1200;

/**
 * MediaRecorder wrapper: start() records webm/opus until stop(), maxMs, or —
 * once speech has been heard — ~1.2s of silence, so the learner never has to
 * click anything to finish an answer.
 */
export function useRecorder() {
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async (maxMs = 7000, autoStopOnSilence = true): Promise<RecordingHandle> => {
    if (!streamRef.current || !streamRef.current.active) {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: { noiseSuppression: true, echoCancellation: true, autoGainControl: true },
      });
    }
    const stream = streamRef.current;
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    let audioCtx: AudioContext | null = null;
    let vadTimer: number | undefined;

    const cleanup = () => {
      window.clearTimeout(maxTimer);
      window.clearInterval(vadTimer);
      void audioCtx?.close().catch(() => {});
    };

    const blob = new Promise<Blob>((resolve) => {
      recorder.onstop = () => {
        cleanup();
        resolve(new Blob(chunks, { type: 'audio/webm' }));
      };
    });

    const stop = () => {
      if (recorder.state === 'recording') recorder.stop();
    };

    recorder.start();
    const maxTimer = window.setTimeout(stop, maxMs);

    if (autoStopOnSilence) {
      audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      let heardSpeech = false;
      let lastVoice = performance.now();
      vadTimer = window.setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        const rms = Math.sqrt(sum / buf.length);
        if (rms > SILENCE_RMS) {
          heardSpeech = true;
          lastVoice = performance.now();
        } else if (heardSpeech && performance.now() - lastVoice > SILENCE_HOLD_MS) {
          stop();
        }
      }, 100);
    }

    return { blob, stop };
  }, []);

  const release = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  return { start, release };
}
