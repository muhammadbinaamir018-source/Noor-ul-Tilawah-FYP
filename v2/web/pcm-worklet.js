// Captures microphone audio, resamples to 16 kHz mono and posts ~100 ms int16 chunks.
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;            // fractional read position into the carried-over input
    this.carry = new Float32Array(0);
    this.out = [];
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const x = new Float32Array(this.carry.length + ch.length);
    x.set(this.carry);
    x.set(ch, this.carry.length);
    // Box-average over each output period (cheap anti-aliasing), then decimate.
    const r = this.ratio;
    let p = this.pos;
    while (p + r <= x.length) {
      const a = Math.floor(p), b = Math.floor(p + r);
      let s = 0;
      for (let i = a; i < b; i++) s += x[i];
      this.out.push(b > a ? s / (b - a) : x[a]);
      p += r;
    }
    const keep = Math.floor(p);
    this.carry = x.slice(keep);
    this.pos = p - keep;
    if (this.out.length >= 1600) {
      const pcm = new Int16Array(this.out.length);
      for (let i = 0; i < this.out.length; i++) {
        const v = Math.max(-1, Math.min(1, this.out[i]));
        pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
      this.out = [];
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);
