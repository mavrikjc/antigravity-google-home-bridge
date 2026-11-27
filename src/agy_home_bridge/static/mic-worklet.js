// AudioWorklet: capture mic at the device's native rate (often 48 kHz on iOS),
// downsample to 16 kHz mono Int16, and post ~40 ms frames to the main thread.
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // `sampleRate` is the AudioContext's real rate
    this.acc = 0;
    this.accN = 0;
    this.pos = 0;
    this.frame = new Int16Array(640); // 40 ms @ 16 kHz
    this.idx = 0;
    this.sumSq = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // Box-filter average over each output sample's span (cheap anti-aliasing)
      this.acc += ch[i];
      this.accN += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const s = Math.max(-1, Math.min(1, this.acc / this.accN));
        this.acc = 0;
        this.accN = 0;
        this.sumSq += s * s;
        this.frame[this.idx++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this.idx === this.frame.length) {
          const rms = Math.sqrt(this.sumSq / this.frame.length);
          this.port.postMessage({ pcm: this.frame.buffer, rms }, [this.frame.buffer]);
          this.frame = new Int16Array(640);
          this.idx = 0;
          this.sumSq = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCaptureProcessor);
