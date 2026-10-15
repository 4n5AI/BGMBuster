/**
 * Real-input FFT for power-of-two sizes, used by the STFT / iSTFT.
 *
 * `forward` computes the one-sided spectrum X[0..N/2] of a real frame x[0..N-1] (like numpy.fft.rfft and
 * torch.fft.rfft); `inverse` is its exact inverse (like numpy.fft.irfft, scaled by 1/N). Both use the
 * standard trick of one complex FFT of size N/2 over z[k] = x[2k] + i x[2k+1]. Tables are cached per size.
 */
export class RealFFT {
  readonly size: number;
  private readonly half: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array; // twiddles for the N/2 complex FFT
  private readonly sin: Float64Array;
  private readonly cosN: Float64Array; // e^{-2πik/N} for the real/complex split, k in [0, N/2]
  private readonly sinN: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;

  private static cache = new Map<number, RealFFT>();

  static get(size: number): RealFFT {
    let f = RealFFT.cache.get(size);
    if (!f) {
      f = new RealFFT(size);
      RealFFT.cache.set(size, f);
    }
    return f;
  }

  constructor(size: number) {
    if (size < 4 || (size & (size - 1)) !== 0) throw new Error(`RealFFT: size must be a power of two >= 4, got ${size}`);
    this.size = size;
    const M = size >> 1;
    this.half = M;
    const bits = Math.log2(M);
    this.rev = new Uint32Array(M);
    for (let i = 0; i < M; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(M >> 1);
    this.sin = new Float64Array(M >> 1);
    for (let k = 0; k < M >> 1; k++) {
      const a = (-2 * Math.PI * k) / M;
      this.cos[k] = Math.cos(a);
      this.sin[k] = Math.sin(a);
    }
    this.cosN = new Float64Array(M + 1);
    this.sinN = new Float64Array(M + 1);
    for (let k = 0; k <= M; k++) {
      const a = (-2 * Math.PI * k) / size;
      this.cosN[k] = Math.cos(a);
      this.sinN[k] = Math.sin(a);
    }
    this.re = new Float64Array(M);
    this.im = new Float64Array(M);
  }

  /** In-place iterative radix-2 complex FFT of length N/2 on (re, im). `inverse` flips the twiddle sign (no scaling). */
  private complexFFT(re: Float64Array, im: Float64Array, inverse: boolean): void {
    const M = this.half;
    const rev = this.rev;
    for (let i = 0; i < M; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? -1 : 1;
    for (let len = 2; len <= M; len <<= 1) {
      const halfLen = len >> 1;
      const step = M / len;
      for (let i = 0; i < M; i += len) {
        for (let j = 0, k = 0; j < halfLen; j++, k += step) {
          const wr = this.cos[k];
          const wi = sgn * this.sin[k];
          const a = i + j;
          const b = a + halfLen;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }

  /**
   * Forward real FFT. `input` has N real samples; writes N/2+1 complex bins to `outRe` / `outIm`.
   */
  forward(input: ArrayLike<number>, outRe: Float32Array | Float64Array, outIm: Float32Array | Float64Array): void {
    const N = this.size;
    const M = this.half;
    const re = this.re;
    const im = this.im;
    for (let k = 0; k < M; k++) {
      re[k] = input[2 * k];
      im[k] = input[2 * k + 1];
    }
    this.complexFFT(re, im, false);
    // X[k] = E[k] + W^k O[k], with E = (Z[k] + conj(Z[M-k]))/2, O = (Z[k] - conj(Z[M-k]))/(2i)
    for (let k = 0; k <= M; k++) {
      const k1 = k === M ? 0 : k;
      const k2 = (M - k) % M;
      const zr1 = re[k1], zi1 = im[k1];
      const zr2 = re[k2], zi2 = -im[k2]; // conj(Z[M-k])
      const er = 0.5 * (zr1 + zr2);
      const ei = 0.5 * (zi1 + zi2);
      // O = (Z - conj) / (2i): (a + bi)/(2i) = (b - ai)/2
      const dr = zr1 - zr2;
      const di = zi1 - zi2;
      const or_ = 0.5 * di;
      const oi = -0.5 * dr;
      const wr = this.cosN[k];
      const wi = this.sinN[k];
      outRe[k] = er + (or_ * wr - oi * wi);
      outIm[k] = ei + (or_ * wi + oi * wr);
    }
  }

  /**
   * Inverse real FFT. Reads N/2+1 complex bins; writes N real samples scaled by 1/N (numpy irfft convention).
   */
  inverse(inRe: ArrayLike<number>, inIm: ArrayLike<number>, output: Float32Array | Float64Array): void {
    const N = this.size;
    const M = this.half;
    const re = this.re;
    const im = this.im;
    // Z[k] = E[k] + i O[k], E = (X[k] + conj(X[M-k]))/2, O = (X[k] - conj(X[M-k])) * W^{-k} / 2
    for (let k = 0; k < M; k++) {
      const xr1 = inRe[k], xi1 = inIm[k];
      const xr2 = inRe[M - k], xi2 = -inIm[M - k]; // conj(X[M-k])
      const er = 0.5 * (xr1 + xr2);
      const ei = 0.5 * (xi1 + xi2);
      const dr = 0.5 * (xr1 - xr2);
      const di = 0.5 * (xi1 - xi2);
      const wr = this.cosN[k];
      const wi = -this.sinN[k]; // W^{-k}
      const or_ = dr * wr - di * wi;
      const oi = dr * wi + di * wr;
      re[k] = er - oi; // E + i*O: real = er - oi, imag = ei + or
      im[k] = ei + or_;
    }
    this.complexFFT(re, im, true);
    const scale = 1 / M;
    for (let k = 0; k < M; k++) {
      output[2 * k] = re[k] * scale;
      output[2 * k + 1] = im[k] * scale;
    }
    void N;
  }
}
