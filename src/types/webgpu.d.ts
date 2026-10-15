// Minimal WebGPU surface used by this app (feature detection and adapter info only).
// onnxruntime-web's own typings reference the full WebGPU types; skipLibCheck covers those.

interface BgmGpuAdapterInfo {
  readonly vendor?: string;
  readonly architecture?: string;
  readonly device?: string;
  readonly description?: string;
}

interface BgmGpuAdapter {
  readonly info?: BgmGpuAdapterInfo;
  readonly isFallbackAdapter?: boolean;
  requestAdapterInfo?(): Promise<BgmGpuAdapterInfo>;
}

interface BgmGpu {
  requestAdapter(options?: { powerPreference?: 'low-power' | 'high-performance' }): Promise<BgmGpuAdapter | null>;
}

interface Navigator {
  readonly gpu?: BgmGpu;
}

interface WorkerNavigator {
  readonly gpu?: BgmGpu;
}
