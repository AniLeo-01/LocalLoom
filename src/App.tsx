import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import {
  Camera,
  Download,
  FolderOpen,
  Loader2,
  Mic,
  MonitorUp,
  Play,
  Save,
  Settings,
  Square,
  Trash2,
  WandSparkles,
  X
} from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { buildMarkdown, type Walkthrough } from "./lib/guide";
import { analyzeRecording, type AppConfig, type ExportAsset } from "./lib/modalClient";

type Recording = {
  id: string;
  title: string;
  path: string;
  createdAt: string;
  durationSeconds: number;
};

type CaptureDevice = {
  index: string;
  name: string;
  kind: string;
};

type CaptureDeviceList = {
  video: CaptureDevice[];
  audio: CaptureDevice[];
};

type Status = "idle" | "recording" | "saving" | "analyzing" | "exported" | "error";

const defaultConfig: AppConfig = {
  recordingDirectory: "",
  exportDirectory: "",
  audioDeviceIndex: "",
  screenDeviceIndex: "",
  webcamEnabled: false,
  webcamDeviceIndex: "",
  webcamPosition: "bottom-right",
  webcamSize: 200,
  videoQuality: "720",
  // OpenAI integration
  openaiApiKey: "",
  vlmProvider: "openai",
  vlmModel: "gpt-5.4-mini",
  asrProvider: "openai",
  asrModel: "gpt-4o-mini-transcribe"
};

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function WebcamPreview({ enabled, size }: { enabled: boolean; size: number }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      stopStream();
      return;
    }
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 320, height: 240, frameRate: 15 }, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(() => {
        /* camera permission denied or unavailable */
      });
    return () => {
      cancelled = true;
      stopStream();
    };
  }, [enabled, stopStream]);

  if (!enabled) return null;

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      marginBottom: 16,
    }}>
      <div style={{
        width: Math.min(size, 200),
        height: Math.min(size, 200),
        borderRadius: 16,
        overflow: "hidden",
        border: "3px solid #0d9488",
        background: "linear-gradient(145deg, #f1f5f9, #e2e8f0)",
        flexShrink: 0,
        boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
      }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: "scaleX(-1)",
          }}
        />
      </div>
    </div>
  );
}

/** Live preview during recording: screen screenshots + webcam pip + timer */
function RecordingPreview({
  elapsed,
  webcamEnabled,
  webcamPosition,
  webcamSize,
}: {
  elapsed: number;
  webcamEnabled: boolean;
  webcamPosition: string;
  webcamSize: number;
}) {
  const [screenSrc, setScreenSrc] = useState<string | null>(null);
  const camRef = useRef<HTMLVideoElement>(null);
  const camStreamRef = useRef<MediaStream | null>(null);

  const stopCam = useCallback(() => {
    if (camStreamRef.current) {
      camStreamRef.current.getTracks().forEach((t) => t.stop());
      camStreamRef.current = null;
    }
  }, []);

  // Poll screenshots every 500ms
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const b64 = await invoke<string>("grab_screenshot");
          if (!cancelled) setScreenSrc(`data:image/jpeg;base64,${b64}`);
        } catch { /* ignore */ }
        await new Promise((r) => setTimeout(r, 500));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  // Webcam stream
  useEffect(() => {
    if (!webcamEnabled) return;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 160, height: 160, frameRate: 15 }, audio: false })
      .then((stream) => {
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        camStreamRef.current = stream;
        if (camRef.current) camRef.current.srcObject = stream;
      })
      .catch(() => {});
    return () => { cancelled = true; stopCam(); };
  }, [webcamEnabled, stopCam]);

  // Webcam pip position
  const pipPos: React.CSSProperties = {};
  if (webcamPosition.includes("top")) pipPos.top = 16;
  else pipPos.bottom = 16;
  if (webcamPosition.includes("left")) pipPos.left = 16;
  else pipPos.right = 16;

  const pipSize = Math.min(webcamSize, 120);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", background: "linear-gradient(145deg, #1e293b 0%, #0f172a 100%)", borderRadius: 12 }}>
      {/* Screen capture preview */}
      {screenSrc ? (
        <img
          src={screenSrc}
          alt="Screen preview"
          style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 12 }}
        />
      ) : (
        <div style={{
          width: "100%", height: "100%",
          display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12,
          color: "rgba(255,255,255,0.4)", fontSize: "0.875rem", fontWeight: 500,
        }}>
          <Loader2 size={24} style={{ animation: "spin 1s linear infinite", opacity: 0.6 }} />
          Capturing screen...
        </div>
      )}

      {/* Recording indicator + timer */}
      <div style={{
        position: "absolute",
        top: 16,
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        alignItems: "center",
        gap: 10,
        background: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(8px)",
        borderRadius: 24,
        padding: "8px 16px",
        color: "white",
        fontSize: "0.8125rem",
        fontWeight: 600,
        fontVariantNumeric: "tabular-nums",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
      }}>
        <div style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: "#ef4444",
          animation: "pulse 1.2s ease-in-out infinite",
          boxShadow: "0 0 8px rgba(239, 68, 68, 0.6)",
        }} />
        REC {formatDuration(elapsed)}
      </div>

      {/* Webcam pip */}
      {webcamEnabled && (
        <div style={{
          position: "absolute",
          ...pipPos,
          width: pipSize,
          height: pipSize,
          borderRadius: "50%",
          overflow: "hidden",
          border: "3px solid rgba(255, 255, 255, 0.9)",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)",
          background: "linear-gradient(145deg, #1e293b, #0f172a)",
        }}>
          <video
            ref={camRef}
            autoPlay
            muted
            playsInline
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scaleX(-1)",
            }}
          />
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("Ready to record a walkthrough.");
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [activeRecording, setActiveRecording] = useState<Recording | null>(null);
  const [walkthrough, setWalkthrough] = useState<Walkthrough | null>(null);
  const [exportAssets, setExportAssets] = useState<ExportAsset[]>([]);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [devices, setDevices] = useState<CaptureDeviceList>({ video: [], audio: [] });

  const startedAt = useRef<number>(0);
  const tick = useRef<number | null>(null);

  const canAnalyze = Boolean(activeRecording);
  const markdown = useMemo(() => (walkthrough ? buildMarkdown(walkthrough) : ""), [walkthrough]);

  async function loadSettings() {
    const saved = await invoke<Partial<AppConfig>>("load_config");
    setConfig({ ...defaultConfig, ...saved });
  }

  async function loadDevices() {
    try {
      const list = await invoke<CaptureDeviceList>("list_capture_devices");
      setDevices(list);
    } catch {
      // ffmpeg may not be installed yet
    }
  }

  React.useEffect(() => {
    void loadSettings();
    void loadDevices();
  }, []);

  async function saveSettings(nextConfig: AppConfig) {
    setConfig(nextConfig);
    await invoke("save_config", { config: nextConfig });
    setMessage("Settings saved.");
  }

  async function startRecording() {
    try {
      const path = await invoke<string>("start_native_recording", { withMic: true });
      setRecordingPath(path);
      startedAt.current = Date.now();
      setElapsed(0);
      tick.current = window.setInterval(() => {
        setElapsed(Math.round((Date.now() - startedAt.current) / 1000));
      }, 1000);
      setStatus("recording");
      setMessage("Recording screen and microphone via ffmpeg.");
    } catch (error) {
      setStatus("error");
      setMessage(String(error));
    }
  }

  async function stopRecording() {
    if (tick.current) window.clearInterval(tick.current);
    setStatus("saving");
    setMessage(config.webcamEnabled ? "Finalizing and compositing webcam overlay..." : "Finalizing recording...");
    try {
      const saved = await invoke<Recording>("stop_native_recording");
      setRecordings((items) => [saved, ...items]);
      setActiveRecording(saved);
      setRecordingPath(null);
      setStatus("idle");
      setMessage("Recording saved. Generate a tutorial when you are ready.");
    } catch (error) {
      setStatus("error");
      setMessage(String(error));
    }
  }

  async function generateGuide() {
    if (!activeRecording) return;
    setStatus("analyzing");
    setMessage("Sending recording to Modal for transcription and visual analysis.");
    try {
      const result = await analyzeRecording(activeRecording.path, config);
      setWalkthrough(result.walkthrough);
      setExportAssets(result.assets);
      setStatus("idle");
      setMessage("Tutorial draft generated.");
    } catch (error) {
      setStatus("error");
      setMessage(String(error));
    }
  }

  async function exportGuide() {
    if (!activeRecording || !walkthrough) return;
    const exported = await invoke<string>("export_guide", {
      request: {
        recordingPath: activeRecording.path,
        walkthrough,
        markdown,
        assets: exportAssets
      }
    });
    setStatus("exported");
    setMessage(`Exported to ${exported}`);
  }

  async function deleteRecording(recording: Recording) {
    try {
      await invoke("delete_recording", { path: recording.path });
      setRecordings((items) => items.filter((r) => r.id !== recording.id));
      if (activeRecording?.id === recording.id) {
        setActiveRecording(null);
        setWalkthrough(null);
        setExportAssets([]);
      }
      setMessage("Recording deleted.");
    } catch (error) {
      setMessage(String(error));
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">OpenLoom</p>
          <h1>Record a walkthrough. Ship the tutorial.</h1>
        </div>
        <div className="topbarRight">
          <div className="statusPill" data-status={status}>
            {status === "analyzing" || status === "saving" ? <Loader2 size={16} className="spin" /> : null}
            <span>{message}</span>
          </div>
          <button className="iconButton" onClick={() => { setShowSettings(true); void loadDevices(); }} title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </section>

      <section className="layout">
        <aside className="panel sidebar">
          <div className="sectionTitle">
            <FolderOpen size={18} />
            <h2>Library</h2>
          </div>
          <div className="recordingList">
            {recordings.length === 0 ? <p className="muted">No recordings yet.</p> : null}
            {recordings.map((recording) => (
              <div
                className={recording.id === activeRecording?.id ? "recording active" : "recording"}
                key={recording.id}
                onClick={() => setActiveRecording(recording)}
              >
                <div className="recordingInfo">
                  <strong>{recording.title}</strong>
                  <span>{new Date(recording.createdAt).toLocaleString()}</span>
                </div>
                <button
                  className="iconButton deleteButton"
                  title="Delete recording"
                  onClick={(event) => { event.stopPropagation(); void deleteRecording(recording); }}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="workspace">
          <div className="recorder">
            <div className="preview">
              {status === "recording" ? (
                <RecordingPreview
                  elapsed={elapsed}
                  webcamEnabled={config.webcamEnabled}
                  webcamPosition={config.webcamPosition}
                  webcamSize={config.webcamSize}
                />
              ) : activeRecording ? (
                <video src={convertFileSrc(activeRecording.path)} controls />
              ) : (
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 16,
                  color: "rgba(148, 163, 184, 0.6)",
                }}>
                  <MonitorUp size={56} strokeWidth={1.5} />
                  <span style={{ fontSize: "0.875rem", fontWeight: 500 }}>
                    Click "Start recording" to begin
                  </span>
                </div>
              )}
            </div>
            <div className="controls">
              {status === "recording" ? (
                <button className="danger" onClick={() => void stopRecording()}>
                  <Square size={18} />
                  Stop {formatDuration(elapsed)}
                </button>
              ) : (
                <button onClick={() => void startRecording()}>
                  <Play size={18} />
                  Start recording
                </button>
              )}
              <button className="secondary" disabled={!canAnalyze || status === "analyzing"} onClick={() => void generateGuide()}>
                <WandSparkles size={18} />
                Generate tutorial
              </button>
              <button className="secondary" disabled={!walkthrough} onClick={() => void exportGuide()}>
                <Download size={18} />
                Export
              </button>
              <span className="hint">
                <Mic size={15} />
                Microphone improves intent detection.
              </span>
              {config.webcamEnabled && (
                <span className="hint">
                  <Camera size={15} />
                  Webcam overlay enabled.
                </span>
              )}
            </div>
          </div>

          <div className="guidePane">
            <div className="sectionTitle">
              <WandSparkles size={18} />
              <h2>Markdown tutorial</h2>
            </div>
            <textarea readOnly value={markdown || "# Your generated guide will appear here\n"} />
          </div>
        </section>
      </section>

      {showSettings && (
        <div className="overlay" onClick={() => setShowSettings(false)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modalHeader">
              <div className="sectionTitle">
                <Settings size={18} />
                <h2>Settings</h2>
              </div>
              <button className="iconButton" onClick={() => setShowSettings(false)}>
                <X size={18} />
              </button>
            </div>
            <label>
              Screen
              <select
                value={config.screenDeviceIndex}
                onChange={(event) => setConfig({ ...config, screenDeviceIndex: event.target.value })}
              >
                <option value="">Auto-detect</option>
                {devices.video
                  .filter((device) => device.kind === "screen")
                  .map((device) => (
                    <option key={device.index} value={device.index}>
                      [{device.index}] {device.name}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              Video quality
              <select
                value={config.videoQuality}
                onChange={(event) => setConfig({ ...config, videoQuality: event.target.value })}
              >
                <option value="720">720p (1280×720) — fastest</option>
                <option value="1080">1080p (1920×1080)</option>
                <option value="1440">1440p (2560×1440)</option>
                <option value="native">Native (Retina) — slowest</option>
              </select>
            </label>
            <label>
              Microphone
              <select
                value={config.audioDeviceIndex}
                onChange={(event) => setConfig({ ...config, audioDeviceIndex: event.target.value })}
              >
                <option value="">Auto-detect</option>
                {devices.audio.map((device) => (
                  <option key={device.index} value={device.index}>
                    [{device.index}] {device.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Camera size={15} style={{ color: "#0d9488" }} />
                Webcam overlay
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <input
                  type="checkbox"
                  checked={config.webcamEnabled}
                  onChange={(event) => setConfig({ ...config, webcamEnabled: event.target.checked })}
                  style={{ width: 18, height: 18, minHeight: "auto", cursor: "pointer" }}
                />
                <span style={{ fontWeight: 500, fontSize: "0.8125rem", color: "#64748b" }}>Enable webcam pip</span>
              </div>
            </label>
            {config.webcamEnabled && (
              <>
                <WebcamPreview enabled={config.webcamEnabled && showSettings} size={config.webcamSize} />
                <label>
                  Webcam device
                  <select
                    value={config.webcamDeviceIndex}
                    onChange={(event) => setConfig({ ...config, webcamDeviceIndex: event.target.value })}
                  >
                    <option value="">Auto-detect</option>
                    {devices.video
                      .filter((device) => device.kind === "camera")
                      .map((device) => (
                        <option key={device.index} value={device.index}>
                          [{device.index}] {device.name}
                        </option>
                      ))}
                  </select>
                </label>
                <label>
                  Webcam position
                  <select
                    value={config.webcamPosition}
                    onChange={(event) => setConfig({ ...config, webcamPosition: event.target.value })}
                  >
                    <option value="bottom-right">Bottom right</option>
                    <option value="bottom-left">Bottom left</option>
                    <option value="top-right">Top right</option>
                    <option value="top-left">Top left</option>
                  </select>
                </label>
                <label>
                  <span>Overlay size — <strong style={{ color: "#0d9488" }}>{config.webcamSize}px</strong></span>
                  <input
                    type="range"
                    min={100}
                    max={400}
                    step={20}
                    value={config.webcamSize}
                    onChange={(event) => setConfig({ ...config, webcamSize: Number(event.target.value) })}
                    style={{ minHeight: "auto", border: "none", padding: 0, cursor: "pointer" }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.6875rem", color: "#94a3b8", marginTop: -2, fontWeight: 500 }}>
                    <span>Small</span>
                    <span>Large</span>
                  </div>
                </label>
              </>
            )}
            <label>
              Recording directory
              <div className="dirPicker">
                <span className="dirPath">{config.recordingDirectory || "App data default"}</span>
                <button className="secondary" onClick={async () => {
                  const selected = await open({ directory: true, title: "Choose recording directory" });
                  if (selected) setConfig({ ...config, recordingDirectory: selected });
                }}>
                  <FolderOpen size={16} />
                  Browse
                </button>
              </div>
            </label>
            <label>
              Export directory
              <div className="dirPicker">
                <span className="dirPath">{config.exportDirectory || "App data default"}</span>
                <button className="secondary" onClick={async () => {
                  const selected = await open({ directory: true, title: "Choose export directory" });
                  if (selected) setConfig({ ...config, exportDirectory: selected });
                }}>
                  <FolderOpen size={16} />
                  Browse
                </button>
              </div>
            </label>

            {/* AI Provider Settings */}
            <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: 16, marginTop: 8 }}>
              <h3 style={{ fontSize: "0.875rem", fontWeight: 600, color: "#64748b", marginBottom: 12 }}>
                AI Provider Settings
              </h3>

              <label>
                Vision Model
                <select
                  value={config.vlmModel}
                  onChange={(e) => setConfig({ ...config, vlmModel: e.target.value })}
                >
                  <option value="gpt-5.4-mini">GPT-5.4 Mini (fastest, cheapest)</option>
                  <option value="gpt-5.2">GPT-5.2</option>
                  <option value="gpt-5.4">GPT-5.4</option>
                  <option value="gpt-5.5">GPT-5.5 (most capable)</option>
                </select>
              </label>

              <label>
                Transcription Model
                <select
                  value={config.asrModel}
                  onChange={(e) => setConfig({ ...config, asrModel: e.target.value })}
                >
                  <option value="gpt-4o-mini-transcribe">GPT-4o Mini Transcribe (faster)</option>
                  <option value="gpt-4o-transcribe">GPT-4o Transcribe (more accurate)</option>
                </select>
              </label>

              <label>
                OpenAI API Key
                <input
                  type="password"
                  placeholder="sk-..."
                  value={config.openaiApiKey}
                  onChange={(e) => setConfig({ ...config, openaiApiKey: e.target.value })}
                  style={{ fontFamily: "monospace" }}
                />
                <span style={{ fontSize: "0.6875rem", color: "#94a3b8", marginTop: 2, display: "block" }}>
                  Stored locally. Used for vision analysis and transcription.
                </span>
              </label>
            </div>

            <div className="modalActions">
              <button className="secondary" onClick={() => setShowSettings(false)}>
                Cancel
              </button>
              <button onClick={() => { void saveSettings(config); setShowSettings(false); }}>
                <Save size={16} />
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
