use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use uuid::Uuid;

static RECORDER: Mutex<Option<RecorderState>> = Mutex::new(None);

struct RecorderState {
    screen_child: Child,
    webcam_child: Option<Child>,
    screen_path: PathBuf,
    webcam_path: Option<PathBuf>,
    final_path: PathBuf,
    started_at: chrono::DateTime<Utc>,
    overlay_size: u32,
    webcam_position: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AppConfig {
    recording_directory: Option<String>,
    export_directory: Option<String>,
    audio_device_index: Option<String>,
    screen_device_index: Option<String>,
    webcam_enabled: Option<bool>,
    webcam_device_index: Option<String>,
    webcam_position: Option<String>,
    webcam_size: Option<u32>,
    video_quality: Option<String>,
    // OpenAI integration
    openai_api_key: Option<String>,
    vlm_provider: Option<String>,
    vlm_model: Option<String>,
    asr_provider: Option<String>,
    asr_model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveRecordingRequest {
    title: String,
    media_base64: String,
    mime_type: String,
    duration_seconds: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SavedRecording {
    id: String,
    title: String,
    path: String,
    created_at: String,
    duration_seconds: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportGuideRequest {
    recording_path: String,
    walkthrough: serde_json::Value,
    markdown: String,
    assets: Option<Vec<ExportAsset>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportAsset {
    name: String,
    media_base64: String,
}

fn find_ffmpeg() -> Result<String, String> {
    let candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ];
    for path in &candidates {
        if Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map_err(|_| "ffmpeg not found. Install it with: brew install ffmpeg".to_string())?;
    Ok("ffmpeg".to_string())
}

#[tauri::command]
fn check_ffmpeg() -> Result<bool, String> {
    Ok(find_ffmpeg().is_ok())
}

/// Extract frames from a video file at a fixed FPS rate as base64 JPEG images.
/// Used to send frames to vision models that don't support video input.
///
/// Parameters:
/// - fps: Frames per second to extract (default: 1.0, i.e., 1 frame per second)
/// - max_frames: Maximum number of frames to extract (default: 60, to limit long videos)
#[tauri::command]
fn extract_frames(
    path: String,
    fps: Option<f64>,
    max_frames: Option<u32>,
) -> Result<Vec<String>, String> {
    let ffmpeg = find_ffmpeg()?;
    let src = PathBuf::from(&path);
    if !src.exists() {
        return Err(format!("Video not found: {path}"));
    }
    let target_fps = fps.unwrap_or(1.0).clamp(0.1, 5.0); // Default 1 fps, max 5 fps
    let max_n = max_frames.unwrap_or(60).clamp(1, 120); // Default max 60 frames

    let tmp = std::env::temp_dir().join("openloom_frames");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(to_string)?;

    // Get video duration first
    let probe = Command::new(&ffmpeg)
        .args(["-i", &path, "-f", "null", "-"])
        .stderr(Stdio::piped())
        .output()
        .map_err(to_string)?;
    let stderr = String::from_utf8_lossy(&probe.stderr);
    let duration: f64 = stderr
        .lines()
        .rev()
        .find_map(|line| {
            if let Some(pos) = line.find("Duration:") {
                let s = &line[pos + 10..];
                let parts: Vec<&str> = s.split(',').next()?.trim().split(':').collect();
                if parts.len() == 3 {
                    let h: f64 = parts[0].parse().ok()?;
                    let m: f64 = parts[1].parse().ok()?;
                    let s: f64 = parts[2].parse().ok()?;
                    return Some(h * 3600.0 + m * 60.0 + s);
                }
            }
            None
        })
        .unwrap_or(10.0);

    // Calculate expected frames and apply max limit
    let expected_frames = (duration * target_fps).ceil() as u32;
    let n = expected_frames.min(max_n);

    println!(
        "[OpenLoom] extract_frames: duration={:.1}s, fps={}, expected={}, max={}, extracting={}",
        duration, target_fps, expected_frames, max_n, n
    );

    // Extract frames at fixed FPS
    let output_pattern = tmp.join("frame_%04d.jpg");
    let status = Command::new(&ffmpeg)
        .args([
            "-i", &path,
            "-vf", &format!("fps={target_fps:.4},scale=512:-2"),
            "-frames:v", &n.to_string(),
            "-q:v", "5",
            "-y",
        ])
        .arg(&output_pattern)
        .stderr(Stdio::null())
        .status()
        .map_err(to_string)?;

    if !status.success() {
        return Err("ffmpeg frame extraction failed".to_string());
    }

    let mut frames = Vec::new();
    for i in 1..=n {
        let frame_path = tmp.join(format!("frame_{:03}.jpg", i));
        if frame_path.exists() {
            let bytes = fs::read(&frame_path).map_err(to_string)?;
            frames.push(STANDARD.encode(bytes));
        }
    }
    let _ = fs::remove_dir_all(&tmp);
    Ok(frames)
}

/// Grab a screenshot using macOS screencapture and return as base64 JPEG.
/// Used for the live recording preview — doesn't conflict with ffmpeg.
#[tauri::command]
fn grab_screenshot() -> Result<String, String> {
    let tmp = std::env::temp_dir().join("openloom_preview.jpg");
    let status = Command::new("screencapture")
        .args(["-x", "-t", "jpg", "-C"])
        .arg(&tmp)
        .status()
        .map_err(to_string)?;
    if !status.success() {
        return Err("screencapture failed".to_string());
    }
    let bytes = fs::read(&tmp).map_err(to_string)?;
    Ok(STANDARD.encode(bytes))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureDevice {
    index: String,
    name: String,
    kind: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureDeviceList {
    video: Vec<CaptureDevice>,
    audio: Vec<CaptureDevice>,
}

#[tauri::command]
fn list_capture_devices() -> Result<CaptureDeviceList, String> {
    let ffmpeg = find_ffmpeg()?;
    let output = Command::new(&ffmpeg)
        .args(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .output()
        .map_err(to_string)?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let mut video = Vec::new();
    let mut audio = Vec::new();
    let mut in_video = false;
    let mut in_audio = false;

    for line in stderr.lines() {
        if line.contains("AVFoundation video devices:") {
            in_video = true;
            in_audio = false;
            continue;
        }
        if line.contains("AVFoundation audio devices:") {
            in_video = false;
            in_audio = true;
            continue;
        }
        if let Some(idx) = extract_device_index(line) {
            let name = line
                .rfind("] ")
                .map(|pos| line[pos + 2..].trim().to_string())
                .unwrap_or_default();
            if in_video {
                let lower = name.to_lowercase();
                let kind = if lower.contains("screen") || lower.contains("capture") {
                    "screen"
                } else {
                    "camera"
                };
                video.push(CaptureDevice { index: idx.clone(), name: name.clone(), kind: kind.to_string() });
            }
            if in_audio {
                audio.push(CaptureDevice { index: idx, name, kind: "audio".to_string() });
            }
        }
    }

    Ok(CaptureDeviceList { video, audio })
}

fn extract_device_index(line: &str) -> Option<String> {
    let mut result = None;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(close) = line[i + 1..].find(']') {
                let content = &line[i + 1..i + 1 + close];
                if !content.is_empty() && content.chars().all(|c| c.is_ascii_digit()) {
                    result = Some(content.to_string());
                }
                i = i + 1 + close + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    result
}

fn detect_devices(ffmpeg: &str) -> (String, Option<String>) {
    let output = Command::new(ffmpeg)
        .args(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .output();

    let stderr = match output {
        Ok(o) => String::from_utf8_lossy(&o.stderr).to_string(),
        Err(_) => return ("0".to_string(), Some("0".to_string())),
    };

    eprintln!("[OpenLoom] avfoundation devices:\n{stderr}");

    let mut screen_index: Option<String> = None;
    let mut mic_index: Option<String> = None;
    let mut mic_fallback: Option<String> = None;
    let mut in_video = false;
    let mut in_audio = false;

    for line in stderr.lines() {
        if line.contains("AVFoundation video devices:") {
            in_video = true;
            in_audio = false;
            continue;
        }
        if line.contains("AVFoundation audio devices:") {
            in_video = false;
            in_audio = true;
            continue;
        }
        if let Some(idx) = extract_device_index(line) {
            if in_video && screen_index.is_none() {
                let lower = line.to_lowercase();
                if lower.contains("screen") || lower.contains("capture") {
                    screen_index = Some(idx.clone());
                }
            }
            if in_audio {
                let lower = line.to_lowercase();
                if mic_index.is_none()
                    && (lower.contains("microphone") || lower.contains("mic"))
                {
                    mic_index = Some(idx.clone());
                }
                if mic_fallback.is_none() {
                    mic_fallback = Some(idx);
                }
            }
        }
    }

    let mic = mic_index.or(mic_fallback);

    eprintln!(
        "[OpenLoom] detected screen={}, mic={:?}",
        screen_index.as_deref().unwrap_or("0"),
        mic
    );

    (
        screen_index.unwrap_or_else(|| "0".to_string()),
        mic,
    )
}

#[tauri::command]
fn start_native_recording(app: AppHandle, with_mic: bool) -> Result<String, String> {
    let mut guard = RECORDER.lock().map_err(to_string)?;
    if guard.is_some() {
        return Err("A recording is already in progress.".into());
    }

    let ffmpeg = find_ffmpeg()?;
    let config = load_config(app.clone()).unwrap_or_default();

    // ── Resolve devices ─────────────────────────────────────────────
    let (auto_screen, auto_mic) = detect_devices(&ffmpeg);
    let screen_idx = config
        .screen_device_index
        .filter(|s| !s.is_empty())
        .unwrap_or(auto_screen);
    let mic_idx = config
        .audio_device_index
        .filter(|s| !s.is_empty())
        .or(auto_mic);
    let use_mic = with_mic && mic_idx.is_some();

    // Webcam
    let webcam_enabled = config.webcam_enabled.unwrap_or(false);
    let webcam_position = config
        .webcam_position
        .as_deref()
        .unwrap_or("bottom-right")
        .to_string();
    let overlay_size = config.webcam_size.unwrap_or(200).clamp(100, 400);

    let webcam_idx: Option<String> = if webcam_enabled {
        let cam = config
            .webcam_device_index
            .filter(|s| !s.is_empty())
            .or_else(|| {
                list_capture_devices().ok().and_then(|d| {
                    d.video
                        .iter()
                        .find(|dev| dev.kind == "camera")
                        .map(|dev| dev.index.clone())
                })
            });
        match cam {
            Some(ref c) if c == &screen_idx => {
                eprintln!("[OpenLoom] WARNING: webcam={c} same as screen — skipping");
                None
            }
            other => other,
        }
    } else {
        None
    };

    // ── Output paths ────────────────────────────────────────────────
    let dir = configured_or_default_dir(&app, config.recording_directory.as_deref(), "recordings")?;
    fs::create_dir_all(&dir).map_err(to_string)?;
    let id = Uuid::new_v4().to_string();
    let final_path = dir.join(format!("{id}.mp4"));
    let screen_path = if webcam_idx.is_some() {
        dir.join(format!("{id}_screen.mp4"))
    } else {
        final_path.clone()
    };
    let webcam_path = webcam_idx.as_ref().map(|_| dir.join(format!("{id}_cam.mp4")));

    // ── Encoder detection ───────────────────────────────────────────
    let hw = Command::new(&ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("h264_videotoolbox"))
        .unwrap_or(false);

    // ── Video quality → scale width ─────────────────────────────────
    let scale_w = match config.video_quality.as_deref().unwrap_or("720") {
        "1440" => "2560".to_string(),
        "1080" => "1920".to_string(),
        "native" => "-1".to_string(),
        _       => "1280".to_string(),
    };

    // ── Combined screen + mic input ─────────────────────────────────
    let input = if use_mic {
        format!("{}:{}", screen_idx, mic_idx.as_ref().unwrap())
    } else {
        format!("{}:none", screen_idx)
    };

    eprintln!(
        "[OpenLoom] input={input}, webcam={:?}, quality={scale_w}, hw={hw}",
        webcam_idx
    );

    // ── Process 1: screen + mic ────────────────────────────────────
    let mut screen_cmd = Command::new(&ffmpeg);
    screen_cmd.args([
        "-use_wallclock_as_timestamps", "1",
        "-thread_queue_size", "1024",
        "-f", "avfoundation",
        "-framerate", "30",
        "-i", &input,
    ]);

    if scale_w != "-1" {
        screen_cmd.args(["-vf", &format!("scale={}:-2", scale_w)]);
    }

    if hw {
        screen_cmd.args(["-c:v", "h264_videotoolbox", "-q:v", "65", "-realtime", "1"]);
    } else {
        screen_cmd.args(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]);
    }
    screen_cmd.args(["-pix_fmt", "yuv420p"]);

    if use_mic {
        screen_cmd.args([
            "-af", "aresample=async=50:min_hard_comp=0.3:first_pts=0",
            "-c:a", "aac", "-b:a", "192k",
        ]);
    }

    screen_cmd.arg("-y").arg(&screen_path);
    screen_cmd.stderr(Stdio::inherit());

    let screen_child = screen_cmd
        .spawn()
        .map_err(|e| format!("Failed to start screen recording: {e}"))?;

    // ── Process 2: webcam (separate process, no audio) ──────────────
    let webcam_child = if let (Some(ref cam_idx), Some(ref wp)) = (&webcam_idx, &webcam_path) {
        let cam_input = format!("{}:none", cam_idx);
        let mut cam_cmd = Command::new(&ffmpeg);
        cam_cmd.args([
            "-use_wallclock_as_timestamps", "1",
            "-thread_queue_size", "1024",
            "-f", "avfoundation",
            "-framerate", "30",
            "-i", &cam_input,
        ]);

        cam_cmd.args([
            "-vf",
            &format!("crop=ih:ih,scale={}:{}", overlay_size, overlay_size),
        ]);

        if hw {
            cam_cmd.args(["-c:v", "h264_videotoolbox", "-q:v", "75", "-realtime", "1"]);
        } else {
            cam_cmd.args(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]);
        }
        cam_cmd.args(["-pix_fmt", "yuv420p", "-an"]);
        cam_cmd.arg("-y").arg(wp);
        cam_cmd.stderr(Stdio::inherit());

        let child = cam_cmd
            .spawn()
            .map_err(|e| format!("Failed to start webcam recording: {e}"))?;
        Some(child)
    } else {
        None
    };

    let state = RecorderState {
        screen_child,
        webcam_child,
        screen_path,
        webcam_path,
        final_path: final_path.clone(),
        started_at: Utc::now(),
        overlay_size,
        webcam_position,
    };
    *guard = Some(state);

    Ok(final_path.to_string_lossy().to_string())
}

/// Stop recording processes and composite webcam overlay if needed.
#[tauri::command]
fn stop_native_recording(_app: AppHandle) -> Result<SavedRecording, String> {
    let mut guard = RECORDER.lock().map_err(to_string)?;
    let mut state = guard.take().ok_or("No recording in progress.")?;

    // ── Stop all processes gracefully ────────────────────────────────
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(state.screen_child.id() as i32, libc::SIGINT);
            if let Some(ref cam) = state.webcam_child {
                libc::kill(cam.id() as i32, libc::SIGINT);
            }
        }
        let _ = state.screen_child.wait();
        if let Some(ref mut cam) = state.webcam_child {
            let _ = cam.wait();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = state.screen_child.kill();
        let _ = state.screen_child.wait();
        if let Some(ref mut cam) = state.webcam_child {
            let _ = cam.kill();
            let _ = cam.wait();
        }
    }

    let duration = (Utc::now() - state.started_at).num_seconds().max(0) as u64;

    // ── Post-processing: webcam overlay only ────────────────────────
    // Only runs when webcam is present. Without webcam, screen_path IS
    // final_path so the recording is already done.
    let has_webcam = state.webcam_path.as_ref().map_or(false, |p| p.exists());
    if has_webcam && state.screen_path.exists() {
        eprintln!("[OpenLoom] compositing webcam overlay");
        let ffmpeg = find_ffmpeg().unwrap_or_else(|_| "ffmpeg".to_string());

        let hw_avail = Command::new(&ffmpeg)
            .args(["-hide_banner", "-encoders"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("h264_videotoolbox"))
            .unwrap_or(false);

        let mut comp = Command::new(&ffmpeg);
        comp.args(["-i", &state.screen_path.to_string_lossy()]);
        if let Some(ref wp) = state.webcam_path {
            comp.args(["-i", &wp.to_string_lossy()]);
        }

        let margin = 20;
        let sz = state.overlay_size;
        let border = 3;
        let total_sz = sz + border * 2; // Total size including border
        let (x, y) = match state.webcam_position.as_str() {
            "top-left"     => (format!("{margin}"), format!("{margin}")),
            "top-right"    => (format!("W-w-{margin}"), format!("{margin}")),
            "bottom-left"  => (format!("{margin}"), format!("H-h-{margin}")),
            _              => (format!("W-w-{margin}"), format!("H-h-{margin}")),
        };
        // Circular webcam pip with white border:
        // 1. Create white circle background (for border)
        // 2. Scale and crop webcam to circle
        // 3. Overlay webcam circle on white background
        // 4. Overlay result on main video
        let r = sz / 2;
        let r_total = total_sz / 2;
        let filter = format!(
            "color=white:{total_sz}x{total_sz},format=rgba,\
             geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':\
             a='if(lte(pow(X-{r_total},2)+pow(Y-{r_total},2),pow({r_total},2)),255,0)'[border];\
             [1:v]scale={sz}:{sz},format=rgba,\
             geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':\
             a='if(lte(pow(X-{r},2)+pow(Y-{r},2),pow({r},2)),255,0)'[cam];\
             [border][cam]overlay={border}:{border}[pip];\
             [0:v][pip]overlay={x}:{y}:eof_action=pass[vout]",
        );
        comp.args(["-filter_complex", &filter, "-map", "[vout]"]);
        if hw_avail {
            comp.args(["-c:v", "h264_videotoolbox", "-q:v", "65"]);
        } else {
            comp.args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]);
        }
        comp.args(["-pix_fmt", "yuv420p"]);

        // Audio: copy as-is, no post-processing
        comp.args(["-map", "0:a?", "-c:a", "copy"]);

        comp.args(["-shortest", "-y"]);
        comp.arg(&state.final_path);
        comp.stderr(Stdio::inherit());

        let result = comp.status();

        match result {
            Ok(s) if s.success() => {
                eprintln!("[OpenLoom] composite done, cleaning up temp files");
                let _ = fs::remove_file(&state.screen_path);
                if let Some(ref wp) = state.webcam_path {
                    let _ = fs::remove_file(wp);
                }
            }
            _ => {
                eprintln!("[OpenLoom] composite failed, using raw screen recording");
                let _ = fs::rename(&state.screen_path, &state.final_path);
                if let Some(ref wp) = state.webcam_path {
                    let _ = fs::remove_file(wp);
                }
            }
        }
    }

    let id = state
        .final_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(SavedRecording {
        id,
        title: format!("Walkthrough {}", state.started_at.format("%Y-%m-%d %H:%M:%S")),
        path: state.final_path.to_string_lossy().to_string(),
        created_at: state.started_at.to_rfc3339(),
        duration_seconds: duration,
    })
}

#[tauri::command]
fn delete_recording(path: String) -> Result<(), String> {
    let file = PathBuf::from(&path);
    if file.exists() {
        fs::remove_file(&file).map_err(to_string)?;
    }
    if let Some(stem) = file.file_stem() {
        if let Some(parent) = file.parent() {
            let stem_str = stem.to_string_lossy();
            let screen_path = parent.join(format!("{}_screen.mp4", stem_str));
            let _ = fs::remove_file(&screen_path);
            let webcam_path = parent.join(format!("{}_cam.mp4", stem_str));
            let _ = fs::remove_file(&webcam_path);
        }
    }
    Ok(())
}

#[tauri::command]
fn load_config(app: AppHandle) -> Result<AppConfig, String> {
    let path = config_path(&app)?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }
    let contents = fs::read_to_string(path).map_err(to_string)?;
    serde_json::from_str(&contents).map_err(to_string)
}

#[tauri::command]
fn save_config(app: AppHandle, config: AppConfig) -> Result<(), String> {
    let path = config_path(&app)?;
    ensure_parent(&path)?;
    let contents = serde_json::to_string_pretty(&config).map_err(to_string)?;
    fs::write(path, contents).map_err(to_string)
}

#[tauri::command]
fn save_recording(app: AppHandle, recording: SaveRecordingRequest) -> Result<SavedRecording, String> {
    let config = load_config(app.clone()).unwrap_or_default();
    let dir = configured_or_default_dir(
        &app,
        config.recording_directory.as_deref(),
        "recordings",
    )?;
    fs::create_dir_all(&dir).map_err(to_string)?;

    let id = Uuid::new_v4().to_string();
    let extension = extension_for_mime(&recording.mime_type);
    let path = dir.join(format!("{id}.{extension}"));
    let bytes = STANDARD.decode(recording.media_base64).map_err(to_string)?;
    fs::write(&path, bytes).map_err(to_string)?;

    Ok(SavedRecording {
        id,
        title: recording.title,
        path: path.to_string_lossy().to_string(),
        created_at: Utc::now().to_rfc3339(),
        duration_seconds: recording.duration_seconds,
    })
}

#[tauri::command]
fn read_file_base64(path: String) -> Result<String, String> {
    let bytes = fs::read(path).map_err(to_string)?;
    Ok(STANDARD.encode(bytes))
}

#[tauri::command]
fn export_guide(app: AppHandle, request: ExportGuideRequest) -> Result<String, String> {
    let config = load_config(app.clone()).unwrap_or_default();
    let root = configured_or_default_dir(&app, config.export_directory.as_deref(), "exports")?;
    let slug = export_slug(&request.walkthrough);
    let export_dir = root.join(format!("{}-{}", Utc::now().format("%Y%m%d-%H%M%S"), slug));
    let assets_dir = export_dir.join("assets");
    fs::create_dir_all(&assets_dir).map_err(to_string)?;

    let source = PathBuf::from(&request.recording_path);
    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("mp4");
    fs::copy(&source, export_dir.join(format!("video.{extension}"))).map_err(to_string)?;
    fs::write(export_dir.join("guide.md"), request.markdown).map_err(to_string)?;
    fs::write(
        export_dir.join("walkthrough.json"),
        serde_json::to_string_pretty(&request.walkthrough).map_err(to_string)?,
    )
    .map_err(to_string)?;
    if let Some(assets) = request.assets {
        for asset in assets {
            let safe_name = sanitize_asset_name(&asset.name);
            let bytes = STANDARD.decode(asset.media_base64).map_err(to_string)?;
            fs::write(assets_dir.join(safe_name), bytes).map_err(to_string)?;
        }
    }

    Ok(export_dir.to_string_lossy().to_string())
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_config_dir()
        .map_err(to_string)?
        .join("config.json"))
}

fn configured_or_default_dir(
    app: &AppHandle,
    configured: Option<&str>,
    child: &str,
) -> Result<PathBuf, String> {
    if let Some(value) = configured {
        if !value.trim().is_empty() {
            return Ok(PathBuf::from(value));
        }
    }

    Ok(app.path().app_data_dir().map_err(to_string)?.join(child))
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(to_string)?;
    }
    Ok(())
}

fn extension_for_mime(mime_type: &str) -> &'static str {
    if mime_type.contains("mp4") {
        "mp4"
    } else {
        "webm"
    }
}

fn export_slug(walkthrough: &serde_json::Value) -> String {
    let title = walkthrough
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("walkthrough");
    let mut slug = title
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').chars().take(48).collect()
}

fn sanitize_asset_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_'))
        .collect::<String>();
    if sanitized.is_empty() {
        "asset.bin".to_string()
    } else {
        sanitized
    }
}

/// Call Whisper transcription via vLLM's /v1/audio/transcriptions endpoint.
/// Accepts base64 audio, decodes it, and sends as multipart form data.
#[tauri::command]
async fn transcribe_audio(
    url: String,
    audio_base64: String,
    filename: String,
    model: String,
    api_token: String,
) -> Result<String, String> {
    use base64::Engine;
    use reqwest::multipart;

    println!("[OpenLoom] transcribe_audio → {url}");
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(&audio_base64)
        .map_err(to_string)?;
    println!("[OpenLoom] audio size: {} bytes, filename: {filename}", audio_bytes.len());

    let file_part = multipart::Part::bytes(audio_bytes)
        .file_name(filename)
        .mime_str("audio/mp4")
        .map_err(to_string)?;

    let form = multipart::Form::new()
        .text("model", model)
        .text("language", "en".to_string())
        .text("response_format", "verbose_json".to_string())
        .text("timestamp_granularities[]", "segment".to_string())
        .part("file", file_part);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(to_string)?;

    let mut req = client.post(&url);
    if !api_token.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_token}"));
    }
    let response = req.multipart(form).send().await.map_err(to_string)?;
    let status = response.status().as_u16();
    let body = response.text().await.map_err(to_string)?;
    println!("[OpenLoom] transcribe_audio ← status={status} body_len={}", body.len());
    if status >= 400 {
        println!("[OpenLoom] transcribe_audio ERROR: {body}");
        return Err(format!("HTTP {status}: {body}"));
    }
    Ok(body)
}

/// Generic HTTP POST that bypasses WKWebView CORS restrictions.
/// Returns the response body as a string. All external API calls go through here.
#[tauri::command]
async fn http_post(
    url: String,
    body: String,
    headers: std::collections::HashMap<String, String>,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(to_string)?;

    println!("[OpenLoom] http_post → {url}");
    println!("[OpenLoom] http_post body length: {} bytes", body.len());

    let mut req = client.post(&url).header("Content-Type", "application/json");
    for (key, value) in &headers {
        req = req.header(key.as_str(), value.as_str());
    }
    let response = req.body(body).send().await.map_err(to_string)?;
    let status = response.status().as_u16();
    let text = response.text().await.map_err(to_string)?;
    println!("[OpenLoom] http_post ← {url} status={status} body_len={}", text.len());
    if status >= 400 {
        println!("[OpenLoom] http_post ERROR: {text}");
        return Err(format!("HTTP {status}: {text}"));
    }
    Ok(text)
}

/// Call OpenAI API for tutorial synthesis.
/// Uses gpt-5.4-mini with reasoning_effort=medium (temperature must be default=1 for this model).
/// Accepts an optional system prompt for better context.
#[tauri::command]
async fn call_openai(
    api_key: String,
    prompt: String,
    system_prompt: Option<String>,
) -> Result<String, String> {
    let url = "https://api.openai.com/v1/chat/completions";

    // Build messages array with optional system prompt
    let mut messages = Vec::new();
    if let Some(sys) = &system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": sys
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": prompt
    }));

    let payload = serde_json::json!({
        "model": "gpt-5.4-mini",
        "messages": messages,
        "reasoning_effort": "medium"
    });

    println!(
        "[OpenLoom] call_openai → gpt-5.4-mini (system prompt: {})",
        if system_prompt.is_some() { "yes" } else { "no" }
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(to_string)?;

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload)
        .send()
        .await
        .map_err(to_string)?;

    let status = response.status().as_u16();
    if status >= 400 {
        let body = response.text().await.unwrap_or_default();
        println!("[OpenLoom] call_openai ERROR: {status} {body}");
        return Err(format!("OpenAI API returned {status}: {body}"));
    }

    let body: serde_json::Value = response.json().await.map_err(to_string)?;
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    println!("[OpenLoom] call_openai ← {} chars", content.len());

    if content.is_empty() {
        Err("OpenAI returned empty response".to_string())
    } else {
        Ok(content)
    }
}

/// Call OpenAI Vision API (gpt-5.x models) for frame analysis.
/// Different API format than vLLM - uses OpenAI's native vision endpoint.
#[tauri::command]
async fn call_openai_vision(
    api_key: String,
    model: String,
    system_prompt: String,
    frames_base64: Vec<String>,
    user_prompt: String,
) -> Result<String, String> {
    let url = "https://api.openai.com/v1/chat/completions";

    // Build content array with images and text
    let mut content: Vec<serde_json::Value> = frames_base64
        .iter()
        .map(|b64| {
            serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:image/jpeg;base64,{}", b64),
                    "detail": "low"
                }
            })
        })
        .collect();

    content.push(serde_json::json!({
        "type": "text",
        "text": user_prompt
    }));

    let payload = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": content }
        ],
        "max_tokens": 2048
    });

    println!(
        "[OpenLoom] call_openai_vision → {} ({} frames)",
        model,
        frames_base64.len()
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(to_string)?;

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload)
        .send()
        .await
        .map_err(to_string)?;

    let status = response.status().as_u16();
    if status >= 400 {
        let body = response.text().await.unwrap_or_default();
        println!("[OpenLoom] call_openai_vision ERROR: {status} {body}");
        return Err(format!("OpenAI Vision API error {}: {}", status, body));
    }

    let body: serde_json::Value = response.json().await.map_err(to_string)?;
    let result = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    println!("[OpenLoom] call_openai_vision ← {} chars", result.len());

    Ok(result)
}

/// Call OpenAI transcription API (gpt-4o-transcribe, gpt-4o-mini-transcribe).
/// These have a 25MB file limit and 1500s duration limit.
#[tauri::command]
async fn transcribe_openai(
    api_key: String,
    audio_base64: String,
    filename: String,
    model: String,
) -> Result<String, String> {
    let url = "https://api.openai.com/v1/audio/transcriptions";

    let audio_bytes = STANDARD.decode(&audio_base64).map_err(to_string)?;

    // Check file size limit (25MB)
    let size_mb = audio_bytes.len() as f64 / (1024.0 * 1024.0);
    if audio_bytes.len() > 25 * 1024 * 1024 {
        return Err(format!(
            "Audio file too large: {:.1} MB (max 25 MB). Use split_audio_at_silence first.",
            size_mb
        ));
    }

    println!(
        "[OpenLoom] transcribe_openai → {} ({:.1} MB, {})",
        model, size_mb, filename
    );

    let file_part = reqwest::multipart::Part::bytes(audio_bytes)
        .file_name(filename)
        .mime_str("audio/mp4")
        .map_err(to_string)?;

    let form = reqwest::multipart::Form::new()
        .text("model", model)
        .text("language", "en")
        .text("response_format", "verbose_json")
        .text("timestamp_granularities[]", "segment")
        .part("file", file_part);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(to_string)?;

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .map_err(to_string)?;

    let status = response.status().as_u16();
    let body = response.text().await.map_err(to_string)?;

    if status >= 400 {
        println!("[OpenLoom] transcribe_openai ERROR: {status} {body}");
        return Err(format!("OpenAI Transcription error {}: {}", status, body));
    }

    println!("[OpenLoom] transcribe_openai ← {} chars", body.len());

    Ok(body)
}

/// Extract audio track from video file as MP3.
/// Used when we need to split audio for OpenAI's 25MB limit.
#[tauri::command]
fn extract_audio(video_path: String) -> Result<String, String> {
    let ffmpeg = find_ffmpeg()?;
    let video = PathBuf::from(&video_path);

    if !video.exists() {
        return Err(format!("Video not found: {}", video_path));
    }

    let audio_path = video.with_extension("mp3");

    println!("[OpenLoom] extract_audio: {} → {}", video_path, audio_path.display());

    let status = Command::new(&ffmpeg)
        .args([
            "-i",
            &video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            "-y",
        ])
        .arg(&audio_path)
        .stderr(Stdio::null())
        .status()
        .map_err(to_string)?;

    if !status.success() {
        return Err("Failed to extract audio from video".to_string());
    }

    Ok(audio_path.to_string_lossy().to_string())
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SilenceSegment {
    start: f64,
    end: f64,
}

/// Detect silence segments in audio file using ffmpeg silencedetect.
/// Returns list of silence start/end points that can be used for splitting.
#[tauri::command]
fn detect_silence(
    audio_path: String,
    noise_db: Option<i32>,
    min_duration: Option<f64>,
) -> Result<Vec<SilenceSegment>, String> {
    let ffmpeg = find_ffmpeg()?;
    let noise = noise_db.unwrap_or(-40);
    let duration = min_duration.unwrap_or(1.0);

    println!(
        "[OpenLoom] detect_silence: {} (noise={}dB, min_duration={}s)",
        audio_path, noise, duration
    );

    let output = Command::new(&ffmpeg)
        .args([
            "-i",
            &audio_path,
            "-af",
            &format!("silencedetect=n={}dB:d={}", noise, duration),
            "-f",
            "null",
            "-",
        ])
        .stderr(Stdio::piped())
        .output()
        .map_err(to_string)?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut segments = Vec::new();
    let mut current_start: Option<f64> = None;

    for line in stderr.lines() {
        if line.contains("silence_start:") {
            if let Some(pos) = line.find("silence_start:") {
                let rest = &line[pos + 14..];
                if let Some(val) = rest.split_whitespace().next() {
                    current_start = val.parse().ok();
                }
            }
        } else if line.contains("silence_end:") {
            if let Some(pos) = line.find("silence_end:") {
                let rest = &line[pos + 12..];
                if let Some(val) = rest.split_whitespace().next() {
                    if let (Some(start), Ok(end)) = (current_start, val.parse::<f64>()) {
                        segments.push(SilenceSegment { start, end });
                        current_start = None;
                    }
                }
            }
        }
    }

    println!("[OpenLoom] detect_silence ← {} segments", segments.len());

    Ok(segments)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioChunk {
    path: String,
    start_time: f64,
    end_time: f64,
    size_bytes: u64,
}

/// Split audio file at specified timestamps.
/// Returns paths to chunk files with their time ranges.
#[tauri::command]
fn split_audio_at_times(
    audio_path: String,
    split_times: Vec<f64>,
) -> Result<Vec<AudioChunk>, String> {
    let ffmpeg = find_ffmpeg()?;
    let audio = PathBuf::from(&audio_path);

    if !audio.exists() {
        return Err(format!("Audio file not found: {}", audio_path));
    }

    // Get total duration using ffprobe-style extraction
    let probe = Command::new(&ffmpeg)
        .args(["-i", &audio_path, "-f", "null", "-"])
        .stderr(Stdio::piped())
        .output()
        .map_err(to_string)?;

    let stderr = String::from_utf8_lossy(&probe.stderr);
    let total_duration: f64 = stderr
        .lines()
        .find_map(|line| {
            if let Some(pos) = line.find("Duration:") {
                let s = &line[pos + 10..];
                let time_str = s.split(',').next()?.trim();
                let parts: Vec<&str> = time_str.split(':').collect();
                if parts.len() == 3 {
                    let h: f64 = parts[0].parse().ok()?;
                    let m: f64 = parts[1].parse().ok()?;
                    let s: f64 = parts[2].parse().ok()?;
                    return Some(h * 3600.0 + m * 60.0 + s);
                }
            }
            None
        })
        .unwrap_or(0.0);

    println!(
        "[OpenLoom] split_audio_at_times: {} split points, duration={:.1}s",
        split_times.len(),
        total_duration
    );

    // Build time ranges: [0, t1], [t1, t2], ..., [tn, end]
    let mut times = vec![0.0];
    times.extend(
        split_times
            .iter()
            .filter(|&&t| t > 0.0 && t < total_duration),
    );
    times.push(total_duration);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup();

    let chunks_dir = audio
        .parent()
        .unwrap_or(Path::new("."))
        .join("audio_chunks");
    let _ = fs::remove_dir_all(&chunks_dir);
    fs::create_dir_all(&chunks_dir).map_err(to_string)?;

    let mut chunks = Vec::new();
    let stem = audio.file_stem().unwrap_or_default().to_string_lossy();

    for i in 0..times.len() - 1 {
        let start = times[i];
        let end = times[i + 1];
        let chunk_path = chunks_dir.join(format!("{}_chunk_{:03}.mp3", stem, i));

        let status = Command::new(&ffmpeg)
            .args([
                "-i",
                &audio_path,
                "-ss",
                &format!("{:.3}", start),
                "-to",
                &format!("{:.3}", end),
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                "-y",
            ])
            .arg(&chunk_path)
            .stderr(Stdio::null())
            .status()
            .map_err(to_string)?;

        if status.success() && chunk_path.exists() {
            let metadata = fs::metadata(&chunk_path).map_err(to_string)?;
            chunks.push(AudioChunk {
                path: chunk_path.to_string_lossy().to_string(),
                start_time: start,
                end_time: end,
                size_bytes: metadata.len(),
            });
        }
    }

    println!("[OpenLoom] split_audio_at_times ← {} chunks", chunks.len());

    Ok(chunks)
}

/// Get file size in bytes.
#[tauri::command]
fn get_file_size(path: String) -> Result<u64, String> {
    let metadata = fs::metadata(&path).map_err(to_string)?;
    Ok(metadata.len())
}

fn to_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            check_ffmpeg,
            extract_frames,
            grab_screenshot,
            list_capture_devices,
            start_native_recording,
            stop_native_recording,
            delete_recording,
            load_config,
            save_config,
            save_recording,
            read_file_base64,
            export_guide,
            transcribe_audio,
            http_post,
            call_openai,
            // OpenAI integration
            call_openai_vision,
            transcribe_openai,
            extract_audio,
            detect_silence,
            split_audio_at_times,
            get_file_size
        ])
        .run(tauri::generate_context!())
        .expect("error while running OpenLoom");
}
