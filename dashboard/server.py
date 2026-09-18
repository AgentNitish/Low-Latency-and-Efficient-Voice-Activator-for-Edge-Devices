#!/usr/bin/env python3
"""
==============================================================================================
ESP32-S3 TinyML Voice Activator & Audio Streaming Node - Unified Server & Live Web Dashboard
==============================================================================================
Features:
  1. Web Dashboard (HTTP on http://localhost:8000):
     - Glassmorphism dark UI with real-time VU meter & dynamic audio visualizer.
     - Live transcription & status indicator (IDLE, LISTENING, STREAMING).
     - In-browser WAV player and download button for recent recordings.
     - Interactive Web Buttons: [ABORT TRANSMISSION] and [RESTART TRANSMISSION].
  2. WebSocket Endpoint for ESP32 (/ws/esp32):
     - High-speed 16 kHz 16-bit mono PCM stream ingestion.
     - Ultra-low latency processing with automatic 0.8s silence detection.
     - Immediate WAV generation and optional Whisper speech-to-text.
==============================================================================================
"""

import argparse
import asyncio
import datetime
import io
import json
import math
import os
import sys
import time
import wave
from typing import Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# Audio & Buffer Constants
SAMPLE_RATE = 16000      # 16 kHz
SAMPLE_WIDTH = 2         # 16-bit PCM = 2 bytes per sample
CHANNELS = 1             # Mono
DEFAULT_AUTO_STOP = 30.0 # Max streaming duration in seconds
DEFAULT_SILENCE = 1.5    # Stop when silence persists for 1.5s
SILENCE_RMS_THRESHOLD = 3000 # Increased to 3000 to ignore loud fan noise

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# State Management
app = FastAPI(title="ESP32 TinyML Voice Activator Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

esp_clients: Set[WebSocket] = set()
dashboard_clients: Set[WebSocket] = set()

if HAS_WHISPER:
    print("[INIT] Loading Whisper model (base.en)...")
    whisper_model = whisper.load_model("base.en")
    print("[INIT] Whisper model loaded successfully.")
else:
    whisper_model = None

# In-memory cyclic buffer (last 60s)
CYCLIC_BUFFER_MAX = 60 * SAMPLE_RATE * SAMPLE_WIDTH
cyclic_buffer = bytearray()
cyclic_lock = asyncio.Lock()

# Current Active Streaming Session
active_session = {
    "is_streaming": False,
    "buffer": bytearray(),
    "start_time": 0.0,
    "stop_sent": False,
    "client_ip": "None",
    "last_wav_path": None,
    "recent_sessions": [],
    "last_partial_size": 0,
    "last_partial_text": "",
    "last_text_change_time": 0.0,
    "transcribing": False
}


def get_local_ip():
    """Find local IPv4 address for ESP32 sketch configuration."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def save_pcm_to_wav(pcm_bytes: bytes, filename: str) -> str:
    """Save raw 16-bit mono 16 kHz PCM to a standard WAV file."""
    filepath = os.path.join(RECORDINGS_DIR, filename)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return filepath


async def broadcast_dashboard(event: dict):
    """Broadcast real-time state, VU levels, and events to all connected browser tabs."""
    if not dashboard_clients:
        return
    dead = set()
    for ws in list(dashboard_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    dashboard_clients.difference_update(dead)


async def send_command_to_esp(cmd_name: str, payload_dict: dict):
    """Send command ('stop', 'start', 'abort') to all connected ESP32 microcontrollers."""
    if not esp_clients:
        return False
    msg = json.dumps(payload_dict)
    dead = set()
    for ws in list(esp_clients):
        try:
            await ws.send_text(msg)
            if cmd_name != "start":
                print(f"[SERVER -> ESP32] >>> Sent '{cmd_name}' command: {msg} <<<")
        except Exception:
            dead.add(ws)
    esp_clients.difference_update(dead)
    return True


async def do_partial_transcription(pcm_data: bytes):
    """Run Whisper partial transcription in a separate thread so it doesn't block the WebSocket loop."""
    try:
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(
            whisper_model.transcribe, 
            audio_np, 
            language="en", 
            fp16=False,
            temperature=0,
            condition_on_previous_text=False
        )
        text = result.get("text", "").strip()
        lower_text = text.lower()
        if lower_text in ["[blank_audio]", "(silence)", "you", ".", ""]:
            text = ""

        if text != active_session.get("last_partial_text", ""):
            active_session["last_partial_text"] = text
            active_session["last_text_change_time"] = time.time()
            if text:
                await broadcast_dashboard({"event": "partial_text", "text": text})
    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
    finally:
        active_session["transcribing"] = False


async def finalize_recording(termination_reason: str):
    """Save recorded stream, run Whisper, update dashboard, and notify ESP32."""
    if len(active_session["buffer"]) == 0:
        active_session["is_streaming"] = False
        return

    pcm_data = bytes(active_session["buffer"])
    active_session["buffer"].clear()
    active_session["is_streaming"] = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    time_display = datetime.datetime.now().strftime("%H:%M:%S")
    wav_name = f"audio_{timestamp}.wav"
    filepath = save_pcm_to_wav(pcm_data, wav_name)
    duration_sec = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH)
    active_session["last_wav_path"] = filepath

    print(f"\n=======================================================")
    print(f"[AUDIO SAVED] File: {filepath}")
    print(f"              Size: {len(pcm_data):,} bytes ({duration_sec:.2f}s)")
    print(f"              Reason: {termination_reason}")
    print(f"=======================================================")

    transcription = ""
    if HAS_WHISPER and whisper_model is not None:
        try:
            print("[WHISPER] Transcribing audio...")
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = whisper_model.transcribe(audio_np, language="en", fp16=False)
            transcription = result.get("text", "").strip()
            print(f"[TRANSCRIPTION] \"{transcription}\"\n")
        except Exception as e:
            print(f"[WHISPER ERROR] {e}")

    session_entry = {
        "time": time_display,
        "filename": wav_name,
        "duration": f"{duration_sec:.2f}s",
        "reason": termination_reason,
        "text": transcription or "Audio Captured",
        "size_kb": round(len(pcm_data) / 1024, 1)
    }
    active_session["recent_sessions"].insert(0, session_entry)
    if len(active_session["recent_sessions"]) > 15:
        active_session["recent_sessions"].pop()

    # Inform Dashboard
    await broadcast_dashboard({
        "event": "session_ended",
        "session": session_entry,
        "state": "IDLE"
    })


# ==============================================================================================
# REST API ENDPOINTS
# ==============================================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the sleek Glassmorphism Live Web Dashboard directly to browsers."""
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/api/audio/recent_wav")
async def get_recent_wav():
    """Stream the most recently recorded WAV or recent cyclic buffer."""
    if active_session["last_wav_path"] and os.path.exists(active_session["last_wav_path"]):
        return FileResponse(active_session["last_wav_path"], media_type="audio/wav")

    async with cyclic_lock:
        data = bytes(cyclic_buffer)

    if not data:
        data = bytes(SAMPLE_RATE * SAMPLE_WIDTH)  # 1 sec silence fallback

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data)

    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/api/control/abort")
async def api_abort():
    """Web Button: Abort transmission."""
    await send_command_to_esp("abort", {"command": "abort", "status": "ABORT"})
    await broadcast_dashboard({"event": "state_changed", "state": "PROCESSING"})
    await finalize_recording("OPERATOR_ABORT")
    return {"status": "ABORT_SENT"}


@app.post("/api/control/restart")
async def api_restart():
    """Web Button: Restart transmission (Stealth)."""
    # Silently send start. The ESP32 will fake a trigger.
    # We update the server state instantly so it works even if ESP32 hasn't been flashed yet.
    active_session["buffer"].clear()
    active_session["is_streaming"] = True
    active_session["stop_sent"] = False
    active_session["start_time"] = time.time()
    active_session["last_text_change_time"] = time.time()
    active_session["last_partial_size"] = 0
    active_session["last_partial_text"] = ""
    active_session["transcribing"] = False
    await send_command_to_esp("start", {"command": "start", "status": "START"})
    await broadcast_dashboard({"event": "state_changed", "state": "STREAMING"})
    return {"status": "START_SENT"}


# ==============================================================================================
# WEBSOCKET ENDPOINTS
# ==============================================================================================

@app.websocket("/ws/dashboard")
async def ws_dashboard_endpoint(websocket: WebSocket):
    """Browser WebSocket for live audio visualizer, VU meter, and status sync."""
    await websocket.accept()
    dashboard_clients.add(websocket)
    try:
        # Send initial state
        state_str = "STREAMING" if active_session["is_streaming"] else "LISTENING" if esp_clients else "IDLE"
        await websocket.send_json({
            "event": "init",
            "state": state_str,
            "connected_nodes": len(esp_clients),
            "recent_sessions": active_session["recent_sessions"]
        })
        while True:
            # Keep browser socket open
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        dashboard_clients.discard(websocket)


@app.websocket("/ws/esp32")
async def ws_esp32_endpoint(websocket: WebSocket):
    """ESP32 WebSocket: Receives 16 kHz 16-bit PCM binary audio stream."""
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "ESP32_Node"
    esp_clients.add(websocket)
    print(f"\n[NODE CONNECTED] ESP32 node connected from {client_ip}")

    await broadcast_dashboard({"event": "state_changed", "state": "LISTENING", "node_ip": client_ip})

    last_console_time = 0
    packet_count = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # 1. BINARY PCM AUDIO FRAME FROM ESP32
            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                now = time.time()

                async with cyclic_lock:
                    cyclic_buffer.extend(chunk)
                    if len(cyclic_buffer) > CYCLIC_BUFFER_MAX:
                        del cyclic_buffer[:-CYCLIC_BUFFER_MAX]

                if active_session["is_streaming"]:
                    active_session["buffer"].extend(chunk)
                    packet_count += 1

                # Calculate RMS & Peak
                samples = np.frombuffer(chunk, dtype=np.int16)
                if len(samples) > 0:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    peak = int(np.max(np.abs(samples)))
                else:
                    rms, peak = 0.0, 0

                duration_sec = len(active_session["buffer"]) / (SAMPLE_RATE * SAMPLE_WIDTH)

                # Broadcast VU audio meter to browser UI
                await broadcast_dashboard({
                    "event": "audio_meter",
                    "rms": round(rms, 1),
                    "peak": peak,
                    "duration": round(duration_sec, 2)
                })

                if active_session["is_streaming"]:
                    # Throttled console progress (every 100ms)
                    if now - last_console_time >= 0.1:
                        last_console_time = now
                        bars = min(10, int((rms / 3000.0) * 10))
                        vu = "[" + "#" * bars + "-" * (10 - bars) + "]"
                        sys.stdout.write(
                            f"\r[STREAMING] Rec: {duration_sec:4.1f}s | RMS: {rms:5.1f} | Peak: {peak:5d} {vu} "
                            f"(Press Enter to STOP)"
                        )
                        sys.stdout.flush()

                    # Partial transcription every 16,000 bytes (~0.5s)
                    buf_len = len(active_session["buffer"])
                    if HAS_WHISPER and whisper_model is not None:
                        if buf_len - active_session.get("last_partial_size", 0) >= 16000:
                            if not active_session.get("transcribing", False):
                                active_session["transcribing"] = True
                                active_session["last_partial_size"] = buf_len
                                asyncio.create_task(do_partial_transcription(bytes(active_session["buffer"])))

                    # A. Automatic End-of-Speech / NLP Silence Detection (3s trailing, 10s initial)
                    if duration_sec >= 3.0 and not active_session["stop_sent"]:
                        time_since_last_text = now - active_session["last_text_change_time"]
                        has_spoken = bool(active_session.get("last_partial_text", "").strip())
                        timeout_limit = 2.0 if has_spoken else 10.0
                        
                        if time_since_last_text >= timeout_limit:
                            active_session["stop_sent"] = True
                            print(f"\n[SPEECH END] NLP Silence detected ({timeout_limit}s without text change). Finalizing...")
                            await send_command_to_esp("stop", {"command": "stop", "status": "STOP"})
                            await broadcast_dashboard({"event": "state_changed", "state": "PROCESSING"})
                            await finalize_recording("NLP_SILENCE_DETECTED")

                    # B. Auto-Stop Duration Limit
                    if duration_sec >= DEFAULT_AUTO_STOP and not active_session["stop_sent"]:
                        active_session["stop_sent"] = True
                        print(f"\n[AUTO-STOP] Reached duration limit ({DEFAULT_AUTO_STOP}s). Finalizing...")
                        await send_command_to_esp("stop", {"command": "stop", "status": "STOP"})
                        await broadcast_dashboard({"event": "state_changed", "state": "PROCESSING"})
                        await finalize_recording("DURATION_LIMIT")

            # 2. TEXT / JSON EVENT FROM ESP32
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    event = data.get("event", "MSG")
                    print(f"\n[ESP32 EVENT] {client_ip} -> {event}: {data}")

                    if event == "ACTIVATED":
                        print(">>> [TINYML TRIGGER] ESP32 reported wake-word trigger! <<<")
                        active_session["buffer"].clear()
                        active_session["is_streaming"] = True
                        active_session["stop_sent"] = False
                        active_session["start_time"] = time.time()
                        active_session["last_text_change_time"] = time.time()
                        active_session["client_ip"] = client_ip
                        active_session["last_partial_size"] = 0
                        active_session["last_partial_text"] = ""
                        active_session["transcribing"] = False
                        await broadcast_dashboard({"event": "state_changed", "state": "STREAMING", "trigger": data})

                    elif event == "HEARTBEAT":
                        # Periodic status from ESP32 tasks — print with timestamp
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        task  = data.get("task", "?")
                        msg   = data.get("msg", "")
                        extra = {k: v for k, v in data.items() if k not in ("event", "task", "msg")}
                        extra_str = " | ".join(f"{k}:{v}" for k, v in extra.items())
                        print(f"[{ts}][{task}] {msg}" + (f" | {extra_str}" if extra_str else ""))

                    elif event == "DEBUG":
                        # One-off debug messages from any ESP32 task
                        ts   = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        task = data.get("task", "ESP32")
                        msg  = data.get("msg", str(data))
                        lvl  = data.get("level", "INFO")
                        prefix = {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️", "OK": "✅"}.get(lvl, "  ")
                        print(f"[{ts}][{task}] {prefix} {msg}")

                except Exception:
                    pass

    except (WebSocketDisconnect, Exception) as e:
        print(f"\n[NODE DISCONNECT] ESP32 disconnected: {e}")
    finally:
        esp_clients.discard(websocket)
        if active_session["is_streaming"] and len(active_session["buffer"]) > 0:
            await finalize_recording("NODE_DISCONNECT")
        await broadcast_dashboard({"event": "state_changed", "state": "IDLE"})


async def console_input_loop():
    """Console listener allowing terminal operator to press Enter or type commands."""
    if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
        return
    loop = asyncio.get_event_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            cmd = line.strip().lower()
            if cmd in ("", "stop", "s", "abort", "a"):
                print("\n[OPERATOR] Broadcasting STOP/ABORT to ESP32...")
                await send_command_to_esp("abort", {"command": "abort", "status": "ABORT"})
                await broadcast_dashboard({"event": "state_changed", "state": "PROCESSING"})
                await finalize_recording("OPERATOR_TERMINAL_STOP")
            elif cmd in ("start", "restart", "r"):
                # Stealth operator start
                active_session["buffer"].clear()
                active_session["is_streaming"] = True
                active_session["stop_sent"] = False
                await send_command_to_esp("start", {"command": "start", "status": "START"})
            elif cmd in ("quit", "exit", "q"):
                print("\n[SHUTDOWN] Terminating server...")
                os._exit(0)
        except Exception:
            break


# ==============================================================================================
# EMBEDDED GLASSMORPHISM WEB DASHBOARD (HTML5 + CSS3 + Vanilla JS)
# ==============================================================================================
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GItTub Live Dashboard</title>
<style>
* { box-sizing: border-box; }

:root {
  --bg: #071423;
  --panel: #0d2137;
  --panel-2: #102943;
  --border: rgba(118, 163, 211, .18);
  --text: #eef4fb;
  --muted: #8da0b5;
  --blue: #64a9ff;
  --red: #ff6273;
  --green: #2abf91;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: radial-gradient(circle at 50% -20%, #142e4b 0, var(--bg) 42%);
  color: var(--text);
}

.app-shell { width: min(1120px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 44px; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 22px;
}
.eyebrow { color: var(--blue); letter-spacing: .16em; font-size: 10px; font-weight: 800; margin: 0 0 3px; }
h1 { font-size: 24px; margin: 0; letter-spacing: -.02em; }
h2 { margin: 0 0 13px; font-size: 10px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }

.badge {
  display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border);
  border-radius: 999px; padding: 8px 12px; font-size: 11px; font-weight: 800; letter-spacing: .05em;
}
.badge span { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.badge.streaming span { background: var(--green); box-shadow: 0 0 14px var(--green); }
.badge.listening span { background: var(--blue); }
.badge.rejected span { background: var(--red); }

.dashboard {
  display: grid;
  grid-template-columns: 1.65fr 1fr;
  gap: 12px;
}
.panel {
  background: linear-gradient(145deg, rgba(17, 43, 70, .96), rgba(9, 26, 44, .98));
  border: 1px solid var(--border);
  border-radius: 11px;
  box-shadow: 0 16px 45px rgba(0,0,0,.18);
}
.live-panel, .metrics-panel { padding: 18px; min-height: 108px; }
#live-text { font-size: 18px; font-weight: 500; min-height: 42px; }
.metric-line { font-size: 13px; line-height: 1.9; color: var(--muted); }
.metric-line strong { color: var(--text); font-weight: 600; }

.audio-panel {
  grid-column: 1 / -1;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.panel-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.panel-header-row h2 { margin: 0; }
.diag-tag {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 6px;
  background: rgba(100, 169, 255, 0.15);
  color: var(--blue);
  border: 1px solid rgba(100, 169, 255, 0.3);
}
.diag-metrics {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
}
.diag-col {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.diag-label {
  font-size: 11px;
  color: var(--muted);
}
.diag-col strong {
  font-size: 14px;
  color: var(--text);
  font-family: monospace;
}
.vu-wrap {
  width: 100%;
}
.vu-track {
  height: 8px;
  background: rgba(8, 22, 38, 0.85);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}
.vu-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #2abf91 0%, #64a9ff 65%, #ff6273 95%);
  transition: width 0.08s ease-out;
  border-radius: 3px;
}
.audio-controls-row {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 4px;
}
audio {
  flex: 1;
  min-width: 260px;
  height: 38px;
  outline: none;
  filter: invert(0.88) hue-rotate(180deg);
}
.audio-btn-group {
  display: flex;
  gap: 10px;
  align-items: center;
}
.action-btn {
  background: rgba(100, 169, 255, 0.14);
  color: var(--text);
  border: 1px solid rgba(100, 169, 255, 0.35);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: 0.15s ease;
}
.action-btn:hover {
  background: rgba(100, 169, 255, 0.28);
  border-color: var(--blue);
}
.download-btn {
  background: rgba(42, 191, 145, 0.14);
  border-color: rgba(42, 191, 145, 0.35);
  color: #a8edd5;
}
.download-btn:hover {
  background: rgba(42, 191, 145, 0.28);
  border-color: var(--green);
}

.sessions-panel { grid-column: 1 / -1; padding: 16px; }

.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 11px; min-width: 780px; }
th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid rgba(118,163,211,.12); }
th { color: var(--muted); font-weight: 500; }
td { color: #d7e2ef; }

.overlay {
  position: fixed; inset: 0; display: grid; place-items: center; padding: 24px;
  background: rgba(2, 10, 20, .70); backdrop-filter: blur(7px);
  opacity: 0; visibility: hidden; transition: .22s ease; z-index: 10;
}
.overlay.show { opacity: 1; visibility: visible; }

.live-modal {
  width: min(570px, 100%); min-height: 330px; padding: 24px 28px;
  border: 1px solid rgba(91, 150, 220, .42); border-radius: 16px;
  background: linear-gradient(145deg, #0c2036, #071525);
  box-shadow: 0 28px 90px rgba(0,0,0,.5);
  display: flex; flex-direction: column; align-items: center;
}
.modal-head { width: 100%; display: flex; justify-content: space-between; align-items: center; }
.live-label { color: #ff9ba5; font-size: 13px; font-weight: 800; letter-spacing: .05em; }
.live-label i { display: inline-block; width: 10px; height: 10px; background: var(--red); border-radius: 50%; margin-right: 8px; box-shadow: 0 0 12px var(--red); }
#close-live { border: 0; background: transparent; color: #aebdd0; font-size: 30px; cursor: pointer; line-height: 1; }

.popup-text {
  flex: 1; width: 100%; display: grid; place-items: center;
  text-align: center; font-size: clamp(32px, 5vw, 54px); font-weight: 750; line-height: 1.08;
  background: linear-gradient(90deg, #f4f6f9 10%, #62a9ff 95%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

.wave { height: 40px; display: flex; align-items: center; gap: 4px; }
.wave span { width: 3px; height: 8px; border-radius: 5px; background: var(--blue); animation: pulse 1s ease-in-out infinite alternate; }
.wave span:nth-child(2n) { animation-delay: .2s; height: 22px; }
.wave span:nth-child(3n) { animation-delay: .45s; height: 30px; }
.wave span:nth-child(5n) { animation-delay: .65s; height: 14px; }
@keyframes pulse { from { transform: scaleY(.45); opacity: .5; } to { transform: scaleY(1.15); opacity: 1; } }
.listening { color: var(--muted); font-size: 13px; margin-top: 9px; }

.toast {
  position: fixed; left: 50%; bottom: 26px; transform: translate(-50%, 20px);
  opacity: 0; pointer-events: none; display: flex; gap: 9px; align-items: center;
  padding: 12px 18px; border: 1px solid rgba(42,191,145,.35); border-radius: 8px;
  background: #083e35; color: #baf5df; font-size: 13px; transition: .25s ease; z-index: 20;
}
.toast.show { opacity: 1; transform: translate(-50%, 0); }
.toast span:first-child { color: #61dfb4; font-weight: 900; }

@media (max-width: 720px) {
  .app-shell { width: min(100% - 24px, 1120px); padding-top: 20px; }
  .dashboard { grid-template-columns: 1fr; }
  .sessions-panel { grid-column: auto; }
  h1 { font-size: 19px; }
  .live-modal { min-height: 300px; }
}

/* Custom Visualizer added */
.wave-container { display: flex; align-items: flex-end; justify-content: center; gap: 4px; height: 50px; margin: 1.5rem 0; }
.wave-bar { width: 6px; height: 6px; background: #334155; border-radius: 9999px; transition: height 0.08s ease; }
.active-wave { background: var(--green); box-shadow: 0 0 8px rgba(42, 191, 145, 0.5); }
.btn-red { background: rgba(239, 68, 68, 0.15); border-color: var(--red); color: #fca5a5; }
.btn-red:hover { background: var(--red); color: #fff; }
</style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">GITTUB</p>
        <h1>Live Voice Dashboard</h1>
      </div>
      <div id="state-badge" class="badge idle"><span></span>IDLE</div>
    </header>

    <main class="dashboard">
      <section class="panel live-panel">
        <h2>Live Transcription</h2>
        <div id="live-text">Waiting for a session…</div>
      </section>

      <section class="panel metrics-panel">
        <h2>Metrics</h2>
        <div class="metric-line">Connected Node: <strong id="nodeIp">Waiting...</strong></div>
        <div class="metric-line">VAD Mode: <strong>NLP (Whisper)</strong></div>
      </section>

      <section class="panel audio-panel">
        <div class="panel-header-row">
          <h2>🎙️ Microphone Diagnostics & Controls</h2>
          <span id="mic-status-tag" class="diag-tag">Ready</span>
        </div>
        
        <div class="diag-metrics" style="margin-top: 8px;">
          <div class="diag-col">
            <div class="diag-label">Signal Level (RMS / Peak):</div>
            <strong><span id="mic-rms">0</span> / <span id="mic-peak">0</span></strong>
          </div>
        </div>

        <div class="vu-wrap" title="Microphone Activity Meter" style="margin-top: 10px;">
          <div class="vu-track">
            <div id="vu-bar" class="vu-fill"></div>
          </div>
        </div>
        
        <div class="wave-container" id="waveContainer">
          <!-- 16 Animated Equalizer Bars -->
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div>
        </div>

        <div class="audio-controls-row">
          <audio id="audio-player" controls preload="none">
            <source id="audio-source" src="/api/audio/recent_wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
          <div class="audio-btn-group">
            <button class="action-btn btn-red" onclick="fetch('/api/control/abort', {method: 'POST'})">⏹ Abort</button>
            <button class="action-btn" onclick="fetch('/api/control/restart', {method: 'POST'})">▶ Restart</button>
            <button id="btn-refresh-audio" class="action-btn">🔄 Reload Buffer</button>
            <a id="btn-download-wav" href="/api/audio/recent_wav" download="mic_recent.wav" class="action-btn download-btn">⬇️ Download WAV</a>
          </div>
        </div>
      </section>

      <section class="panel sessions-panel">
        <h2>Recent Sessions</h2>
        <div class="table-wrap">
          <table id="log-table">
            <thead>
              <tr>
                <th>Time</th><th>Duration</th><th>End Reason</th>
                <th>Transcribed Text</th><th>Action</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <div id="live-overlay" class="overlay" aria-hidden="true">
    <div class="live-modal">
      <div class="modal-head">
        <div class="live-label"><i></i> LIVE</div>
        <button id="close-live" aria-label="Close">×</button>
      </div>
      <div id="popup-text" class="popup-text">Listening…</div>
      <div class="wave" aria-label="Audio activity">
        <span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span>
      </div>
      <div class="listening">Listening...</div>
    </div>
  </div>

  <div id="toast" class="toast"><span>✓</span><span id="toast-text"></span></div>
  
  <script>
    const badge = document.getElementById("state-badge");
    const liveText = document.getElementById("live-text");
    const popupText = document.getElementById("popup-text");
    const nodeIpEl = document.getElementById("nodeIp");
    const tableBody = document.querySelector("#log-table tbody");
    const overlay = document.getElementById("live-overlay");
    const closeLive = document.getElementById("close-live");
    const toast = document.getElementById("toast");
    const toastText = document.getElementById("toast-text");
    const waveBars = document.querySelectorAll('.wave-bar');

    function setState(state) {
      badge.innerHTML = `<span></span>${state}`;
      badge.className = "badge " + state.toLowerCase();
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({
        "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
      })[c]);
    }
    
    window.downloadTxt = function(b64_text, time) {
      const text = decodeURIComponent(escape(atob(b64_text)));
      const blob = new Blob([text], {type: "text/plain"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `transcript_${time}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    };

    function addLogRow(s) {
      const b64 = btoa(unescape(encodeURIComponent(s.text)));
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(s.time)}</td>
        <td>${escapeHtml(s.duration)}</td>
        <td>${escapeHtml(s.reason)}</td>
        <td><strong>${escapeHtml(s.text)}</strong></td>
        <td><button class="action-btn" onclick="downloadTxt('${b64}', '${escapeHtml(s.time)}')">Save TXT</button></td>
      `;
      tableBody.prepend(tr);
      while (tableBody.rows.length > 20) tableBody.deleteRow(-1);
    }

    function openLive() {
      popupText.textContent = "Listening…";
      overlay.classList.add("show");
      overlay.setAttribute("aria-hidden", "false");
    }

    function closeLivePopup() {
      overlay.classList.remove("show");
      overlay.setAttribute("aria-hidden", "true");
    }

    function showToast(reason) {
      toastText.textContent = `Session completed (${reason || "ENDED"})`;
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 3500);
    }

    closeLive.addEventListener("click", closeLivePopup);

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    let ws;
    
    function connectWs() {
        ws = new WebSocket(`${protocol}://${location.host}/ws/dashboard`);
        
        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);

          if (msg.event === "init" || msg.event === "state_changed") {
            setState(msg.state);
            if (msg.node_ip) nodeIpEl.textContent = msg.node_ip;
            if (msg.recent_sessions) {
              tableBody.innerHTML = '';
              msg.recent_sessions.forEach(addLogRow);
            }

            
            if (msg.state === "LISTENING" || msg.state === "IDLE") {
              liveText.textContent = "Waiting for a session…";
              closeLivePopup();
              if (msg.state === "IDLE") {
                if (vuBarEl) vuBarEl.style.width = "0%";
                waveBars.forEach(b => { b.style.height = '6px'; b.classList.remove('active-wave'); });
                if (micStatusTag) {
                  micStatusTag.textContent = "Ready";
                  micStatusTag.style.color = "var(--blue)";
                }
                document.querySelector('.transcription-box').style.borderColor = 'var(--card-border)';
                document.querySelector('.transcription-box').style.background = 'rgba(11, 15, 25, 0.8)';
              }
            }
            if (msg.state === "STREAMING") {
              openLive();
              document.querySelector('.live-label').innerHTML = `<i></i> LIVE`;
              document.querySelector('.listening').textContent = 'Listening...';
              document.querySelector('.transcription-box').style.borderColor = 'var(--emerald)';
              document.querySelector('.transcription-box').style.background = 'rgba(16, 185, 129, 0.05)';
            }
            if (msg.state === "PROCESSING") {
              document.querySelector('.live-label').innerHTML = `<i style="background: var(--amber); box-shadow: 0 0 12px var(--amber);"></i> PROCESSING`;
              document.querySelector('.listening').textContent = 'Transcribing audio...';
              document.querySelector('.transcription-box').style.borderColor = 'var(--amber)';
              document.querySelector('.transcription-box').style.background = 'rgba(245, 158, 11, 0.1)';
            }
          }

          if (msg.event === "audio_meter") {
            if (micRmsEl) micRmsEl.textContent = msg.rms;
            if (micPeakEl) micPeakEl.textContent = msg.peak;
            if (micStatusTag) {
              micStatusTag.textContent = "Streaming";
              micStatusTag.style.color = "var(--green)";
            }
            const pct = Math.min(100, Math.max(0, Math.round((msg.rms / 6000) * 100)));
            if (vuBarEl) vuBarEl.style.width = `${pct}%`;
            
            // Animate wave
            waveBars.forEach((bar, idx) => {
              const factor = Math.sin((idx / waveBars.length) * Math.PI) * (msg.rms / 3000);
              const h = Math.max(6, Math.min(48, factor * 48));
              bar.style.height = h + 'px';
              if (msg.rms > 800) { bar.classList.add('active-wave'); }
              else { bar.classList.remove('active-wave'); }
            });
          }

          if (msg.event === "partial_text") {
            if (!overlay.classList.contains("show")) openLive();
            popupText.textContent = msg.text || "Listening…";
            liveText.textContent = msg.text || "Listening…";
          }

          if (msg.event === "session_ended") {
            setState("IDLE");
            closeLivePopup();
            showToast(msg.session.reason);
            if (msg.session.text) liveText.textContent = msg.session.text;
            addLogRow(msg.session);
            reloadAudioBuffer();
          }
        };

        ws.onopen = () => console.log("Connected to dashboard socket");
        ws.onclose = () => {
            console.log("Dashboard socket closed");
            setTimeout(connectWs, 2000);
        }
    }

    // Audio Diagnostics Controller
    const micRmsEl = document.getElementById("mic-rms");
    const micPeakEl = document.getElementById("mic-peak");
    const vuBarEl = document.getElementById("vu-bar");
    const micStatusTag = document.getElementById("mic-status-tag");
    const audioPlayer = document.getElementById("audio-player");
    const btnRefreshAudio = document.getElementById("btn-refresh-audio");
    const btnDownloadWav = document.getElementById("btn-download-wav");

    function reloadAudioBuffer() {
      const cacheBuster = Date.now();
      const url = `/api/audio/recent_wav?t=${cacheBuster}`;
      if (audioPlayer) audioPlayer.src = url;
      if (btnDownloadWav) btnDownloadWav.href = url;
      if (audioPlayer) audioPlayer.load();
    }

    if (btnRefreshAudio) {
      btnRefreshAudio.addEventListener("click", () => {
        reloadAudioBuffer();
        if (audioPlayer) audioPlayer.play().catch(() => {});
      });
    }

    window.onload = connectWs;
  </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="ESP32 TinyML Voice Activator Unified Server & Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--whisper", action="store_true", help="Enable Whisper speech transcription")
    args = parser.parse_args()

    local_ip = get_local_ip()

    print("==================================================================")
    print("      ESP32 TinyML Voice Activator Unified Server & Dashboard     ")
    print("==================================================================")
    print(f"  Web Dashboard (Browser) : http://localhost:{args.port}")
    print(f"  ESP32 WebSocket Node    : ws://{local_ip}:{args.port}/ws/esp32")
    print(f"  Recordings Directory    : {RECORDINGS_DIR}")
    print("------------------------------------------------------------------")
    print("  Controls:")
    print("    - Open http://localhost:8000 in Chrome/Edge to view live UI")
    print("    - Press [Enter] or type 'a' to ABORT streaming")
    print("    - Type 'r' + Enter to RESTART streaming")
    print("    - Press Ctrl+C to shutdown")
    print("==================================================================\n")

    if args.whisper and HAS_WHISPER:
        print("[INIT] Loading Whisper speech model (base.en)...")
        whisper_model = whisper.load_model("base.en")
        print("[INIT] Whisper ready.")

    async def run_server():
        config = uvicorn.Config(app=app, host=args.host, port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        # Run background terminal input loop concurrently
        asyncio.create_task(console_input_loop())
        await server.serve()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n[STOPPED] Server closed by user.")
