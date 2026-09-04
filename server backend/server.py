"""
SURAKSHA BACKEND SERVER
========================
This ONE file does everything Module 4 needs:
  1. Accepts audio from the ESP32 (or our test simulator) over WebSocket
  2. Runs a "Stage-2" wake-word double-check
  3. Feeds live audio into Vosk (offline speech-to-text)
  4. Sends live captions + status to a browser dashboard
  5. Logs every session to a CSV file for reporting

You do NOT need to understand every line. Read the comments -
they explain WHAT each part does and WHY.
"""

import asyncio
import json
import os
import time
import uuid

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Vosk is optional at import-time so the dashboard still runs even if you
# haven't downloaded the speech model yet (see README "Get the ASR model").
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# -------------------------------------------------------------------
# CONFIG - change these if needed
# -------------------------------------------------------------------
MODEL_PATH = "models/vosk-model-small-en-us-0.15"   # folder you download
LOG_FILE = "logs/session_logs.csv"
PRE_ROLL_BYTES = 16000          # 500ms of 16kHz 16-bit mono audio
VERIFY_THRESHOLD = 0.70         # confidence needed to accept the wake word
SAMPLE_RATE = 16000

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create the CSV log file with headers if it doesn't exist yet
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "session_id", "timestamp", "verification_result",
        "confidence_score", "transcribed_text", "duration_ms",
        "termination_reason"
    ]).to_csv(LOG_FILE, index=False)

# Load the ASR model ONCE when the server starts (loading it per-request
# would be very slow)
asr_model = None
if VOSK_AVAILABLE and os.path.isdir(MODEL_PATH):
    print(f"[INIT] Loading Vosk model from {MODEL_PATH} ...")
    asr_model = Model(MODEL_PATH)
    print("[INIT] ASR model ready.")
else:
    print("[INIT] WARNING: No Vosk model found yet at", MODEL_PATH)
    print("[INIT] The server will still run, but transcription will be skipped.")
    print("[INIT] See README.md -> 'Get the free speech model' to fix this.")

app = FastAPI(title="Suraksha Backend")

# A list of every browser dashboard currently watching, so we can push
# live updates to all of them at once.
dashboard_clients: list[WebSocket] = []


async def broadcast_to_dashboard(message: dict):
    """Send a status/telemetry update to every connected browser tab."""
    dead = []
    for client in dashboard_clients:
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for d in dead:
        dashboard_clients.remove(d)


def verify_stage_2(pre_roll_pcm: bytes) -> tuple[bool, float]:
    """
    STAGE-2 VERIFICATION (placeholder)
    -----------------------------------
    In the full project, Module 1 (the ML teammate) gives you a trained
    model that checks "did this really sound like the wake word?".

    Until that model is ready, this placeholder ALWAYS says yes with a
    fixed confidence score, so you can build and test everything else
    right now without waiting on anyone.

    When the real model is ready, replace the inside of this function
    with the real prediction call - the rest of the server doesn't
    need to change at all.
    """
    return True, 0.94


def log_session(entry: dict):
    """Append one row to the CSV log file."""
    pd.DataFrame([entry]).to_csv(LOG_FILE, mode="a", header=False, index=False)


def transcribe_setup():
    """Create a fresh recognizer for one session (or None if no model)."""
    if asr_model is None:
        return None
    rec = KaldiRecognizer(asr_model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec


# ---------------------------------------------------------------------
# ENDPOINT 1: The ESP32 (or the test simulator) connects here
# ---------------------------------------------------------------------
@app.websocket("/ws/esp32")
async def handle_esp32(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    recognizer = transcribe_setup()
    is_verified = False
    pre_roll_buffer = bytearray()
    final_text = ""
    termination_reason = "UNKNOWN"
    confidence = 0.0

    print(f"\n[CONNECT] Node connected. Session: {session_id}")
    await broadcast_to_dashboard({"event": "state", "state": "LISTENING", "session_id": session_id})

    try:
        while True:
            message = await websocket.receive()

            # --- Client closed the connection ---
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            # --- Binary audio frame ---
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]

                if not is_verified:
                    # Phase A: collect 500ms, then verify the wake word
                    pre_roll_buffer.extend(chunk)
                    if len(pre_roll_buffer) >= PRE_ROLL_BYTES:
                        verified, confidence = verify_stage_2(bytes(pre_roll_buffer[:PRE_ROLL_BYTES]))
                        if verified:
                            print(f"[{session_id}] Verified (score {confidence:.2f})")
                            await websocket.send_json({"status": "VERIFIED"})
                            await broadcast_to_dashboard({
                                "event": "state", "state": "STREAMING",
                                "session_id": session_id, "confidence": confidence
                            })
                            is_verified = True
                            if recognizer:
                                recognizer.AcceptWaveform(bytes(pre_roll_buffer))
                        else:
                            print(f"[{session_id}] Rejected (score {confidence:.2f})")
                            await websocket.send_json({"status": "ABORT"})
                            await broadcast_to_dashboard({
                                "event": "state", "state": "REJECTED", "session_id": session_id
                            })
                            termination_reason = "SERVER_ABORT"
                            break
                else:
                    # Phase B: live streaming + transcription
                    if recognizer is None:
                        continue  # no ASR model loaded yet, just skip
                    if recognizer.AcceptWaveform(chunk):
                        result = json.loads(recognizer.Result())
                        final_text = result.get("text", "")
                        print(f"[{session_id}] Final: \"{final_text}\"")
                        await websocket.send_json({"action": "STOP"})
                        await broadcast_to_dashboard({
                            "event": "final_text", "session_id": session_id, "text": final_text
                        })
                        termination_reason = "SERVER_ENDPOINT"
                        break
                    else:
                        partial = json.loads(recognizer.PartialResult())
                        if partial.get("partial"):
                            await broadcast_to_dashboard({
                                "event": "partial_text", "session_id": session_id,
                                "text": partial["partial"]
                            })

            # --- Text control frame (JSON) ---
            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                if data.get("event") == "TIMEOUT":
                    print(f"[{session_id}] Client hard-timeout.")
                    if recognizer:
                        result = json.loads(recognizer.FinalResult())
                        final_text = result.get("text", "")
                    termination_reason = "CLIENT_TIMEOUT"
                    break

    except WebSocketDisconnect:
        print(f"[{session_id}] Disconnected.")
        if termination_reason == "UNKNOWN":
            termination_reason = "CLIENT_DISCONNECT"

    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        entry = {
            "session_id": session_id,
            "timestamp": timestamp_str,
            "verification_result": "TRUE_POSITIVE" if is_verified else "FALSE_POSITIVE",
            "confidence_score": confidence,
            "transcribed_text": final_text,
            "duration_ms": duration_ms,
            "termination_reason": termination_reason,
        }
        log_session(entry)
        await broadcast_to_dashboard({"event": "state", "state": "IDLE", "session_id": session_id})
        await broadcast_to_dashboard({"event": "log_row", "row": entry})
        print(f"[{session_id}] Logged. Duration {duration_ms}ms | {entry['verification_result']}")


# ---------------------------------------------------------------------
# ENDPOINT 2: The browser dashboard connects here for live updates
# ---------------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def handle_dashboard(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # dashboard doesn't send us anything meaningful
    except WebSocketDisconnect:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)


# ---------------------------------------------------------------------
# Serve the dashboard webpage itself
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/recent_logs")
async def recent_logs():
    """Used by the dashboard to show the last 10 sessions on page load."""
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE).fillna("")
    return df.tail(10).to_dict(orient="records")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("[RUNNING] Open http://localhost:8000 in your browser")
    print("[RUNNING] ESP32 (or simulator) should connect to ws://<this-computer-ip>:8000/ws/esp32")
    uvicorn.run(app, host="0.0.0.0", port=8000)
