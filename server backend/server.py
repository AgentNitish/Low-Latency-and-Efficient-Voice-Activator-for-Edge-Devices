import asyncio
import csv
import json
import os
import tempfile
import time
import uuid
import wave

import pandas as pd
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WHISPER_MODEL_NAME = "base.en"
LOG_FILE = "logs/session_logs.csv"
PRE_ROLL_BYTES = 16000
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

CSV_HEADERS = [
    "session_id", "timestamp", "verification_result",
    "confidence_score", "transcribed_text", "duration_ms",
    "termination_reason"
]

os.makedirs("logs", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADERS)

print(f"[INIT] Loading Whisper model: {WHISPER_MODEL_NAME} ...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("[INIT] Whisper model ready.")

app = FastAPI(title="Suraksha Backend")
dashboard_clients: set[WebSocket] = set()


async def broadcast_to_dashboard(message: dict):
    if not dashboard_clients:
        return

    async def _send(client: WebSocket):
        try:
            await client.send_json(message)
            return None
        except Exception:
            return client

    results = await asyncio.gather(
        *[_send(c) for c in list(dashboard_clients)],
        return_exceptions=True
    )
    for result in results:
        if isinstance(result, WebSocket):
            dashboard_clients.discard(result)


def verify_stage_2(pre_roll_pcm: bytes) -> tuple[bool, float]:
    """
    Existing Stage-2 placeholder.
    Replace only this function later when your real wake-word ML model is ready.
    """
    return True, 0.94


def transcribe_audio_sync(pcm_audio: bytes) -> str:
    """Convert raw 16 kHz / 16-bit / mono PCM to WAV and transcribe with Whisper."""
    if not pcm_audio:
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_audio)

        print("[WHISPER] Transcribing complete audio...")
        result = whisper_model.transcribe(
            temp_path,
            language="en",
            fp16=False,
            temperature=0,
            condition_on_previous_text=False,
        )
        return result.get("text", "").strip()

    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
        return ""

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _append_log_sync(entry: dict):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([entry[h] for h in CSV_HEADERS])


async def log_session(entry: dict):
    await asyncio.to_thread(_append_log_sync, entry)


@app.websocket("/ws/esp32")
async def handle_esp32(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    is_verified = False
    pre_roll_buffer = bytearray()
    session_audio_buffer = bytearray()
    final_text = ""
    termination_reason = "UNKNOWN"
    confidence = 0.0

    print(f"\n[CONNECT] Node connected. Session: {session_id}")
    await broadcast_to_dashboard({
        "event": "state",
        "state": "LISTENING",
        "session_id": session_id
    })

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]

                if not is_verified:
                    pre_roll_buffer.extend(chunk)

                    if len(pre_roll_buffer) >= PRE_ROLL_BYTES:
                        pcm_sample = bytes(pre_roll_buffer[:PRE_ROLL_BYTES])
                        verified, confidence = await asyncio.to_thread(
                            verify_stage_2, pcm_sample
                        )

                        if verified:
                            print(
                                f"[{session_id}] Verified "
                                f"(score {confidence:.2f})"
                            )
                            await websocket.send_json({"status": "VERIFIED"})
                            await broadcast_to_dashboard({
                                "event": "state",
                                "state": "STREAMING",
                                "session_id": session_id,
                                "confidence": confidence
                            })

                            is_verified = True

                            # Keep the pre-roll because it may contain the
                            # beginning of the user's spoken sentence.
                            session_audio_buffer.extend(pre_roll_buffer)
                            pre_roll_buffer.clear()

                        else:
                            print(
                                f"[{session_id}] Rejected "
                                f"(score {confidence:.2f})"
                            )
                            await websocket.send_json({"status": "ABORT"})
                            await broadcast_to_dashboard({
                                "event": "state",
                                "state": "REJECTED",
                                "session_id": session_id
                            })
                            termination_reason = "SERVER_ABORT"
                            break

                else:
                    # Whisper transcription happens when the session ends.
                    # Collect every verified audio chunk here.
                    session_audio_buffer.extend(chunk)

            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("event") == "TIMEOUT":
                    print(f"[{session_id}] Client finished sending audio.")
                    termination_reason = "CLIENT_TIMEOUT"
                    break

                if data.get("event") == "END":
                    print(f"[{session_id}] Client sent END event.")
                    termination_reason = "CLIENT_END"
                    break

    except WebSocketDisconnect:
        print(f"[{session_id}] Disconnected.")
        if termination_reason == "UNKNOWN":
            termination_reason = "CLIENT_DISCONNECT"

    finally:
        # IMPORTANT: Whisper transcribes the complete audio once the
        # simulator/ESP32 has finished or disconnected.
        if is_verified and session_audio_buffer:
            print(
                f"[{session_id}] Audio received: "
                f"{len(session_audio_buffer)} bytes"
            )

            final_text = await asyncio.to_thread(
                transcribe_audio_sync,
                bytes(session_audio_buffer)
            )

            print(f'[{session_id}] Whisper Final: "{final_text}"')

            await broadcast_to_dashboard({
                "event": "final_text",
                "session_id": session_id,
                "text": final_text
            })
        else:
            print(f"[{session_id}] No verified audio available for transcription.")

        duration_ms = int((time.time() - start_time) * 1000)

        entry = {
            "session_id": session_id,
            "timestamp": timestamp_str,
            "verification_result": (
                "TRUE_POSITIVE" if is_verified else "FALSE_POSITIVE"
            ),
            "confidence_score": confidence,
            "transcribed_text": final_text,
            "duration_ms": duration_ms,
            "termination_reason": termination_reason,
        }

        await log_session(entry)

        await broadcast_to_dashboard({
            "event": "state",
            "state": "IDLE",
            "session_id": session_id
        })

        await broadcast_to_dashboard({
            "event": "log_row",
            "row": entry
        })

        print(
            f"[{session_id}] Logged. Duration {duration_ms}ms | "
            f"{entry['verification_result']}"
        )


@app.websocket("/ws/dashboard")
async def handle_dashboard(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.add(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_clients.discard(websocket)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/recent_logs")
async def recent_logs():
    if not os.path.exists(LOG_FILE):
        return []

    def _read_recent():
        df = pd.read_csv(LOG_FILE).fillna("")
        return df.tail(10).to_dict(orient="records")

    return await asyncio.to_thread(_read_recent)


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    print("[RUNNING] Open http://localhost:8000 in your browser")
    print(
        "[RUNNING] ESP32 (or simulator) should connect to "
        "ws://<this-computer-ip>:8000/ws/esp32"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
