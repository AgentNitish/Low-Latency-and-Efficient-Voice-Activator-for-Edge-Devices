"""
TEST SIMULATOR - pretend to be the ESP32
==========================================
You don't have the hardware yet, so use this script to test your
backend end-to-end. It "plays" a WAV file to your server exactly the
way the real ESP32 will: first 500ms as pre-roll, then in small chunks.

USAGE:
  1. Put any 16kHz, 16-bit, mono .wav file in this folder
     (record yourself saying something using Voice Recorder / Audacity,
      or use the sample one this script generates automatically).
  2. Run:  python test_client_simulator.py
  3. Watch the terminal AND open http://localhost:8000 in your browser
     to see the dashboard update live.
"""

import asyncio
import json
import os
import struct
import wave

import websockets

SERVER_URL = "ws://localhost:8000/ws/esp32"
SAMPLE_WAV = "sample_audio.wav"
CHUNK_SIZE = 1024  # matches the ESP32's live streaming chunk size


def make_sample_wav_if_missing(path=SAMPLE_WAV, seconds=3, freq=440):
    """
    Creates a simple sine-wave tone WAV file so you have SOMETHING to
    test with immediately. This won't produce real transcribed text
    (Vosk can't transcribe a tone!) but it proves the plumbing works:
    connection -> pre-roll -> verification -> streaming -> logging.

    For a REAL test, replace sample_audio.wav with a recording of your
    own voice (16kHz, 16-bit, mono).
    """
    if os.path.exists(path):
        return
    import math
    rate = 16000
    n_samples = rate * seconds
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(3000 * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", value)
        f.writeframes(frames)
    print(f"[SETUP] Created a placeholder tone file: {path}")


async def run_simulation():
    make_sample_wav_if_missing()

    with wave.open(SAMPLE_WAV, "rb") as wf:
        assert wf.getframerate() == 16000, "WAV must be 16kHz"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getnchannels() == 1, "WAV must be mono"
        audio_bytes = wf.readframes(wf.getnframes())

    print(f"[SIM] Connecting to {SERVER_URL} ...")
    async with websockets.connect(SERVER_URL) as ws:
        print("[SIM] Connected. Sending pre-roll (500ms)...")

        pos = 0
        pre_roll = audio_bytes[:16000]
        await ws.send(pre_roll)
        pos = 16000

        # Wait for VERIFIED / ABORT
        response = await ws.recv()
        print("[SIM] Server says:", response)
        status = json.loads(response)
        if status.get("status") != "VERIFIED":
            print("[SIM] Server rejected the wake word. Stopping.")
            return

        print("[SIM] Streaming rest of the audio in small chunks...")
        while pos < len(audio_bytes):
            chunk = audio_bytes[pos:pos + CHUNK_SIZE]
            pos += CHUNK_SIZE
            await ws.send(chunk)
            await asyncio.sleep(0.03)  # roughly mimic real-time mic speed

            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=0.01)
                print("[SIM] Server says:", reply)
                data = json.loads(reply)
                if data.get("action") == "STOP":
                    print("[SIM] Server ended the session.")
                    break
            except asyncio.TimeoutError:
                pass

        print("[SIM] Done. Check the dashboard and logs/session_logs.csv")


if __name__ == "__main__":
    asyncio.run(run_simulation())
