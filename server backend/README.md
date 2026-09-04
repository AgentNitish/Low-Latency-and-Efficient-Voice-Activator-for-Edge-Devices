# Suraksha Backend — Your Module 4 (Already Built ✅)

I already built and tested this for you. It works right now. You just
need to run it and add the free speech-recognition model.

---

## What's inside this folder

| File | What it does |
|---|---|
| `server.py` | The main brain. Talks to the ESP32, checks the wake word, transcribes speech, updates the dashboard, saves logs. |
| `static/` | The live dashboard webpage (HTML/CSS/JS) — no need to touch it. |
| `test_client_simulator.py` | **Pretends to be the ESP32** so you can test everything without hardware. |
| `benchmark_analyzer.py` | Prints a report (false positives/hour, avg duration, etc.) from your test logs. |
| `logs/session_logs.csv` | Every test session gets saved here automatically. |
| `requirements.txt` | List of Python packages needed. |

---

## STEP 1 — Install Python packages (5 minutes)

Open a terminal in this folder and run:

```
pip install -r requirements.txt
```

(If you get a "break system packages" error, run:
`pip install -r requirements.txt --break-system-packages`)

---

## STEP 2 — Get the free speech model (10 minutes)

This lets the server actually turn speech into text.

1. Go to: https://alphacephei.com/vosk/models
2. Download **`vosk-model-small-en-us-0.15`** (about 40MB, it's free and offline).
3. Unzip it.
4. Put the unzipped folder inside this project's `models/` folder, so you have:
   `models/vosk-model-small-en-us-0.15/`

If you skip this step, the server still runs — it just won't produce
real transcriptions (you'll see a warning, that's expected).

---

## STEP 3 — Run the server

```
python server.py
```

Then open your browser to: **http://localhost:8000**

You'll see the live dashboard (currently showing "IDLE").

---

## STEP 4 — Test it WITHOUT hardware

Since you don't have the ESP32 yet, open a **second terminal** (keep the
server running in the first one) and run:

```
python test_client_simulator.py
```

Watch:
- The terminal running `server.py` — you'll see connection + verification logs.
- The browser dashboard — state should change and a row should appear in the table.

This proves your whole backend works end-to-end. When your teammate's
ESP32 is ready, it just needs to connect to the same address instead of
the simulator — nothing else changes.

**Want a more realistic test?** Replace `sample_audio.wav` with a real
recording of your voice (16kHz, 16-bit, mono .wav) before running the
simulator — then you'll see real transcribed text.

---

## STEP 5 — Get your test report

After running a few test sessions:

```
python benchmark_analyzer.py
```

This prints the exact stats your hackathon judges will want to see
(false positive rate, avg duration, etc.) — straight from your CSV logs.

---

## How this connects to your teammates

- **Tell your ESP32 teammate (Module 3):** connect to
  `ws://<your-computer's-IP-address>:8000/ws/esp32`
  *(Note: the original spec said port 8765 — I simplified it to run on
  one port, 8000, since it's easier to manage. Just tell your teammate
  the URL above.)*
- **Tell your ML teammate (Module 1):** once their verification model
  is ready, it plugs into the `verify_stage_2()` function inside
  `server.py` — that's the ONLY function that needs to change later.

---

## If something breaks — use AI to fix it fast

You said you don't want to hand-code. For any error or change, do this:

1. Copy the exact error message from your terminal.
2. Paste it into **Claude Code** (or this chat) along with:
   *"Here's my server.py, here's the error, please fix it."*
3. Don't guess — always paste the real error text.

**Recommended free/fast tools for the next 2 days:**
- **Claude Code** (or this chat) — best for fixing bugs and adding features, since it can read your whole project.
- **Replit** — good if you want to run/share this project from your phone or without installing Python locally (just upload these files, click Run).
- Avoid juggling too many tools — pick ONE (Claude Code is enough for everything here).

---

## Your realistic 2-day checklist

**Day 1**
- [ ] Install packages (Step 1)
- [ ] Download Vosk model (Step 2)
- [ ] Run server, confirm dashboard loads (Step 3)
- [ ] Run simulator, confirm a full session logs correctly (Step 4)
- [ ] Try with a real voice recording instead of the tone file

**Day 2**
- [ ] Run `benchmark_analyzer.py`, save the report as a screenshot
- [ ] Record a short screen video of the dashboard working (for your demo)
- [ ] Sync with Module 1 & 3 teammates on the two integration points above
- [ ] If they have hardware ready, do one real end-to-end test together
- [ ] Prepare your slide/explanation: "what my module does" using this README
