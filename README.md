# Low-Latency-and-Efficient-Voice-Activator-for-Edge-Devices

An ultra-lightweight hybrid TinyML keyword spotting and cloud-streaming ASR architecture running on low-power microcontrollers.

Project WhisperEdge is a hardware-efficient, open-source voice activation pipeline designed for resource-constrained IoT devices. It combines a localized, INT8-quantized TinyML Convolutional Neural Network (<256KB RAM, <10% idle CPU) running on an ESP32-S3 to handle real-time custom keyword spotting, paired with an instantaneous WebSocket-based cloud streaming bridge for high-accuracy speech-to-text transcription via Whisper. Built for low-latency, privacy-first edge AI.

---

## 📂 Project Structure

This repository is organized to be extremely simple to set up and run:

```text
Low-Latency-and-Efficient-Voice-Activator-for-Edge-Devices/
│
├── firmware/                        # The Arduino IDE sketch
│   └── esp32_audio_ml/              # Core ESP32-S3 firmware with TinyML inference & streaming
│
├── library/                         # The Machine Learning Model 
│   └── voice_activator_model/       # Edge Impulse exported Arduino library
│
└── dashboard/                       # The sleek Python Web Dashboard & WebSocket server
    ├── server.py                    # Handles audio reception, Whisper transcription, & UI
    └── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Install the Model Library
1. Open Arduino IDE.
2. Go to **Sketch $\rightarrow$ Include Library $\rightarrow$ Add .ZIP Library...**
3. Navigate to `library/voice_activator_model/` (you can zip it first, or just copy the folder into your `Documents/Arduino/libraries/` folder).

### 2. Flash the Firmware
1. Open `firmware/esp32_audio_ml/esp32_audio_ml.ino` in Arduino IDE.
2. Edit the `WIFI_SSID`, `WIFI_PASS`, and `WS_SERVER_IP` (your computer's IP address) at the top of the file.
3. Select your ESP32-S3 board and click **Upload**.

### 3. Run the Dashboard
1. Open a terminal and navigate to the `dashboard/` folder.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python server.py
   ```
4. Open your web browser and go to `http://localhost:8000` to see the live  dashboard!

---

Please check the individual `README.md` inside `firmware/esp32_audio_ml` for more detailed hardware wiring instructions (e.g. INMP441 Microphone pinouts).
