# Low-Latency-and-Efficient-Voice-Activator-for-Edge-Devices
An ultra-lightweight hybrid TinyML keyword spotting and cloud-streaming ASR architecture running on low-power microcontrollers.

Project WhisperEdge is a hardware-efficient, open-source voice activation pipeline designed for resource-constrained IoT devices. It combines a localized, INT8-quantized TinyML Convolutional Neural Network (<256KB RAM, <10% idle CPU) running on an ESP32-S3 to handle real-time custom keyword spotting, paired with an instantaneous WebSocket-based cloud streaming bridge for high-accuracy speech-to-text transcription via Whisper. Built for low-latency, privacy-first edge AI.
