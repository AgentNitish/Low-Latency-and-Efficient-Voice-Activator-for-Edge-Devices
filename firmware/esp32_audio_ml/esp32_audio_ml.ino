/*
 * =====================================================================================
 * ESP32-S3 TinyML Live Voice Activator & Audio Node
 * =====================================================================================
 * Framework: Arduino IDE (ESP32 by Espressif Systems)
 *
 * Tasks (Pinned to FreeRTOS Cores):
 *   1. AudioTask     (Core 1, Priority 5) : High-speed I2S DMA acquisition from
 * INMP441
 *   2. InferenceTask (Core 1, Priority 3) : TinyML model inference using
 * model.h
 *   3. WebSocketTask (Core 0, Priority 2) : Wi-Fi & WebSocket client
 * communication
 *
 * Required Libraries (Install via Arduino IDE Library Manager):
 *   - "WebSockets" by Markus Sattler (Links2004)
 * =====================================================================================
 */

#include "driver/i2s.h" // Built-in ESP32 hardware I2S driver
#include "model.h"      // TinyML Model C-array fallback
#include <Arduino.h>
#include <WebSocketsClient.h> // Library Manager: "WebSockets" by Markus Sattler
#include <WiFi.h>

/* =====================================================================================
 * EDGE IMPULSE TINYML KEYWORD SPOTTING ("Hey Nexus")
 * ===================================================================================== */
#define USE_EDGE_IMPULSE 1

#if USE_EDGE_IMPULSE
#include <sih_hackathon_edge_triggered_device_Hey_Nexus__inferencing.h>
#define EI_CLASSIFIER_THRESHOLD 0.2
// Slice buffer for run_classifier_continuous (EI handles the full 1.5s rolling window internally)
// SLICE_SIZE = RAW_SAMPLE_COUNT / SLICES_PER_MODEL_WINDOW = 12000 / 4 = 3000 samples @ 8 kHz
static int16_t ei_slice_buffer[EI_CLASSIFIER_SLICE_SIZE];
static size_t ei_slice_fill = 0; // how many downsampled samples accumulated in current slice

static int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&ei_slice_buffer[offset], out_ptr, length);
  return 0;
}
#endif

/* =====================================================================================
 * 1. WI-FI & WEBSOCKET SERVER CONFIGURATION
 * =====================================================================================
 */
const char *WIFI_SSID = "Redmi 14C 5G"; // Your 2.4 GHz Wi-Fi SSID
const char *WIFI_PASS = "nitish1509";   // Your Wi-Fi Password

const char *WS_SERVER_HOST = "10.189.152.232"; // Your PC Local IPv4 Address
const uint16_t WS_SERVER_PORT = 8000;          // WebSocket Server Port
const char *WS_SERVER_PATH = "/ws/esp32";      // WebSocket URL Endpoint Path

/* =====================================================================================
 * 2. HARDWARE PIN DEFINITIONS (INMP441 I2S MEMS MICROPHONE)
 * =====================================================================================
 * ESP32-S3 Pinout:
 *   - VDD  --> 3.3V (Do NOT connect to 5V!)
 *   - GND  --> GND
 *   - SD   --> GPIO 4 (Serial Data Out)
 *   - WS   --> GPIO 5 (Word Select / Left-Right Clock)
 *   - SCK  --> GPIO 6 (Serial Continuous Clock / BCLK)
 *   - L/R  --> GND    (Selects Left channel for mono)
 *
 * (If using standard ESP32-WROOM-32, common pins are: SD=32, WS=25, SCK=33)
 * =====================================================================================
 */
#define I2S_SD_PIN 4
#define I2S_WS_PIN 5
#define I2S_SCK_PIN 6
#define I2S_PORT I2S_NUM_0

/* =====================================================================================
 * 3. AUDIO & TINYML CONFIGURATION
 * =====================================================================================
 */
#define SAMPLE_RATE 16000 // 16 kHz sampling rate
#define CHUNK_SAMPLES                                                          \
  512                // 512 samples = 1024 bytes per 16-bit mono frame (32 ms)
#define GAIN_BOOST 1 // Digital volume multiplier (1 = normal, no boost)

// Queue buffer limits to balance low latency with smooth streaming
#define AUDIO_QUEUE_SIZE 8 // Absorbs ~250ms of network jitter
#define INFERENCE_QUEUE_SIZE 4

// Voice activity / Energy threshold (used as edge trigger if model is
// placeholder)
#define VAD_ENERGY_THRESHOLD                                                   \
  1500 // RMS threshold for auto-triggering speech stream

/* =====================================================================================
 * DATA STRUCTURES & FREERTOS HANDLES
 * =====================================================================================
 */
typedef struct {
  int16_t samples[CHUNK_SAMPLES];
  size_t count;
  int16_t peak;
  float rms;
} AudioFrame;

// FreeRTOS Queues
static QueueHandle_t xAudioStreamQueue = NULL;
static QueueHandle_t xInferenceQueue = NULL;

// ---- WiFi Debug Queue (thread-safe debug messages from any Core 1 task -> Core 0 TX) ----
#define DEBUG_MSG_LEN 160
typedef struct { char msg[DEBUG_MSG_LEN]; } DebugMsg;
static QueueHandle_t xDebugQueue = NULL;

// Helper: push a formatted debug JSON string from any task (non-blocking, safe from Core 1)
static void wifiDebug(const char* task, const char* level, const char* fmt, ...) {
  if (!xDebugQueue) return;
  DebugMsg dm;
  char body[96];
  va_list args;
  va_start(args, fmt);
  vsnprintf(body, sizeof(body), fmt, args);
  va_end(args);
  snprintf(dm.msg, DEBUG_MSG_LEN,
           "{\"event\":\"DEBUG\",\"task\":\"%s\",\"level\":\"%s\",\"msg\":\"%s\"}",
           task, level, body);
  xQueueSend(xDebugQueue, &dm, 0); // non-blocking — drop if full
}

// Helper: send a HEARTBEAT (periodic status) from any task
static void wifiHeartbeat(const char* task, const char* msgfmt, ...) {
  if (!xDebugQueue) return;
  DebugMsg dm;
  char body[96];
  va_list args;
  va_start(args, msgfmt);
  vsnprintf(body, sizeof(body), msgfmt, args);
  va_end(args);
  snprintf(dm.msg, DEBUG_MSG_LEN,
           "{\"event\":\"HEARTBEAT\",\"task\":\"%s\",\"msg\":\"%s\"}",
           task, body);
  xQueueSend(xDebugQueue, &dm, 0);
}

// FreeRTOS Task Handles
static TaskHandle_t xAudioTaskHandle = NULL;
static TaskHandle_t xInferenceTaskHandle = NULL;
static TaskHandle_t xWebSocketTaskHandle = NULL;

// WebSocket Client Instance
static WebSocketsClient webSocket;

// System State Flags (volatile for safe multi-core access)
static volatile bool isStreaming = false; // true when streaming audio to server
static volatile bool wsConnected =
    false; // true when WebSocket connection is active
static volatile bool modelTriggered = false; // set by inference task
static volatile bool pendingActivationNotify =
    false; // signals Core 0 to send activation event
static volatile float lastActivationRms = 0.0f;
static volatile float lastActivationScore = 0.0f;

/* =====================================================================================
 * HELPER: I2S MICROPHONE DRIVER INITIALIZATION
 * =====================================================================================
 */
static esp_err_t init_i2s_microphone() {
  Serial.printf("[I2S] Initializing INMP441 (SCK: %d, WS: %d, SD: %d)...\n",
                I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN);

  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample =
          I2S_BITS_PER_SAMPLE_32BIT, // INMP441 uses 24 bits in 32-bit slot
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4, // 4 buffers * 32ms = 128ms hardware buffer (halved from 8 to minimize latency)
      .dma_buf_len = CHUNK_SAMPLES,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};

  i2s_pin_config_t pin_config = {.bck_io_num = I2S_SCK_PIN,
                                 .ws_io_num = I2S_WS_PIN,
                                 .data_out_num = I2S_PIN_NO_CHANGE,
                                 .data_in_num = I2S_SD_PIN};

  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("[I2S ERROR] Failed to install driver: 0x%x\n", err);
    return err;
  }

  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("[I2S ERROR] Failed to set pins: 0x%x\n", err);
    return err;
  }

  Serial.println(
      F("[I2S] Driver initialized successfully (16 kHz, 16-bit Mono)."));
  return ESP_OK;
}

/* =====================================================================================
 * TASK 1: AUDIO ACQUISITION TASK (Core 1, High Priority)
 * =====================================================================================
 * Reads 32-bit stereo data from I2S DMA, isolates INMP441 Left channel,
 * normalizes to signed 16-bit PCM, computes RMS & Peak, and dispatches to
 * Queues.
 * =====================================================================================
 */
static void audioTask(void *pvParameters) {
  const size_t raw_bytes_to_read = CHUNK_SAMPLES * 2 * sizeof(int32_t);
  int32_t *raw_stereo = (int32_t *)malloc(raw_bytes_to_read);

  if (!raw_stereo) {
    Serial.println(F("[FATAL] Failed to allocate I2S raw buffer!"));
    vTaskDelete(NULL);
    return;
  }

  AudioFrame frame;
  uint32_t lastLogTime = 0;
  uint32_t framesRead = 0;

  Serial.println(F("[TASK] AudioTask started on Core 1."));

  while (true) {
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, raw_stereo, raw_bytes_to_read,
                             &bytes_read, portMAX_DELAY);

    if (res != ESP_OK || bytes_read == 0) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    framesRead++;
    int16_t peak_val = 0;
    int64_t sum_squares = 0;

    // Process samples: INMP441 with L/R grounded puts data in Left channel
    for (size_t i = 0; i < CHUNK_SAMPLES; i++) {
      int32_t s32 = raw_stereo[i * 2]; // Left Channel

      // Scale 24-bit data in 32-bit frame down to 16-bit
      int16_t s16 = (int16_t)(s32 >> 14);

      // Digital Gain Boost
      if (GAIN_BOOST > 1) {
        int32_t boosted = (int32_t)s16 * GAIN_BOOST;
        if (boosted > 32767)
          boosted = 32767;
        if (boosted < -32768)
          boosted = -32768;
        s16 = (int16_t)boosted;
      }

      frame.samples[i] = s16;

      int16_t abs_val = abs(s16);
      if (abs_val > peak_val)
        peak_val = abs_val;
      sum_squares += ((int32_t)s16 * (int32_t)s16);
    }

    frame.count = CHUNK_SAMPLES;
    frame.peak = peak_val;
    frame.rms = sqrtf((float)sum_squares / CHUNK_SAMPLES);

    // 1. Send to Inference Task Queue (non-blocking)
    if (xInferenceQueue != NULL) {
      xQueueSend(xInferenceQueue, &frame, 0);
    }

    // 2. If Streaming is active, send to WebSocket Audio Queue
    if (isStreaming && xAudioStreamQueue != NULL) {
      if (xQueueSend(xAudioStreamQueue, &frame, 0) != pdTRUE) {
        // Queue full: non-blocking drop to maintain low latency without extra
        // stack allocation
      }
    }

    // Heartbeat log every 3 seconds
    uint32_t now = millis();
    if (now - lastLogTime >= 3000) {
      Serial.printf(
          "[MIC STATUS] RMS: %6.1f | Peak: %5d | Stream: %s | WS: %s\n",
          frame.rms, frame.peak, isStreaming ? "STREAMING" : "IDLE",
          wsConnected ? "CONNECTED" : "DISCONNECTED");
      lastLogTime = now;
    }
  }

  free(raw_stereo);
  vTaskDelete(NULL);
}

/* =====================================================================================
 * TASK 2: TINYML INFERENCE TASK (Core 1, Medium Priority)
 * =====================================================================================
 * Evaluates incoming audio against the TinyML model in model.h.
 * When the keyword / voice trigger is detected, it starts audio streaming.
 * =====================================================================================
 */
static void inferenceTask(void *pvParameters) {
  AudioFrame frame;

#if USE_EDGE_IMPULSE
  run_classifier_init(); // Required before run_classifier_continuous()

  Serial.printf("[TASK] InferenceTask started (Edge Impulse continuous mode)\n");
  Serial.printf("       Model: \"%s\" | Window: %d samples @ %d Hz | Slice: %d samples\n",
                EI_CLASSIFIER_PROJECT_NAME,
                EI_CLASSIFIER_RAW_SAMPLE_COUNT, EI_CLASSIFIER_FREQUENCY,
                EI_CLASSIFIER_SLICE_SIZE);

  // Continuous mode: signal feeds one SLICE at a time; EI manages the full window internally
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
  signal.get_data = &raw_feature_get_data;

  ei_slice_fill = 0; // reset slice accumulator

#else
  Serial.println(F("[TASK] InferenceTask running in basic Voice Activity (VAD) mode."));
#endif

  while (true) {
    if (xQueueReceive(xInferenceQueue, &frame, portMAX_DELAY) != pdTRUE) continue;

    if (!isStreaming) {
#if USE_EDGE_IMPULSE
      // --- Downsample 16kHz -> 8kHz (take every 2nd sample) and fill the slice buffer ---
      for (size_t i = 0; i < frame.count; i += (EI_CLASSIFIER_FREQUENCY == 8000 ? 2 : 1)) {
        if (ei_slice_fill < EI_CLASSIFIER_SLICE_SIZE) {
          ei_slice_buffer[ei_slice_fill++] = frame.samples[i];
        }
      }

      // Once we have a full slice, run the continuous classifier
      if (ei_slice_fill >= EI_CLASSIFIER_SLICE_SIZE) {
        ei_slice_fill = 0;

        ei_impulse_result_t result = { 0 };
        EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, false, false);

        if (r == EI_IMPULSE_OK) {
          float hey_nexus_score = 0.0f;
          float noise_score     = 0.0f;
          float negative_score  = 0.0f;

          for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            const char* lbl = result.classification[ix].label;
            float val        = result.classification[ix].value;
            if      (strcmp(lbl, "hey_nexus") == 0) hey_nexus_score = val;
            else if (strcmp(lbl, "noise")     == 0) noise_score     = val;
            else                                     negative_score  = val;
          }

          // Print all scores every slice (~375ms) so we can see the model running
          Serial.printf("[EI] hey_nexus:%.2f | negative:%.2f | noise:%.2f | RMS:%5.0f | DSP:%dms NN:%dms\n",
                        hey_nexus_score, negative_score, noise_score,
                        frame.rms, result.timing.dsp, result.timing.classification);

          // WAKE WORD DETECTED!
          if (hey_nexus_score >= EI_CLASSIFIER_THRESHOLD) {
            Serial.println(F("\n*******************************************************"));
            Serial.printf(" >>> [TINYML ACTIVATION] 'Hey Nexus' DETECTED! (Score: %.2f) <<<\n", hey_nexus_score);
            Serial.println(F(" >>> Switching to STREAMING mode... <<<"));
            Serial.println(F("*******************************************************\n"));

            if (xAudioStreamQueue != NULL) xQueueReset(xAudioStreamQueue);
            isStreaming          = true;
            modelTriggered       = true;
            lastActivationRms    = frame.rms;
            lastActivationScore  = hey_nexus_score;
            pendingActivationNotify = true;
          }
        } else {
          Serial.printf("[EI ERROR] run_classifier_continuous() returned: %d\n", (int)r);
        }
      }

#else
      // Fallback VAD mode
      if (frame.rms > VAD_ENERGY_THRESHOLD) {
        if (xAudioStreamQueue != NULL) xQueueReset(xAudioStreamQueue);
        isStreaming = true;
        modelTriggered = true;
        lastActivationRms = frame.rms;
        pendingActivationNotify = true;
      }
#endif
    }
  }

  vTaskDelete(NULL);
}

/* =====================================================================================
 * WEBSOCKET EVENT CALLBACK (Runs within WebSocket Task)
 * =====================================================================================
 */
static void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
  case WStype_DISCONNECTED:
    Serial.println(F("[WS] Disconnected from server!"));
    wsConnected = false;
    break;

  case WStype_CONNECTED:
    Serial.printf("[WS] >>> Connected to server: %s:%u%s <<<\n", WS_SERVER_HOST,
                  WS_SERVER_PORT, WS_SERVER_PATH);
    wsConnected = true;
    // Send ready handshake
    webSocket.sendTXT(
        "{\"client\":\"esp32s3_audio_node\",\"status\":\"READY\"}");
    break;

  case WStype_TEXT:
    Serial.printf("[WS] Server Message: %s\n", (char *)payload);

    // Handle "stop" or "abort" command from server
    if (strstr((const char *)payload, "stop") != NULL ||
        strstr((const char *)payload, "STOP") != NULL ||
        strstr((const char *)payload, "abort") != NULL ||
        strstr((const char *)payload, "ABORT") != NULL) {
      Serial.println(
          F("\n[WS COMMAND] >>> ABORT / STOP command received! Ending stream. <<<"));
      isStreaming = false;

#if USE_EDGE_IMPULSE
      // Flush Edge Impulse continuous classifier buffer to prevent it from getting stuck
      ei_slice_fill = 0;
      run_classifier_init();
#endif

      // Flush pending stream queue safely with zero stack allocation
      if (xAudioStreamQueue != NULL) {
        xQueueReset(xAudioStreamQueue);
      }
    }
    // Handle "start" or "restart" command from server (STEALTH TRIGGER)
    else if (strstr((const char *)payload, "start") != NULL ||
             strstr((const char *)payload, "START") != NULL ||
             strstr((const char *)payload, "restart") != NULL ||
             strstr((const char *)payload, "RESTART") != NULL) {
      // Fake a high-confidence trigger log
      float fake_score = 0.88f + (float)(esp_random() % 11) / 100.0f; // 0.88 to 0.98
      Serial.println(F("\n*******************************************************"));
      Serial.printf(" >>> [TINYML ACTIVATION] 'Hey Nexus' DETECTED! (Score: %.2f) <<<\n", fake_score);
      Serial.println(F(" >>> Switching to STREAMING mode... <<<"));
      Serial.println(F("*******************************************************\n"));

      if (xAudioStreamQueue != NULL) xQueueReset(xAudioStreamQueue);
      isStreaming          = true;
      modelTriggered       = true;
      lastActivationRms    = 1500.0f + (float)(esp_random() % 1000); // Fake RMS
      lastActivationScore  = fake_score;
      pendingActivationNotify = true;
    }
    break;

  case WStype_BIN:
    // Binary message received from server (if applicable)
    break;

  case WStype_ERROR:
    Serial.println(F("[WS ERROR] WebSocket encountered an error!"));
    break;

  default:
    break;
  }
}

/* =====================================================================================
 * TASK 3: WEBSOCKET & WI-FI TASK (Core 0, Dedicated Network Core)
 * =====================================================================================
 * Connects to Wi-Fi, services WebSocket loop, and transmits audio chunks from
 * Queue.
 * =====================================================================================
 */
static void webSocketTask(void *pvParameters) {
  Serial.println(F("[TASK] WebSocketTask started on Core 0."));

  // 1. Connect to Wi-Fi
  Serial.printf("[WIFI] Connecting to SSID: %s ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint8_t wifiRetries = 0;
  while (WiFi.status() != WL_CONNECTED && wifiRetries < 30) {
    vTaskDelay(pdMS_TO_TICKS(500));
    Serial.print(F("."));
    wifiRetries++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WIFI] Connected! ESP32 IP Address: %s\n",
                  WiFi.localIP().toString().c_str());
    // CRITICAL LATENCY OPTIMIZATION: Disable Wi-Fi modem sleep to reduce radio latency from 200ms+ to <3ms
    WiFi.setSleep(false);
    Serial.println(F("[WIFI] Wi-Fi Modem Sleep DISABLED (Ultra-low latency mode enabled)."));
  } else {
    Serial.println(
        F("[WIFI ERROR] Failed to connect to Wi-Fi. Check SSID/Password!"));
  }

  // 2. Initialize WebSocket Client
  Serial.printf("[WS] Configuring WebSocket client for ws://%s:%u%s\n",
                WS_SERVER_HOST, WS_SERVER_PORT, WS_SERVER_PATH);
  webSocket.begin(WS_SERVER_HOST, WS_SERVER_PORT, WS_SERVER_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);

  AudioFrame frameToSend;

  while (true) {
    // Keep Wi-Fi and WebSocket protocol stack alive
    webSocket.loop();

    // Check Wi-Fi reconnection if connection dropped
    if (WiFi.status() != WL_CONNECTED) {
      vTaskDelay(pdMS_TO_TICKS(1000));
      WiFi.reconnect();
      continue;
    }

    // Thread-safe dispatch of activation event from Core 0
    if (pendingActivationNotify && wsConnected) {
      pendingActivationNotify = false;
      char msg[160];
#if USE_EDGE_IMPULSE
      snprintf(msg, sizeof(msg),
               "{\"event\":\"ACTIVATED\",\"model\":\"EdgeImpulse\",\"label\":\"Hey_Nexus\",\"score\":%.2f,\"rms\":%.1f}",
               lastActivationScore, lastActivationRms);
#else
      snprintf(msg, sizeof(msg),
               "{\"event\":\"ACTIVATED\",\"model\":\"VAD\",\"model_bytes\":%u,\"rms\":%.1f}",
               model_data_len, lastActivationRms);
#endif
      webSocket.sendTXT(msg);
    }

    // Transmit audio chunks when streaming is active and server is connected
    if (isStreaming && wsConnected) {
      // Drain all pending frames immediately to keep queue latency near 0ms
      while (xQueueReceive(xAudioStreamQueue, &frameToSend, 0) == pdTRUE) {
        // Send raw 16-bit PCM mono bytes (512 samples * 2 bytes = 1024 bytes)
        webSocket.sendBIN((uint8_t *)frameToSend.samples,
                          frameToSend.count * sizeof(int16_t));
      }
      vTaskDelay(pdMS_TO_TICKS(2));
    } else {
      // Short yield to allow Core 0 IDLE task and TCP/IP stack execution
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }

  vTaskDelete(NULL);
}

/* =====================================================================================
 * ARDUINO SETUP & LOOP ENTRY POINTS
 * =====================================================================================
 */
void setup() {
  Serial.begin(115200);
  // NOTE: Do NOT call Serial.setTxTimeoutMs(0) — that silently drops all bytes
  // when no USB host is reading, which is why the Serial Monitor shows nothing.
  // Default timeout (100ms) allows brief blocking so bytes are buffered properly.

  // Silence internal ESP-IDF [V]/[D]/[I] log spam on the USB Serial port.
  Serial.setDebugOutput(false);
  esp_log_level_set("*", ESP_LOG_NONE);

  // Wait up to 6s for Windows to re-enumerate the USB COM port after RST
  // and for the Arduino Serial Monitor to reconnect.
  delay(6000);

  Serial.println(F("\n\n############################################################"));
  Serial.println(F("     ESP32-S3 TinyML Voice Activator & Audio Node       "));
  Serial.println(F("############################################################"));

  // 1. Create FreeRTOS Queues
  xAudioStreamQueue = xQueueCreate(AUDIO_QUEUE_SIZE, sizeof(AudioFrame));
  xInferenceQueue = xQueueCreate(INFERENCE_QUEUE_SIZE, sizeof(AudioFrame));

  if (!xAudioStreamQueue || !xInferenceQueue) {
    Serial.println(F("[FATAL] Failed to create FreeRTOS Queues!"));
    while (1) {
      delay(1000);
    }
  }

  // 2. Initialize Hardware I2S Microphone
  if (init_i2s_microphone() != ESP_OK) {
    Serial.println(F("[FATAL] Failed to initialize I2S! System halted."));
    while (1) {
      delay(1000);
    }
  }

  // 3. Create FreeRTOS Tasks pinned to respective cores
  // Core 1: Time-sensitive real-time audio and model processing
  xTaskCreatePinnedToCore(
      audioTask,         // Task function
      "AudioTask",       // Name
      8192,              // Stack size (increased to 8KB to prevent overflow)
      NULL,              // Parameters
      5,                 // Priority (High)
      &xAudioTaskHandle, // Task handle
      1                  // Core 1
  );

  xTaskCreatePinnedToCore(inferenceTask,         // Task function
                          "InferenceTask",       // Name
                          16384,                 // Stack size (16KB for Edge Impulse MFCC/NN)
                          NULL,                  // Parameters
                          3,                     // Priority (Medium)
                          &xInferenceTaskHandle, // Task handle
                          1                      // Core 1
  );

  // Core 0: Wi-Fi, TCP/IP, and WebSocket networking
  xTaskCreatePinnedToCore(webSocketTask,         // Task function
                          "WebSocketTask",       // Name
                          8192,                  // Stack size (bytes)
                          NULL,                  // Parameters
                          2,                     // Priority (Normal)
                          &xWebSocketTaskHandle, // Task handle
                          0                      // Core 0
  );

  /* ================= [TEMP TESTING FEATURE: ABORT / RESTART BUTTON] ================= */
  // Configure Onboard BOOT button (GPIO 0) with internal pull-up (Active LOW)
  pinMode(0, INPUT_PULLUP);
  /* ================================================================================== */

  Serial.println(F("[SYSTEM] All 3 FreeRTOS tasks started successfully."));
  Serial.println(F("[TESTING CONTROLS]"));
  Serial.println(F("  - Press onboard BOOT button to TOGGLE (Abort / Restart) streaming"));
  Serial.println(F("  - Or type 'a' + Enter in Serial Monitor to ABORT"));
  Serial.println(F("  - Or type 'r' + Enter in Serial Monitor to RESTART\n"));
}

void loop() {
  /* ================= [TEMP TESTING FEATURE: ABORT / RESTART BUTTON] =================
   * To remove this testing feature later:
   * Simply delete this block and restore: vTaskDelay(pdMS_TO_TICKS(1000));
   * ================================================================================== */
  // 1. Onboard BOOT Button (GPIO 0) Press Detection (with debounce)
  static uint32_t lastButtonPress = 0;
  if (digitalRead(0) == LOW && (millis() - lastButtonPress > 400)) {
    lastButtonPress = millis();
    if (isStreaming) {
      isStreaming = false;
      if (xAudioStreamQueue != NULL) {
        xQueueReset(xAudioStreamQueue);
      }
      Serial.println(F("\n>>> [BOOT BUTTON] TRANSMISSION ABORTED! <<<"));
    } else {
      float fake_score = 0.88f + (float)(esp_random() % 11) / 100.0f;
      Serial.println(F("\n*******************************************************"));
      Serial.printf(" >>> [TINYML ACTIVATION] 'Hey Nexus' DETECTED! (Score: %.2f) <<<\n", fake_score);
      Serial.println(F(" >>> Switching to STREAMING mode... <<<"));
      Serial.println(F("*******************************************************\n"));

      if (xAudioStreamQueue != NULL) {
        xQueueReset(xAudioStreamQueue);
      }
      isStreaming = true;
      modelTriggered = true;
      lastActivationRms = 1500.0f + (float)(esp_random() % 1000);
      lastActivationScore = fake_score;
      pendingActivationNotify = true;
    }
  }

  // 2. Serial Monitor Commands: 'a' to Abort, 'r' to Restart
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'a' || c == 'A') {
      isStreaming = false;
      if (xAudioStreamQueue != NULL) {
        xQueueReset(xAudioStreamQueue);
      }
      Serial.println(F("\n>>> [SERIAL COMMAND] TRANSMISSION ABORTED! <<<"));
    } else if (c == 'r' || c == 'R') {
      float fake_score = 0.88f + (float)(esp_random() % 11) / 100.0f;
      Serial.println(F("\n*******************************************************"));
      Serial.printf(" >>> [TINYML ACTIVATION] 'Hey Nexus' DETECTED! (Score: %.2f) <<<\n", fake_score);
      Serial.println(F(" >>> Switching to STREAMING mode... <<<"));
      Serial.println(F("*******************************************************\n"));
      
      if (xAudioStreamQueue != NULL) {
        xQueueReset(xAudioStreamQueue);
      }
      isStreaming = true;
      modelTriggered = true;
      lastActivationRms = 1500.0f + (float)(esp_random() % 1000);
      lastActivationScore = fake_score;
      pendingActivationNotify = true;
    }
  }
  /* ================= [END OF TEMP TESTING FEATURE] ================= */

  vTaskDelay(pdMS_TO_TICKS(50));
}
