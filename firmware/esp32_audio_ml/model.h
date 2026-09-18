#ifndef MODEL_H_
#define MODEL_H_

#include <stdint.h>

/* ====================================================================
 * TinyML Model Header (TensorFlow Lite for Microcontrollers)
 * ====================================================================
 * Instructions:
 * 1. Train or export your voice activator / keyword spotting model
 *    as a .tflite file (e.g. "model.tflite").
 * 2. Convert it into a C-array using one of the following methods:
 *
 *    Method A (Python script):
 *      python -c "open('model.h','w').write('const unsigned char model_data[] = {' + ','.join(hex(b) for b in open('model.tflite','rb').read()) + '};\nconst unsigned int model_data_len = ' + str(len(open('model.tflite','rb').read())) + ';')"
 *
 *    Method B (Linux / Git Bash / WSL xxd tool):
 *      xxd -i model.tflite > model.h
 *
 * 3. Replace the placeholder array below with your array.
 * ==================================================================== */

// Ensure 16-byte memory alignment required by TensorFlow Lite Micro SIMD / ESP-NN kernels
#if defined(__GNUC__) || defined(__clang__)
#define MODEL_ALIGNMENT __attribute__((aligned(16)))
#else
#define MODEL_ALIGNMENT alignas(16)
#endif

// Placeholder model byte array (Replace with your converted TFLite model data)
MODEL_ALIGNMENT const unsigned char model_data[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, // TFL3 Identifier
    0x00, 0x00, 0x12, 0x00, 0x1c, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
    0x18, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x18, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

const unsigned int model_data_len = sizeof(model_data);

// Convenient alias pointers for various TFLite Micro codebases
const unsigned char *g_model = model_data;
const unsigned int g_model_len = sizeof(model_data);

#endif // MODEL_H_
