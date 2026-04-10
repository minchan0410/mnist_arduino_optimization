/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "mnist_model_data.h" // model
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

class ArduinoMillisProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxEvents = 32;

  uint32_t BeginEvent(const char* tag) override {
    uint32_t idx = num_events_ % kMaxEvents;
    tags_[idx] = tag;
    start_ms_[idx] = millis();
    num_events_++;
    return idx;
  }

  void EndEvent(uint32_t event_handle) override {
    unsigned long elapsed = millis() - start_ms_[event_handle];
    Serial.print("[PROF] ");
    Serial.print(tags_[event_handle]);
    Serial.print(": ");
    Serial.print(elapsed);
    Serial.println(" ms");
  }

 private:
  const char* tags_[kMaxEvents];
  unsigned long start_ms_[kMaxEvents];
  uint32_t num_events_ = 0;
};

// EMNIST(Upper letter)
const int kInputTensorSize = 1 * 784;
const int kNumClass = 27;
const int kAccQueueSize = 100;
const int kGroundTruth = 23; // W

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  ArduinoMillisProfiler profiler;

  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.

  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 100 * 1024;
  // Keep aligned to 16 bytes for CMSIS
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
  Serial.begin(9600);
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(emnist_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize,
      nullptr, &profiler);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  int8_t x_test[kInputTensorSize]{-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -108, -14, -3, -46, -91, -91, -89, -13, 29, -34, -107, -118, -96, -91, -91, -91, -91, -91, -91, -91, -46, -3, -14, -108, -128, -128, -128, -125, -19, 117, 121, 105, 89, 89, 89, 117, 123, 106, 44, 14, 75, 89, 89, 89, 89, 89, 89, 89, 105, 121, 117, -19, -125, -128, -128, -124, -14, 125, 126, 126, 126, 126, 126, 126, 126, 126, 124, 123, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, -1, -124, -128, -128, -128, -83, 86, 105, 123, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 123, -15, -124, -128, -128, -128, -128, -123, -106, -46, 0, 32, 56, 89, 89, 90, 121, 126, 127, 127, 127, 127, 127, 126, 126, 119, 88, 42, -45, -121, -128, -128, -128, -128, -128, -128, -128, -126, -123, -112, -103, -91, -88, -78, 99, 126, 127, 127, 126, 126, 124, 117, 92, 31, -90, -107, -125, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -119, -51, 12, 119, 126, 126, 126, 126, 116, 49, -14, -78, -108, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -94, 47, 118, 126, 126, 126, 125, 93, -13, -82, -121, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -94, -44, 47, 123, 126, 126, 125, 106, 74, -37, -124, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -118, -49, 76, 105, 123, 126, 127, 127, 111, -12, -93, -121, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -108, -5, 92, 126, 126, 126, 127, 127, 127, 120, 37, -91, -107, -96, -119, -124, -124, -124, -124, -128, -128, -128, -128, -128, -128, -128, -128, -128, -28, 110, 126, 127, 127, 127, 127, 127, 127, 126, 125, 107, 106, 117, 94, 89, 89, 89, 75, -13, -96, -128, -128, -128, -128, -128, -128, -128, -95, 35, 116, 126, 126, 126, 127, 127, 127, 127, 126, 126, 126, 126, 126, 126, 126, 126, 126, 112, -17, -125, -128, -128, -128, -128, -128, -128, -127, -95, -2, 88, 105, 122, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 105, -46, -126, -128, -128, -128, -128, -128, -128, -128, -128, -128, -123, -106, -46, 123, 127, 127, 127, 127, 127, 127, 126, 126, 122, 105, 89, 75, -18, -110, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -118, 0, 126, 127, 127, 126, 126, 126, 126, 118, 80, 12, -46, -89, -96, -124, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -106, -33, 93, 126, 127, 126, 125, 106, 89, 87, -1, -82, -119, -126, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, 26, 104, 125, 126, 126, 124, 79, -33, -89, -91, -120, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -51, 119, 126, 127, 126, 114, 3, -119, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -124, -3, 126, 126, 124, 104, 3, -96, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -124, -15, 125, 118, 49, -19, -106, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -120, -17, -51, -120, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -126, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128};
  
  for(int i =0; i < kInputTensorSize; i++){
    input->data.int8[i] = x_test[i];
  }

  unsigned long t_start = micros();
  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }
  unsigned long t_end = micros();

  TfLiteTensor *output = interpreter->output(0);

  int predicated_class = 0;
  float max_score = -1;
  for (int i = 0; i < kNumClass; i++)
  {
    int8_t score = output->data.int8[i];
    if (score > max_score)
    {
      predicated_class = i;
      max_score = score;
    }
  }


  Serial.print("Predicted: ");
  Serial.print(predicated_class);
  Serial.print("  GT: ");
  Serial.print(kGroundTruth);
  Serial.print("  Inference: ");
  Serial.print(t_end - t_start);
  Serial.println(" us ");

  delay(1000);
}
