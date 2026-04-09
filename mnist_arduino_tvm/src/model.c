/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "model.h"

#include "Arduino.h"
#include "standalone_crt/include/dlpack/dlpack.h"
#include "standalone_crt/include/tvm/runtime/crt/stack_allocator.h"
#include "standalone_crt/include/tvm/runtime/c_runtime_api.h"

// tvmgen_default___tvm_main__ 선언 (packed API, DLTensor 포인터를 TVMValue로 감싸서 전달)
extern int32_t tvmgen_default___tvm_main__(void* args, int32_t* arg_type_ids,
                                           int32_t num_args, void* out_ret_value,
                                           int32_t* out_ret_tcode, void* resource_handle);

// AOT memory array, stack allocator wants it aligned
static uint8_t g_aot_memory[WORKSPACE_SIZE]
    __attribute__((aligned(TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)));
tvm_workspace_t app_workspace;

// Blink code for debugging purposes
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: 0x%08x\n", error);
  for (;;) {
#ifdef LED_BUILTIN
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(750);
#endif
  }
}

void TVMLogf(const char* msg, ...) {}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

unsigned long g_utvm_start_time_micros;
int g_utvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 1;
  g_utvm_start_time_micros = micros();
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time_micros;
  *elapsed_time_seconds = ((double)g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i++) {
    buffer[i] = rand();
  }
  return kTvmErrorNoError;
}

void TVMInitialize() { StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE); }

void TVMExecute(void* input_data, void* output_data) {
  static int64_t input_shape[4]  = {1, 28, 28, 1};
  static int64_t output_shape[2] = {1, 27};

  DLTensor input_tensor = {
    .data        = input_data,
    .device      = {kDLCPU, 0},
    .ndim        = 4,
    .dtype       = {kDLInt, 8, 1},
    .shape       = input_shape,
    .strides     = NULL,
    .byte_offset = 0,
  };

  DLTensor output_tensor = {
    .data        = output_data,
    .device      = {kDLCPU, 0},
    .ndim        = 2,
    .dtype       = {kDLInt, 8, 1},
    .shape       = output_shape,
    .strides     = NULL,
    .byte_offset = 0,
  };

  TVMValue args[2];
  args[0].v_handle = &input_tensor;
  args[1].v_handle = &output_tensor;

  int32_t type_codes[2] = {7, 7};  // kTVMDLTensorHandle = 7
  TVMValue ret_val;
  int32_t ret_type_code = 0;

  int ret = tvmgen_default___tvm_main__(
      (void*)args, type_codes, 2, &ret_val, &ret_type_code, NULL);
  if (ret != 0) {
    TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
  }
}
