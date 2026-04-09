#ifndef TVM_RUNTIME_C_BACKEND_API_H_
#define TVM_RUNTIME_C_BACKEND_API_H_

#include "c_runtime_api.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TVM_DLL
#define TVM_DLL
#endif

TVM_DLL void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                       uint64_t nbytes, int dtype_code_hint,
                                       int dtype_bits_hint);
TVM_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_C_BACKEND_API_H_
