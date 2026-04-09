#ifndef TVM_RUNTIME_C_RUNTIME_API_H_
#define TVM_RUNTIME_C_RUNTIME_API_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TVM_DLL
#define TVM_DLL
#endif

typedef union {
    int64_t   v_int64;
    double    v_float64;
    void*     v_handle;
    const char* v_str;
    int       v_type;
    int       v_device;
} TVMValue;

typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct { int device_type; int device_id; }             DLDevice;

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_C_RUNTIME_API_H_
