#ifndef TVM_RUNTIME_CRT_MODULE_H_
#define TVM_RUNTIME_CRT_MODULE_H_

#include <stdint.h>
#include "../c_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TVM_DLL
#define TVM_DLL
#endif

typedef int32_t (*TVMBackendPackedCFunc)(
    TVMValue* args, int* type_code, int num_args,
    TVMValue* out_value, int* out_type_code, void* resource_handle
);

typedef struct TVMFuncRegistry {
    const char*                  names;
    const TVMBackendPackedCFunc* funcs;
} TVMFuncRegistry;

typedef struct TVMModule {
    const TVMFuncRegistry* registry;
} TVMModule;

const TVMModule* TVMSystemLibEntryPoint(void);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_MODULE_H_
