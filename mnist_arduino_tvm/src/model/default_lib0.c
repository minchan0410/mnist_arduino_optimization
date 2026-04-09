#include "../../src/standalone_crt/include/tvm/runtime/c_runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(void* serving_default_input_1_0,void* output0);
int32_t tvmgen_default_run(void* args, void* type_code, int num_args, void* out_value, void* out_type_code, void* resource_handle) {
return tvmgen_default___tvm_main__(((DLTensor*)(((TVMValue*)args)[0].v_handle))[0].data,((DLTensor*)(((TVMValue*)args)[1].v_handle))[0].data);
}
#ifdef __cplusplus
}
#endif
;