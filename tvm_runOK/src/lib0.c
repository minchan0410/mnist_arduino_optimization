#include "tvm/runtime/c_runtime_api.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "tvmgen_default.h"
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* serving_default_input_1_0_buffer_var, int8_t* StatefulPartitionedCall_0_buffer_var);
int32_t tvmgen_default_run(struct tvmgen_default_inputs* inputs, struct tvmgen_default_outputs* outputs) {
    return tvmgen_default___tvm_main__((int8_t*)inputs->serving_default_input_1_0, (int8_t*)outputs->StatefulPartitionedCall_0);
}
#ifdef __cplusplus
}
#endif
