#ifndef TVMGEN_DEFAULT_H_
#define TVMGEN_DEFAULT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tvmgen_default_inputs {
    void* serving_default_input_1_0;
};

struct tvmgen_default_outputs {
    void* StatefulPartitionedCall_0;
};

int32_t tvmgen_default_run(
    struct tvmgen_default_inputs*  inputs,
    struct tvmgen_default_outputs* outputs
);

#ifdef __cplusplus
}
#endif

#endif  // TVMGEN_DEFAULT_H_
