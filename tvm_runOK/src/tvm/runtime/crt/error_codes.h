#ifndef TVM_RUNTIME_CRT_ERROR_CODES_H_
#define TVM_RUNTIME_CRT_ERROR_CODES_H_

#include <stdint.h>

typedef int32_t tvm_crt_error_t;

#define kTvmErrorNoError                    ((tvm_crt_error_t)0x00000000)
#define kTvmErrorFunctionCallInvalidArg     ((tvm_crt_error_t)0x00000008)
#define kTvmErrorFunctionCallFail           ((tvm_crt_error_t)0x00000009)

#endif  /* TVM_RUNTIME_CRT_ERROR_CODES_H_ */
