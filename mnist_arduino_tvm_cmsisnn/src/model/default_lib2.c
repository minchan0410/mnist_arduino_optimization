// tvm target: cmsis-nn 
#define TVM_EXPORTS
#include "../../src/standalone_crt/include/tvm/runtime/c_runtime_api.h"
#include "../../src/standalone_crt/include/tvm/runtime/c_backend_api.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "../../src/standalone_crt/include/dlpack/dlpack.h"
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "arm_nn_math_types.h"
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 4, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,28,28,1};
  cmsis_nn_dims filter_dims = {1,3,3,4};
  cmsis_nn_dims bias_dims = {1,1,1,4};
  cmsis_nn_dims output_dims = {1,14,14,4};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_fc_params fc_params = {128, 0, -128, activation};
  cmsis_nn_per_tensor_quant_params quant_params = {1909917322, -9};
  cmsis_nn_dims input_dims = {1,1,1,784};
  cmsis_nn_dims filter_dims = {784,1,1,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,1,1,64};
  arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_fc_params fc_params = {128, 0, 46, activation};
  cmsis_nn_per_tensor_quant_params quant_params = {1335504374, -8};
  cmsis_nn_dims input_dims = {1,1,1,64};
  cmsis_nn_dims filter_dims = {64,1,1,27};
  cmsis_nn_dims bias_dims = {1,1,1,27};
  cmsis_nn_dims output_dims = {1,1,1,27};
  arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t* input_, int8_t* output_) {
  arm_softmax_s8(input_, 1, 27, 1853690880, 24, -124, output_);
  return 0;
}

