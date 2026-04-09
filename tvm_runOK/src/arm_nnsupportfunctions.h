#ifndef ARM_NNSUPPORTFUNCTIONS_H
#define ARM_NNSUPPORTFUNCTIONS_H

#include <stdint.h>

/* Force-inline attribute (from CMSIS cmsis_compiler.h) */
#ifndef __STATIC_FORCEINLINE
  #ifdef __GNUC__
    #define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
  #else
    #define __STATIC_FORCEINLINE static inline
  #endif
#endif

/*
 * ARM Cortex-M4 DSP intrinsics used by TVM CMSIS-NN generated kernels.
 * These map directly to hardware instructions available on ARMv7E-M (Cortex-M4).
 */
#ifdef __cplusplus
extern "C" {
#endif

/* Signed Multiply Accumulate Dual: result = op3 + (op1[15:0]*op2[15:0]) + (op1[31:16]*op2[31:16]) */
__STATIC_FORCEINLINE int32_t __SMLAD(int32_t op1, int32_t op2, int32_t op3) {
    int32_t result;
    __asm volatile ("smlad %0, %1, %2, %3"
                    : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}

/* Signed Subtract 8-bit (sets GE flags): each byte of result = op1[byte] - op2[byte] */
__STATIC_FORCEINLINE int32_t __SSUB8(int32_t op1, int32_t op2) {
    int32_t result;
    __asm volatile ("ssub8 %0, %1, %2"
                    : "=r"(result) : "r"(op1), "r"(op2));
    return result;
}

/* Select bytes using GE flags: each byte selected from op1 or op2 based on GE bit */
__STATIC_FORCEINLINE uint32_t __SEL(uint32_t op1, uint32_t op2) {
    uint32_t result;
    __asm volatile ("sel %0, %1, %2"
                    : "=r"(result) : "r"(op1), "r"(op2));
    return result;
}

#ifdef __cplusplus
}
#endif

#endif /* ARM_NNSUPPORTFUNCTIONS_H */
