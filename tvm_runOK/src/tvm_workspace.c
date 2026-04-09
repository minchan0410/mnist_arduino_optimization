#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void* TVMBackendAllocWorkspace(int device_type, int device_id,
                               unsigned long long nbytes,
                               int dtype_code_hint, int dtype_bits_hint) {
    (void)device_type;
    (void)device_id;
    (void)dtype_code_hint;
    (void)dtype_bits_hint;
    return malloc((size_t)nbytes);
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
    (void)device_type;
    (void)device_id;
    free(ptr);
    return 0;
}

#ifdef __cplusplus
}
#endif
