/*
 * crt_backend_api.c - 정적 워크스페이스 패치 적용됨
 * malloc 대신 고정 버퍼 사용 (Nano 33 BLE 안정성)
 */
#include <stdint.h>
#include <stddef.h>

static uint8_t g_workspace[39616] __attribute__((aligned(4)));
static size_t  g_workspace_used = 0;

void* TVMBackendAllocWorkspace(int device_type, int device_id,
                               uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
    (void)device_type; (void)device_id;
    (void)dtype_code_hint; (void)dtype_bits_hint;

    size_t aligned = (g_workspace_used + 3U) & ~(size_t)3U;
    if (aligned + (size_t)nbytes > 39616) return (void*)0;

    void* ptr = &g_workspace[aligned];
    g_workspace_used = aligned + (size_t)nbytes;
    return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
    (void)device_type; (void)device_id; (void)ptr;
    g_workspace_used = 0;
    return 0;
}
