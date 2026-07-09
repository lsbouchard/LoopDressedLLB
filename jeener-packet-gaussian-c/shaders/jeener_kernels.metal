#include <metal_stdlib>
using namespace metal;

kernel void matmul16_kernel(const device float *A [[buffer(0)]],
                            const device float *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 16 || gid.y >= 16) return;
    float acc = 0.0f;
    for (uint k = 0; k < 16; ++k) {
        acc += A[gid.y * 16 + k] * B[k * 16 + gid.x];
    }
    C[gid.y * 16 + gid.x] = acc;
}
