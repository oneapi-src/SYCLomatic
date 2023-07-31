// RUN: dpct --format-range=none --optimize-migration -out-root %T/memcpy_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_optimization/memcpy_optimization.dp.cpp %s
#include <cuda_runtime.h>

int main(){

    float *a, *b;
//CHECK:  /*
//CHECK:  DPCT1114:{{[0-9]+}}: There is no synchronize operation append after memcpy. This is based on the assumption that the original cudaMemcpy operations are asynchronous when data is transferred to the device and the memory involved is neither Unified Memory nor pinned memory. If this assumption doesn't hold, please append .wait() after memcpy for correct synchronization.
//CHECK:  */
    cudaMemcpy(a, b, 10, cudaMemcpyHostToDevice);
    return 0;
}