// RUN: dpct --format-range=none --optimize-migration -out-root %T/memcpy_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_optimization/memcpy_optimization.dp.cpp %s
#include <cuda_runtime.h>

int main(){

    float *a, *b;
//CHECK:  /*
//CHECK:  DPCT1114:{{[0-9]+}}: cudaMemcpy is migrated to asynchronization memcpy, assuming in original code the source host memory is pageable memory. If this assumption doesn't hold, please append '.wait()' after memcpy to synchronize.
//CHECK:  */
//CHECK:  q_ct1.memcpy(a, b, 10);
//CHECK:  q_ct1.memcpy(a, b, 10);
    cudaMemcpy(a, b, 10, cudaMemcpyHostToDevice);
    cudaMemcpy(a, b, 10, cudaMemcpyDeviceToDevice);

    return 0;
}