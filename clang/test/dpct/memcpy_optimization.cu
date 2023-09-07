// RUN: dpct --format-range=none --optimize-migration -out-root %T/memcpy_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_optimization/memcpy_optimization.dp.cpp %s
#include <cuda_runtime.h>
#include <iostream>
int main(){

    float *a, *b, *c, *d, *e, *f;
    bool sync;
    c = (float *)malloc(10);
    d = (float *)malloc(10);
//CHECK:  /*
//CHECK:  DPCT1114:{{[0-9]+}}: cudaMemcpy is migrated to asynchronization memcpy, assuming in the original code the source host memory is pageable memory. If the memory is not pageable, call wait() on event return by memcpy API to ensure synchronization behavior.
//CHECK:  */
//CHECK:  q_ct1.memcpy(a, b, 10);
//CHECK:  q_ct1.memcpy(a, b, 10);
    cudaMemcpy(a, b, 10, cudaMemcpyHostToDevice);
    cudaMemcpy(a, b, 10, cudaMemcpyDeviceToDevice);

//CHECK:  for(int i = 0; i < 10; i++) {
//CHECK:    int src;
//CHECK:    q_ct1.memcpy(a, &src, 10).wait();
//CHECK:  }
    for(int i = 0; i < 10; i++) {
      int src;
      cudaMemcpy(a, &src, 10, cudaMemcpyHostToDevice);
    }

//CHECK:  q_ct1.memcpy(a, c, 10).wait();
//CHECK:  free(c);
    cudaMemcpy(a, c, 10, cudaMemcpyHostToDevice);
    free(c);

//CHECK:  q_ct1.memcpy(a, d, 10);
//CHECK:  dev_ct1.queues_wait_and_throw();
//CHECK:  free(d);
    cudaMemcpy(a, d, 10, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(d);

//CHECK:  if(DPCT_CHECK_ERROR(q_ct1.memcpy(a, e, 10))) {
//CHECK:    std::cout << "failed" << std::endl;
//CHECK:  }
    if(cudaMemcpy(a, e, 10, cudaMemcpyHostToDevice)) {
        std::cout << "failed" << std::endl;
    }

//CHECK:  q_ct1.memcpy(a, f, 10).wait();
//CHECK:  if(sync) {
//CHECK:    dev_ct1.queues_wait_and_throw();
//CHECK:  }
    free(f);
    cudaMemcpy(a, f, 10, cudaMemcpyHostToDevice);
    if(sync) {
      cudaDeviceSynchronize();
    }
    free(f);

    return 0;
}