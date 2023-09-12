// RUN: dpct --format-range=none --optimize-migration -out-root %T/memcpy_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_optimization/memcpy_optimization.dp.cpp %s
#include <cuda_runtime.h>
#include <iostream>

int main(){

    int *dev_a, *dev_b;
    int *host_c, *host_d, *host_e, *host_f;
    bool sync;
    cudaMalloc(&dev_a, sizeof(int));
    cudaMalloc(&dev_a, sizeof(int));
    host_c = (int *)malloc(sizeof(int));
    host_d = (int *)malloc(sizeof(int));
    host_e = (int *)malloc(sizeof(int));
    host_f = (int *)malloc(sizeof(int));
//CHECK:  /*
//CHECK:  DPCT1114:{{[0-9]+}}: cudaMemcpy is migrated to asynchronization memcpy, assuming in the original code the source host memory is pageable memory. If the memory is not pageable, call wait() on event return by memcpy API to ensure synchronization behavior.
//CHECK:  */
//CHECK:  q_ct1.memcpy(dev_a, dev_b, sizeof(int));
//CHECK:  q_ct1.memcpy(dev_a, dev_b, sizeof(int));
    cudaMemcpy(dev_a, dev_b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a, dev_b, sizeof(int), cudaMemcpyDeviceToDevice);

//CHECK:  if(DPCT_CHECK_ERROR(q_ct1.memcpy(dev_a, host_e, 10))) {
//CHECK:    std::cout << "failed" << std::endl;
//CHECK:  }
    if(cudaMemcpy(dev_a, host_e, 10, cudaMemcpyHostToDevice)) {
        std::cout << "failed" << std::endl;
    }

//CHECK:  for(int i = 0; i < 10; i++) {
//CHECK:    int src = i;
//CHECK:    q_ct1.memcpy(dev_a, &src, sizeof(int)).wait();
//CHECK:  }
    for(int i = 0; i < 10; i++) {
        int src = i;
        cudaMemcpy(dev_a, &src, sizeof(int), cudaMemcpyHostToDevice);
    }

//CHECK:  q_ct1.memcpy(dev_a, host_c, sizeof(int)).wait();
//CHECK:  free(host_c);
    cudaMemcpy(dev_a, host_c, sizeof(int), cudaMemcpyHostToDevice);
    free(host_c);
    
//CHECK:  q_ct1.memcpy(dev_a, host_d, sizeof(int));
//CHECK:  dev_ct1.queues_wait_and_throw();
//CHECK:  free(host_d);
    cudaMemcpy(dev_a, host_d, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(host_d);

//CHECK:  q_ct1.memcpy(dev_a, host_f, sizeof(int)).wait();
//CHECK:  if(sync) {
//CHECK:    dev_ct1.queues_wait_and_throw();
//CHECK:  }
    cudaMemcpy(dev_a, host_f, sizeof(int), cudaMemcpyHostToDevice);
    if(sync) {
      cudaDeviceSynchronize();
    }
    free(host_f);
    return 0;
}