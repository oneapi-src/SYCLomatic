// RUN: syclct -report-type=apis -report-file=%T/api-name-migrated_log -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --match-full-lines --input-file %T/api-name-migrated.sycl.cpp %s
// RUN: echo "// `perl -e 'print "CH","ECK"'`: API name, Migrated, Frequency" >%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaFree,true,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaFreeHost,false,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMalloc,true,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMalloc3D,false,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMallocHost,false,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMemcpy,true,2" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMemset,true,1" >>%T/api-name-migrated_log_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: make_cudaExtent,false,1" >>%T/api-name-migrated_log_check.txt
// RUN: cat %T/api-name-migrated_log.csv >>%T/api-name-migrated_log_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/api-name-migrated_log_check.txt %T/api-name-migrated_log_check.txt

#include <cuda_runtime.h>

void fooo() {
  size_t size = 10 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;

  size_t length = size * size * size;
  size_t bytes = length * sizeof(float);
  float *src;

  // CHECK: /*
  // CHECK-NEXT:SYCLCT1007:{{[0-9]+}}: cudaFreeHost: not support API, need manual porting.
  // CHECK-NEXT:*/
  cudaFreeHost(d_A);

  // CHECK: /*
  // CHECK-NEXT:SYCLCT1007:{{[0-9]+}}: cudaMallocHost: not support API, need manual porting.
  // CHECK-NEXT:*/
  cudaMallocHost(&src, bytes);

  struct cudaPitchedPtr srcGPU;

  // CHECK: /*
  // CHECK-NEXT:SYCLCT1007:{{[0-9]+}}: make_cudaExtent: not support API, need manual porting.
  // CHECK-NEXT:*/
  struct cudaExtent extent = make_cudaExtent(size * sizeof(float), size, size);

  // CHECK: /*
  // CHECK-NEXT:SYCLCT1007:{{[0-9]+}}: cudaMalloc3D: not support API, need manual porting.
  // CHECK-NEXT:*/
  cudaMalloc3D(&srcGPU, extent);
}

void cool() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  // CHECK: syclct::sycl_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK: syclct::sycl_memset((void*)(d_A), (int)(0xf), (size_t)(size));
  cudaMemset(d_A, 0xf, size);
  // CHECK: syclct::sycl_memcpy((void*)(d_A), (void*)(h_A), size, syclct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: syclct::sycl_memcpy((void*)(h_A), (void*)(d_A), size, syclct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK: syclct::sycl_free(d_A);
  cudaFree(d_A);
  free(h_A);
}
