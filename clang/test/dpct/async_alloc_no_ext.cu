// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// RUN: dpct --format-range=none --out-root %T/async_alloc_no_ext %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/async_alloc_no_ext/async_alloc_no_ext.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/async_alloc_no_ext/async_alloc_no_ext.dp.cpp -o %T/async_alloc_no_ext/async_alloc_no_ext.dp.o %}

void foo_1(float *f, cudaStream_t hStream) {
  // CHECK: cudaMemPool_t memPool;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMallocAsync is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMallocAsync(&f, 1024, memPool, hStream);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaMallocAsync is not supported, please try to remigrate with option: --use-experimental-features=virtual_mem.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMallocAsync(&f, 1024, hStream);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaFreeAsync is not supported, please try to remigrate with option: --use-experimental-features=virtual_mem.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaFreeAsync(f, hStream);
#ifndef NO_BUILD_TEST
  cudaMemPool_t memPool;
  cudaMallocAsync(&f, 1024, memPool, hStream);
  cudaMallocAsync(&f, 1024, hStream);
  cudaFreeAsync(f, hStream);
#endif
}
