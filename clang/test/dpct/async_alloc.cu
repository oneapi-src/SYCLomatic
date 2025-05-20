// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// RUN: dpct --format-range=none --out-root %T/async_alloc %s --cuda-include-path="%cuda-path/include" --use-experimental-features=async_alloc
// RUN: FileCheck --match-full-lines --input-file %T/async_alloc/async_alloc.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/async_alloc/async_alloc.dp.cpp -o %T/async_alloc/async_alloc.dp.o %}

// CHECK: #include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

void foo_1(float *f, cudaStream_t hStream) {
  // CHECK: cudaMemPool_t memPool;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMallocAsync is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMallocAsync(&f, 1024, memPool, hStream);
  // CHECK-NEXT: f = sycl::ext::oneapi::experimental::async_malloc(*hStream, sycl::usm::alloc::device, 1024);
  // CHECK-NEXT: sycl::ext::oneapi::experimental::async_free(*hStream, f);
#ifndef NO_BUILD_TEST
  cudaMemPool_t memPool;
  cudaMallocAsync(&f, 1024, memPool, hStream);
#endif
  cudaMallocAsync(&f, 1024, hStream);
  cudaFreeAsync(f, hStream);
}
