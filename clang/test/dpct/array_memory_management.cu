// RUN: dpct --format-range=none -out-root %T/array_memory_management %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/array_memory_management/array_memory_management.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>

#define CHECK_ERR(x) do {                         \
  cudaError_t err = x;                            \
  if (err != cudaSuccess) {                          \
    return; \
  }                                                  \
} while(0)

void checkError(cudaError_t err) {
}

class C {
public:
  int *data{nullptr};
};

void foo() {
  int *data;
  size_t width, height, depth, pitch, woffset, hoffset;
  C c;
  // CHECK: dpct::queue_ptr s;
  // CHECK-NEXT: dpct::image_matrix_p a1;
  // CHECK-NEXT: dpct::image_matrix* a2;
  // CHECK-NEXT: dpct::err0 err;
  // CHECK-NEXT: sycl::range<3> extent{0, 0, 0};
  // CHECK-NEXT: dpct::image_channel channel;
  cudaStream_t s;
  cudaArray_t a1;
  cudaArray* a2;
  cudaError_t err;
  cudaExtent extent;
  cudaChannelFormatDesc channel;

  // CHECK: a1 = new dpct::image_matrix(channel, sycl::range<2>(width, height));
  cudaMallocArray(&a1, &channel, width, height);

  // CHECK: a1 = new dpct::image_matrix(channel, sycl::range<2>(width, height));
  cudaMallocArray(&a1, &channel, width, height, 0);

  // CHECK: a1 = new dpct::image_matrix(channel, extent);
  cudaMalloc3DArray(&a1, &channel, extent);

  // CHECK: a1 = new dpct::image_matrix(channel, extent);
  cudaMalloc3DArray(&a1, &channel, extent, 0);

  // CHECK: dpct::dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DFromArray(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, 0);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1), dpct::automatic, *s);
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, s);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DToArray(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, 0);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1), dpct::automatic, *s);
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, s);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height);

  // CHECK: dpct::dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyFromArray(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, 0);

  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1), dpct::automatic, *s);
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, s);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyToArray(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(((dpct::image_matrix *)c.data)->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyToArray((cudaArray *)c.data, woffset, hoffset, data, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, 0);

  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1), dpct::automatic, *s);
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, s);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  cudaMemcpyArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width);

  // CHECK: delete a1;
  cudaFreeArray(a1);
  // CHECK: delete a2;
  cudaFreeArray(a2);

  // CHECK:  err = DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, sycl::range<2>(width, height)));
  err = cudaMallocArray(&a1, &channel, width, height);
  // CHECK:  checkError(DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, sycl::range<2>(width, height))));
  checkError(cudaMallocArray(&a1, &channel, width, height));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, sycl::range<2>(width, height))));
  CHECK_ERR(cudaMallocArray(&a1, &channel, width, height));

  // CHECK:  err = DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, extent));
  err = cudaMalloc3DArray(&a1, &channel, extent);
  // CHECK:  checkError(DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, extent)));
  checkError(cudaMalloc3DArray(&a1, &channel, extent));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(a1 = new dpct::image_matrix(channel, extent)));
  CHECK_ERR(cudaMalloc3DArray(&a1, &channel, extent));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)));
  err = cudaMemcpy2DFromArray(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  checkError(cudaMemcpy2DFromArray(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  CHECK_ERR(cudaMemcpy2DFromArray(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1), dpct::automatic, *s));
  err = cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, s);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1), dpct::automatic, *s)));
  checkError(cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, s));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1), dpct::automatic, *s)));
  CHECK_ERR(cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, s));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)));
  err = cudaMemcpy2DToArray(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  checkError(cudaMemcpy2DToArray(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  CHECK_ERR(cudaMemcpy2DToArray(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)));
  err = cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, 0);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  checkError(cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, 0));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  CHECK_ERR(cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, 0));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)));
  err = cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  checkError(cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  CHECK_ERR(cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  err = cudaMemcpyFromArray(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  checkError(cudaMemcpyFromArray(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  CHECK_ERR(cudaMemcpyFromArray(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  err = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  checkError(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  CHECK_ERR(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)));
  err = cudaMemcpyToArray(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  checkError(cudaMemcpyToArray(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  CHECK_ERR(cudaMemcpyToArray(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)));
  err = cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  checkError(cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  CHECK_ERR(cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  err = cudaMemcpyArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  checkError(cudaMemcpyArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, cudaMemcpyDeviceToHost));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), a2->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  CHECK_ERR(cudaMemcpyArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, cudaMemcpyDeviceToHost));

  // CHECK:  err = DPCT_CHECK_ERROR(delete a1);
  err = cudaFreeArray(a1);
  // CHECK:  checkError(DPCT_CHECK_ERROR(delete a1));
  checkError(cudaFreeArray(a1));
  // CHECK:  CHECK_ERR(DPCT_CHECK_ERROR(delete a1));
  CHECK_ERR(cudaFreeArray(a1));
}

