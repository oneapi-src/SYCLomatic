// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_tuple.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_tuple.dp.cpp --match-full-lines %s

// CHECK: #include <tuple>
#include <cuda/std/tuple>

template <class T>
__host__ __device__ void
test(T *res) {
  // CHECK: std::tuple<T, T, T> t = std::make_tuple(2.0, 3.0, 4.0);
  cuda::std::tuple<T, T, T> t = cuda::std::make_tuple(2.0, 3.0, 4.0);
  // CHECK: *(res) = std::get<0>(t);
  *(res) = cuda::std::get<0>(t);
  // CHECK: *(res + 1) = std::get<1>(t);
  *(res + 1) = cuda::std::get<1>(t);
  // CHECK: *(res + 2) = std::get<2>(t);
  *(res + 2) = cuda::std::get<2>(t);
}

__global__ void test_global(float *res) {
  test<float>(res);
}

int main(int, char **) {

  float *floatRes = (float *)malloc(3 * sizeof(float));
  test<float>(floatRes);
  // test<double>(doubleRes);
  float *hostRes = (float *)malloc(3 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 3 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 3, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0; i < 3; ++i) {
    if (hostRes[i] != floatRes[i]) {
      free(hostRes);
      free(floatRes);
      return 1;
    }
  }
  free(hostRes);
  free(floatRes);
  return 0;
}
