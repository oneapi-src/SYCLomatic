// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_array.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_array.dp.cpp --match-full-lines %s

// CHECK: #include <array>
#include <cuda/std/array>

template <class T>
__host__ __device__ void
test(T *res) {
  // CHECK: std::array<T, 3> arr = {1, 2, 3.5};
  cuda::std::array<T, 3> arr = {1, 2, 3.5};
  // CHECK: *(res) = arr[0];
  *(res) = arr.at(0);
  // CHECK: *(res + 1) = arr[1];
  *(res + 1) = arr.at(1);
  // CHECK: *(res + 2) = arr[2];
  *(res + 2) = arr.at(2);
  *(res + 3) = *(arr.begin());
  *(res + 4) = arr.size();
}

__global__ void test_global(float *res) {
  test<float>(res);
}

int main(int, char **) {

  float *floatRes = (float *)malloc(5 * sizeof(float));
  test<float>(floatRes);
  // test<double>(doubleRes);
  float *hostRes = (float *)malloc(5 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 5 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 5, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0; i < 5; ++i) {
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
