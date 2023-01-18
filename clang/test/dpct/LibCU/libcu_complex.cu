// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_complex.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_complex.dp.cpp --match-full-lines %s

#include <cuda/std/complex>

template <class T>
__host__ __device__ void
test(T *res) {
  // CHECK: std::complex<T> x(1.5, 2.5);
  cuda::std::complex<T> x(1.5, 2.5);
  // CHECK: std::complex<T> y(2.5, 3);
  cuda::std::complex<T> y(2.5, 3);
  T *a = (T *)&x;
  a[0] = 5;
  a[1] = 6;
  *(res) = x.real() * x.imag();
  // CHECK: std::complex<T> z = x / y;
  cuda::std::complex<T> z = x / y;
  *(res + 1) = z.real();
  *(res + 2) = z.imag();
  z = x + y;
  *(res + 3) = z.real();
  *(res + 4) = z.imag();
  z = x - y;
  *(res + 5) = z.real();
  *(res + 6) = z.imag();
  z = x * y;
  *(res + 7) = z.real();
  *(res + 8) = z.imag();
}

__global__ void test_global(float *res) {
  test<float>(res);
}

int main(int, char **) {

  float *floatRes = (float *)malloc(9 * sizeof(float));
  test<float>(floatRes);
  // test<double>(doubleRes);
  float *hostRes = (float *)malloc(9 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 9 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 9, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0; i < 9; ++i) {
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
