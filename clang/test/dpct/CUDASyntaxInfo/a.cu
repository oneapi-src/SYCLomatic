// UNSUPPORTED: system-windows
// RUN: cd %S
// RUN: dpct --out-root %T -p . --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/MainSourceFiles.yaml
// RUN: %if build_lit %{icpx -c -fsycl %T/a.cpp.dp.cpp -o %T/a.cpp.dp.o %}
// RUN: %if build_lit %{icpx -c -fsycl %T/a.dp.cpp -o %T/a.dp.o %}

// CHECK: HasCUDASyntax:   true
// CHECK: HasCUDASyntax:   true

__host__ __device__ int test(int axis) {
  int a = 1;
#if !defined(__CUDA_ARCH__)
  cudaDeviceSynchronize();
  return 0;
#else
  return axis == 0 ? threadIdx.x : axis == 1 ? threadIdx.y : threadIdx.z;
#endif
  return a;
}
