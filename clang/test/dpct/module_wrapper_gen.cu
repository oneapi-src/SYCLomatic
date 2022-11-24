// RUN: dpct --extra-arg="--ptx" \
// RUN:      --out-root=%T/module_wrapper_gen \
// RUN:      --cuda-include-path="%cuda-path/include" %s
// RUN: FileCheck %s --input-file=%T/module_wrapper_gen/module_wrapper_gen.dp.cpp

// START

__device__ float2 operator+(float2 a, float2 b) {
  return float2{a.x - b.x, a.y - b.y};
}

extern "C" __device__ void externCNonKernel() {}

__device__ void deviceFun() {}

__global__ void nonExternCKernel() {}

extern "C" __global__ void exampleKernel() {
  float2 x;
  float2 y;
  float2 z = x + y;
}

// END

// Only one _wrapper should be generated.
// CHECK:     // START
// CHECK:     _wrapper
// CHECK-NOT: _wrapper
// CHECK:     // END
