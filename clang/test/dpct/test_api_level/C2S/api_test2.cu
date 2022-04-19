// RUN: c2s --format-range=none --sycl-named-lambda  --use-custom-helper=api -out-root %T/C2S/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/C2S/api_test2_out/MainSourceFiles.yaml | wc -l > %T/C2S/api_test2_out/count.txt
// RUN: cat %T/C2S/api_test2_out/include/c2s/c2s.hpp >> %T/C2S/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/C2S/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/C2S/api_test2_out

// CHECK: 17

// CHECK: template <class... Args> class c2s_kernel_name;
// CHECK-NEXT: template <int Arg> class c2s_kernel_scalar;

// TEST_FEATURE: C2S_c2s_named_lambda

__global__ void foo() {}

int main() {
  foo<<<1, 1>>>();
  return 0;
}
