// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test32_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test32_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test32_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test32_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test32_out

// CHECK: 28
// TEST_FEATURE: Memory_access_wrapper

__global__ void foo(float* f) {
}

int main() {
  float* f;
  cudaMalloc(&f, 8);
  f = f + 1;
  foo<<<1, 1>>>(f);
  return 0;
}