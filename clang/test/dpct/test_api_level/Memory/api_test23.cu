// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test23_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test23_out

// CHECK: 55
// TEST_FEATURE: Memory_device_memory_get_access
// TEST_FEATURE: Memory_device_memory_init
// TEST_FEATURE: Memory_memory_region

__device__ float c[16][16];

__global__ void kernel() {
  c[0][0] = 1.0f;
}

int main() {
  kernel<<<1, 1>>>();
  return 0;
}
