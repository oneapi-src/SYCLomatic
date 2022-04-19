// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test12_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test12_out

// CHECK: 7
// TEST_FEATURE: Memory_pitched_data_get_data_ptr
// TEST_FEATURE: Memory_pitched_data_get_pitch
// TEST_FEATURE: Memory_pitched_data_get_x
// TEST_FEATURE: Memory_pitched_data_get_y

int main() {
  cudaPitchedPtr p1;
  void* f_A;
  f_A = p1.ptr;
  size_t a;
  a = p1.pitch;
  a = p1.xsize;
  a = p1.ysize;
  return 0;
}
