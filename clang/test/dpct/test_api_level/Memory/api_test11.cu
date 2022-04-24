// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test11_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test11_out

// CHECK: 7
// TEST_FEATURE: Memory_pitched_data_set_data_ptr
// TEST_FEATURE: Memory_pitched_data_set_pitch
// TEST_FEATURE: Memory_pitched_data_set_x
// TEST_FEATURE: Memory_pitched_data_set_y

int main() {
  cudaPitchedPtr p1;
  float* f_A;
  p1.ptr = f_A;
  p1.pitch = 1;
  p1.xsize = 1;
  p1.ysize = 1;
  return 0;
}
