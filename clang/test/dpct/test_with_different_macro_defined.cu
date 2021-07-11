// RUN: mkdir %T/test_with_different_macro_defined_output
// RUN: cat %s > %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu
// RUN: dpct -format-range=none -out-root %T/test_with_different_macro_defined_output %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DFOO_1
// RUN: dpct -format-range=none -out-root %T/test_with_different_macro_defined_output %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DFOO_2
// RUN: FileCheck --input-file %T/test_with_different_macro_defined_output/test_with_different_macro_defined.dp.cpp --match-full-lines %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu

// RUN: cat %S/test_with_different_macro_defined_ref.txt > %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu
// RUN: dpct -format-range=none -out-root %T/test_with_different_macro_defined_output %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DFOO_1
// RUN: dpct -format-range=none -out-root %T/test_with_different_macro_defined_output %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DFOO_2

// RUN: cat %S/test_with_different_macro_defined_ref.txt > %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu
// RUN: echo " " >> %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu
// RUN: dpct -format-range=none -out-root %T/test_with_different_macro_defined_output %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DFOO_2
// RUN: FileCheck --input-file %T/test_with_different_macro_defined_output/test_with_different_macro_defined.dp.cpp --match-full-lines %T/test_with_different_macro_defined_output/test_with_different_macro_defined.cu
// RUN: rm %T/test_with_different_macro_defined_output -rf


__global__ void foo_1() {}
__global__ void foo_2() {}

int main() {
// CHECK: #ifdef FOO_1
// CHECK-NEXT:  dpct::get_default_queue().parallel_for(
// CHECK-NEXT:            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:              foo_1();
// CHECK-NEXT:            });
// CHECK-NEXT:#elif  defined(FOO_2)
// CHECK-NEXT:  dpct::get_default_queue().parallel_for(
// CHECK-NEXT:            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:              foo_2();
// CHECK-NEXT:            });
// CHECK-NEXT:#endif
#ifdef FOO_1
  foo_1<<<1, 1>>>();
#elif  defined(FOO_2)
  foo_2<<<1, 1>>>();
#endif
  return 0;
}
