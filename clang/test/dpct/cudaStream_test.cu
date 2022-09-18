// RUN: dpct --usm-level=none -out-root %T/cudaStream_test %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaStream_test/cudaStream_test.dp.cpp --match-full-lines %s

int main(){
  // CHECK: sycl::queue *s0, *s1{&dpct::get_default_queue()};
  cudaStream_t s0, s1{0};

  // CHECK: sycl::queue *s2{&dpct::get_default_queue()};
  cudaStream_t s2{0};

  // CHECK: s0 = dpct::get_current_device().create_queue();
  cudaStreamCreate(&s0);
}