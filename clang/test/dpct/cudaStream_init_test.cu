// RUN: dpct --format-range=none --usm-level=none -out-root %T/cudaStream_init_test %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaStream_init_test/cudaStream_init_test.dp.cpp --match-full-lines %s

int main(){
  // CHECK: dpct::queue_ptr s0, s1{&dpct::get_default_queue()};
  cudaStream_t  s0, s1{0};

  // CHECK: dpct::queue_ptr s2{&dpct::get_default_queue()};
  cudaStream_t s2{0};

  // CHECK: s0 = dpct::get_current_device().create_queue();
  cudaStreamCreate(&s0);

  // CHECK: dpct::queue_ptr s3(&dpct::get_default_queue());
  cudaStream_t s3(0);
  // CHECK: dpct::queue_ptr s4 = &dpct::get_default_queue();
  cudaStream_t s4 = 0;
  // CHECK: dpct::queue_ptr s5, s6(&dpct::get_default_queue()), s7 = &dpct::get_default_queue();
  cudaStream_t s5, s6(0), s7 = 0;

}