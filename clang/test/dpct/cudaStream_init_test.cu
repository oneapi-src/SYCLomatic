// RUN: dpct --format-range=none -out-root %T/cudaStream_init_test %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaStream_init_test/cudaStream_init_test.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>

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

  // CHECK: dpct::queue_ptr s8 = &dpct::get_default_queue();
  cudaStream_t s8 = NULL;

  struct a{
    // CHECK: dpct::queue_ptr s9;
    cudaStream_t s9;
    // CHECK: dpct::queue_ptr s10 = &dpct::get_default_queue(), s11, s12{&dpct::get_default_queue()};
    cudaStream_t s10 = 0, s11, s12{0};
  };

}


