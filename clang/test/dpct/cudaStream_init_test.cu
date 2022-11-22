// RUN: dpct --format-range=none -out-root %T/cudaStream_init_test %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaStream_init_test/cudaStream_init_test.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>

struct a{
  // CHECK: dpct::queue_ptr s9;
  cudaStream_t s9;
  
  // CHECK: dpct::queue_ptr s10 = &dpct::get_default_queue(), s11, s12{&dpct::get_default_queue()};
  cudaStream_t s10 = 0, s11, s12{0};
  // CHECK: dpct::queue_ptr sarr[2] = {&dpct::get_default_queue(), &dpct::get_default_queue()};
  cudaStream_t sarr[2] = {0, NULL};

  // CHECK: a() : s9(&dpct::get_default_queue()) {}
  a() : s9(0) {}
};

struct a2{
  // CHECK: dpct::queue_ptr s9;
  CUstream s9;
  
  // CHECK: dpct::queue_ptr s10 = &dpct::get_default_queue(), s11, s12{&dpct::get_default_queue()};
  CUstream s10 = 0, s11, s12{0};
  // CHECK: dpct::queue_ptr sarr[2] = {&dpct::get_default_queue(), &dpct::get_default_queue()};
  CUstream sarr[2] = {0, NULL};

  // CHECK: a2() : s9(&dpct::get_default_queue()) {}
  a2() : s9(0) {}
};

struct b {
  // CHECK: b(dpct::queue_ptr s = &dpct::get_default_queue()) {}
  b(cudaStream_t s = 0) {}
};

struct b2 {
  // CHECK: b2(dpct::queue_ptr s = &dpct::get_default_queue()) {}
  b2(CUstream s = 0) {}
}

void foo(){
  // CHECK: dpct::queue_ptr s13 = &dpct::get_default_queue();
  cudaStream_t s13 = 0;
}

void foo2(){
  // CHECK: dpct::queue_ptr s13 = &dpct::get_default_queue();
  CUstream s13 = 0;
}

void my_synchronize(cudaStream_t s) {
  cudaStreamSynchronize(s);
}

void my_synchronize1(CUstream s) {
  cudaStreamSynchronize(s);
}

int main(){
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: dpct::queue_ptr s0, s1{&q_ct1};
  cudaStream_t  s0, s1{0};

  // CHECK: dpct::queue_ptr s2{&q_ct1};
  cudaStream_t s2{0};

  // CHECK: s0 = dev_ct1.create_queue();
  cudaStreamCreate(&s0);

  // CHECK: dpct::queue_ptr s3(&q_ct1);
  cudaStream_t s3(0);

  // CHECK: dpct::queue_ptr s4 = &q_ct1;
  cudaStream_t s4 = 0;

  // CHECK: dpct::queue_ptr s5, s6(&q_ct1), s7 = &q_ct1;
  cudaStream_t s5, s6(0), s7 = 0;

  // CHECK: dpct::queue_ptr s8 =  &q_ct1;
  cudaStream_t s8 = NULL;

  // CHECK: dpct::queue_ptr sarr[2] = {&q_ct1, &q_ct1};
  cudaStream_t sarr[2] = {0, NULL};

  // CHECK: my_synchronize(&q_ct1);
  my_synchronize(0);

  // CHECK: (q_ct1.wait(), 0), my_synchronize(&q_ct1);
  cudaStreamSynchronize(0), my_synchronize(0);

  // CHECK: [](dpct::queue_ptr) {}(&q_ct1);
  [](cudaStream_t) {}(0);
  // CHECK: [](const dpct::queue_ptr) {}(&q_ct1);
  [](const cudaStream_t) {}(0);
  // CHECK: [](const dpct::queue_ptr&) {}(&q_ct1);
  [](const cudaStream_t&) {}(0);
  // CHECK: [](dpct::queue_ptr&&) {}(&q_ct1);
  [](cudaStream_t&&) {}(0);
}

int main2(){
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: dpct::queue_ptr s0, s1{&q_ct1};
  CUstream  s0, s1{0};

  // CHECK: dpct::queue_ptr s2{&q_ct1};
  CUstream s2{0};

  // CHECK: s0 = dev_ct1.create_queue();
  cudaStreamCreate(&s0);

  // CHECK: dpct::queue_ptr s3(&q_ct1);
  CUstream s3(0);

  // CHECK: dpct::queue_ptr s4 = &q_ct1;
  CUstream s4 = 0;

  // CHECK: dpct::queue_ptr s5, s6(&q_ct1), s7 = &q_ct1;
  CUstream s5, s6(0), s7 = 0;

  // CHECK: dpct::queue_ptr s8 =  &q_ct1;
  CUstream s8 = NULL;

  // CHECK: dpct::queue_ptr sarr[2] = {&q_ct1, &q_ct1};
  CUstream sarr[2] = {0, NULL};

  // CHECK: my_synchronize1(&q_ct1);
  my_synchronize1(0);

  // CHECK: (q_ct1.wait(), 0), my_synchronize1(&q_ct1);
  cuStreamSynchronize(0), my_synchronize1(0);

  // CHECK: [](dpct::queue_ptr) {}(&q_ct1);
  [](CUstream) {}(0);
  // CHECK: [](const dpct::queue_ptr) {}(&q_ct1);
  [](const CUstream) {}(0);
  // CHECK: [](const dpct::queue_ptr&) {}(&q_ct1);
  [](const CUstream&) {}(0);
  // CHECK: [](dpct::queue_ptr&&) {}(&q_ct1);
  [](CUstream&&) {}(0);
}

