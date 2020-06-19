// RUN: dpcpp device_memory_usm.cpp -o device_memory_usm

#define DPCT_NAMED_LAMBDA
#include <CL/sycl.hpp>
#include "../include/dpct.hpp"

class TestStruct {
public:
  void test() {}
  template<class T> void testTemplate() {}
};

template<class T>
class TemplateStuct {
public:
  void test() {}
  template<class Ty> void testTemplate() {}
};

dpct::device_memory<volatile int, 0> d1_a(0);
dpct::device_memory<int, 1> d2_a(36);
dpct::device_memory<TemplateStuct<int>, 0> d3_a;
dpct::device_memory<TestStruct, 0> d4_a;
dpct::constant_memory<int, 1> c1_a(16);
dpct::constant_memory<int, 0> c2_a;
dpct::constant_memory<TemplateStuct<int>, 0> c3_a;
dpct::constant_memory<TestStruct, 0> c4_a;

dpct::constant_memory<int, 2> c_2d_a(sycl::range<2>(5, 3),
{{0, 10, 20},
{30, 40, 50},
{60, 70, 80},
{90, 100, 110},
{120, 130, 140}});
dpct::constant_memory<int, 2> c_2d_b(sycl::range<2>(3, 5),
{{0, 10, 20, 30, 40},
{50, 60, 70, 80, 90},
{100, 110, 120, 130, 140}});
dpct::constant_memory<int, 1> c_1d(sycl::range<1>(15),
                                   {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                    110, 120, 130, 140});

bool verify_init(int *data) {  
  for(auto i = 0; i < 15; ++i) {
    if (data[i] != i * 10)
      return false;
  }
  return true;
}

bool verify() {
  const int size = 15;
  auto size_bytes = 15 * sizeof(int);

  int h_result[15];
  dpct::dpct_memcpy(h_result, c_2d_a.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  dpct::dpct_memcpy(h_result, c_2d_b.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  dpct::dpct_memcpy(h_result, c_1d.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  return true;
}

void test4(TemplateStuct<int> *d3, TestStruct *d4) {
  d3->test();
  d3->testTemplate<int>();
  d4->test();
  d4->testTemplate<int>();
}

void test3(TemplateStuct<int> c3, TestStruct c4) {
  c3.test();
  c3.testTemplate<int>();
  c4.test();
  c4.testTemplate<int>();
}

void test2(volatile int &a) {
  a = 3;
}

void test1(volatile int *acc_d1, int *acc_d2, int *c1, int c2) {
  unsigned d_a = 1;
  *acc_d1 = 0;
  *acc_d2 = d_a;
  unsigned d_c = (unsigned)(*acc_d1);
  unsigned *d_d = (unsigned *)acc_d2;
  unsigned *d_e = (unsigned *)(acc_d2 + 5);
  int *d_f = acc_d2 - 6;
  test2(*acc_d1);
}

int main() try {
  dpct::get_default_queue().submit(
    [&](cl::sycl::handler &cgh) {
      auto d1_acc = d1_a.get_ptr();
      auto d2_acc = d2_a.get_ptr();
      auto c1_acc = c1_a.get_ptr();
      auto c2_acc = c2_a.get_ptr();
      cgh.parallel_for<dpct_kernel_name<class kernel_test>>(
        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
        [=] (cl::sycl::nd_item<3> item) {
          test1(d1_acc, d2_acc, c1_acc, *c2_acc);
        });
    });
  dpct::get_default_queue().submit(
    [&](cl::sycl::handler &cgh) {
      auto c3_acc = c3_a.get_ptr();
      auto c4_acc = c4_a.get_ptr();
      cgh.parallel_for<dpct_kernel_name<class kernel_test>>(
        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
        [=] (cl::sycl::nd_item<3> item) {
          test3(*c3_acc, *c4_acc);
        });
    });
  dpct::get_default_queue().submit(
    [&](cl::sycl::handler &cgh) {
      auto d3_acc = d3_a.get_ptr();
      auto d4_acc = d4_a.get_ptr();
      cgh.parallel_for<dpct_kernel_name<class kernel_test>>(
        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
        [=] (cl::sycl::nd_item<3> item) {
          test4(d3_acc, d4_acc);
        });
    });

  if (verify()) {
    printf("Init Constant Memory Success!\n");
  } else {
    printf("Init Constant Memory Fail!\n");
  }
  return 0;
}
catch(cl::sycl::exception const &exc){}
