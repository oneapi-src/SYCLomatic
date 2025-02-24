// UNSUPPORTED: system-windows
// RUN: cat %S/cuda.json > %T/cuda.json
// RUN: cat %S/sycl.json > %T/sycl.json
// RUN: cat %S/cuda.bin > %T/cuda.bin
// RUN: cat %S/sycl.bin > %T/sycl.bin
// RUN: cd %T
// RUN: dpct --codepin-report --instrumented-cuda-log cuda.json --instrumented-sycl-log sycl.json || true

// RUN: cat %S/CodePin_Report_ref.csv > %T/CodePin_Report_Expected.csv
// RUN: cat %T/CodePin_Report.csv >> %T/CodePin_Report_Expected.csv

// RUN: FileCheck --match-full-lines --input-file %T/CodePin_Report_Expected.csv %T/CodePin_Report_Expected.csv

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "codepin_autogen_util.hpp"
#include <dpct/codepin/codepin.hpp>
#include <iostream>

#define CUDA_CALL(func)                                                      \
  {                                                                          \
    dpct::err0 err = (func);                                                 \
    if (err != 0) {                                                          \
      std::cerr << "CUDA error calling \"" #func "\", error code is " << err \
                << std::endl;                                                \
      exit(1);                                                               \
    }                                                                        \
  }

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float host_data[10];
  for (int i = 0; i < 10; ++i) {
    host_data[i] = static_cast<float>(i + 1);
  }

  float *device_data;
  CUDA_CALL(DPCT_CHECK_ERROR(device_data = sycl::malloc_device<float>(10, q_ct1)));
  dpctexp::codepin::set_ptr_size_map(*((void **)&device_data), 10 * sizeof(float));

  CUDA_CALL(DPCT_CHECK_ERROR(q_ct1.memcpy(device_data, host_data, 10 * sizeof(float)).wait()));
  dpctexp::codepin::set_ptr_size_map(device_data, 10 * sizeof(float));

  for (int i = 0; i < 10; ++i) {
    std::cout << host_data[i] << " ";
  }
  std::cout << std::endl;

  CUDA_CALL(DPCT_CHECK_ERROR(dpct::dpct_free(device_data, q_ct1)));
  dpctexp::codepin::gen_prolog_API_CP(
      "test_gen.cu:40:5",
      &dpct::get_in_order_queue(), "device_data", device_data);
  for (int i = 0; i < 10; ++i) {
    host_data[i] = static_cast<float>(i);
  }
  CUDA_CALL(DPCT_CHECK_ERROR(q_ct1.memcpy(device_data, host_data, 10 * sizeof(float)).wait()));
  dpctexp::codepin::set_ptr_size_map(device_data, 10 * sizeof(float));
  dpctexp::codepin::gen_epilog_API_CP(
      "test_gen.cu:40:5",
      &dpct::get_in_order_queue(), "device_data", device_data);
  return 0;
}
