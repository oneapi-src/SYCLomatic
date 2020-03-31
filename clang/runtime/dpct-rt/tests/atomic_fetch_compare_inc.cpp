//===--- atomic_fetch_compare_inc.cpp ---------------------*- cpp -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
// #build:  dpcpp  atomic_fetch_compare_inc.cpp
// #run: ./a.out
#define DPCT_NAMED_LAMBDA
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

void atomic_test_kernel(unsigned int  *ddata, cl::sycl::nd_item<3> item_ct1) {
  unsigned int tid = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  // add test
  dpct::atomic_fetch_compare_inc(ddata, 99);  //  1, 2, 3, ...,99, 0,; ...key=(256*64)%100=84;  
}

int main(int argc, char **argv) try {
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  int err = 0;

  unsigned int Hdata;
  unsigned int Hdata2;

  printf("atomic test \n");

  Hdata = 0;                      // add

  // allocate device memory for result
  unsigned int *Ddata;
  *((void **)&Ddata) = cl::sycl::malloc_device(sizeof(int), dpct::dev_mgr::instance().current_device(), dpct::get_default_queue().get_context());

  dpct::get_default_queue().memcpy((void*)(Ddata), (void*)(&Hdata), sizeof(int)).wait();

  {
    dpct::get_default_queue().submit(
      [&](cl::sycl::handler &cgh) {
        auto dpct_global_range = cl::sycl::range<3>(numBlocks, 1, 1) * cl::sycl::range<3>(numThreads, 1, 1);
        auto dpct_local_range = cl::sycl::range<3>(numThreads, 1, 1);
        cgh.parallel_for<dpct_kernel_name<class atomic_test_kernel_f6c444>>(
          cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                                cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
          [=](cl::sycl::nd_item<3> item_ct1) {
            atomic_test_kernel(Ddata, item_ct1);
          });
      });
  }

  dpct::get_default_queue().memcpy((void*)(&Hdata2), (void*)(Ddata), sizeof(int)).wait();

  // check add
  //if (Hdata2 != (numThreads * numBlocks)) {
  if (Hdata2 != 84) {
    err = -1;
    printf("atomic_fetch_compare_inc test failed\n");
  }
  

  cl::sycl::free(Ddata, dpct::get_default_queue().get_context());
  printf("atomic_fetch_compare_inc test completed, returned %s\n", err == 0 ? "OK" : "ERROR");
  return err;
}
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}
