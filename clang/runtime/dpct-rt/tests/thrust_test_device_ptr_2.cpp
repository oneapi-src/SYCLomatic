//===--- thrust_test_device_ptr_2.cpp---------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/*
Thrust test case:
package required: inteloneapi package
$ . <path-to-inteloneapi>/setvars.sh

build:
$ dpcpp thrust_test_device_ptr.cpp

run:
$ ./a.out

*/

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

#include <dpct/dpct.hpp>
#include <dpct/dpstd_utils.hpp>

int main(void) {
  const int numsH = 100;
  const int value = -1;

  int *mapsp1H = new int[numsH];
  int *mapspkeyH = new int[numsH];
  int *mapspvalH = new int[numsH];

  std::fill(dpstd::execution::make_sycl_policy<class Policy_1>(dpstd::execution::sycl),mapsp1H, mapsp1H + numsH, value);
  std::fill(dpstd::execution::make_sycl_policy<class Policy_2>(dpstd::execution::sycl),mapspkeyH, mapspkeyH + numsH, value);
  std::fill(dpstd::execution::make_sycl_policy<class Policy_3>(dpstd::execution::sycl),mapspvalH, mapspvalH + numsH, value);

  // cudaMalloc
  dpct::device_ptr<int> mapsp1D = dpct::device_malloc<int>(numsH);
  dpct::device_ptr<int> mapspkeyD = dpct::device_malloc<int>(numsH);
  dpct::device_ptr<int> mapspvalD = dpct::device_malloc<int>(numsH);


  // cudaMemcpy
  std::copy(dpstd::execution::make_sycl_policy<class Policy_4>(dpstd::execution::sycl),mapsp1H, mapsp1H + numsH, mapsp1D);
  std::copy(dpstd::execution::make_sycl_policy<class Policy_5>(dpstd::execution::sycl),mapspkeyH, mapspkeyH + numsH, mapspkeyD);
  std::copy(dpstd::execution::make_sycl_policy<class Policy_6>(dpstd::execution::sycl),mapspvalH, mapspvalH + numsH, mapspvalD);

  // snapshot from Pennant
  dpct::device_ptr<int> mapsp1T(mapsp1D);
  dpct::device_ptr<int> mapspkeyT(mapspkeyD);
  dpct::device_ptr<int> mapspvalT(mapspvalD);

  std::copy(dpstd::execution::make_sycl_policy<class Policy_7>(dpstd::execution::sycl),mapsp1T, mapsp1T + numsH, mapspkeyT);
  dpct::sequence(dpstd::execution::make_sycl_policy<class Policy_8>(dpstd::execution::sycl),mapspvalT, mapspvalT + numsH);

  for (int i = 0; i < numsH; ++i) {
    if (mapspkeyT[i] != value && mapspvalT[i] != i) {
      std::cout << "i = " << i << ", " << mapspkeyT[i] << " " << mapspvalT[i]
                << "; ";
    }
  }
  std::cout << std::endl << "done" << std::endl;

  return 0;
}
