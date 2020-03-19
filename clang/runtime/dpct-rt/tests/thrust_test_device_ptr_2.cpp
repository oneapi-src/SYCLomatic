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

Environment setup:
  oneAPI environment: dpcpp, dpct, and tbb

build:
  dpcpp -fno-sycl-unnamed-lambda thrust_test_device_ptr_2.cpp

run:
  ./a.out

expected output:
Passed

*/

#define DPCT_NAMED_LAMBDA

#include <cstdio>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpstd_utils.hpp>
#include <dpstd/algorithm>
#include <dpstd/execution>

static void dumpMaps(const char *name, dpct::device_ptr<int> &maps, int num) {
  std::cout << name << "\n";
  for (int i = 0; i < num; ++i) {
    std::cout << i << ": " << maps[i] << "\n";
  }
}
static void dumpMaps(const char *name, int *maps, int num) {
  std::cout << name << "\n";
  for (int i = 0; i < num; ++i) {
    std::cout << i << ": " << maps[i] << "\n";
  }
}

int main(void) {
  const int numsH = 10;
  const int valuep1 = -1;
  const int valuepkey = -2;
  const int valuepval = -3;

  int *mapsp1H = new int[numsH];
  int *mapspkeyH = new int[numsH];
  int *mapspvalH = new int[numsH];

  std::fill(mapsp1H, mapsp1H + numsH, valuep1);
  std::fill(mapspkeyH, mapspkeyH + numsH, valuepkey);
  std::fill(mapspvalH, mapspvalH + numsH, valuepval);
  // dumpMaps("mapsp1H", mapsp1H, numsH);
  // dumpMaps("mapspkeyH", mapspkeyH, numsH);
  // dumpMaps("mapspvalH", mapspvalH, numsH);

  // cudaMalloc
  int *mapsp1D = (int *)sycl::malloc_device(numsH * sizeof(int),
                                            dpct::get_current_device(),
                                            dpct::get_default_context());
  int *mapspkeyD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                              dpct::get_current_device(),
                                              dpct::get_default_context());
  int *mapspvalD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                              dpct::get_current_device(),
                                              dpct::get_default_context());

  // cudaMemcpy
  std::copy(mapsp1H, mapsp1H + numsH, mapsp1D);
  std::copy(mapspkeyH, mapspkeyH + numsH, mapspkeyD);
  std::copy(mapspvalH, mapspvalH + numsH, mapspvalD);

  // snapshot from Pennant
  dpct::device_ptr<int> mapsp1T(mapsp1D);
  dpct::device_ptr<int> mapspkeyT(mapspkeyD);
  dpct::device_ptr<int> mapspvalT(mapspvalD);
  // dumpMaps("mapsp1T", mapsp1T, numsH);
  // dumpMaps("mapspkeyT", mapspkeyT, numsH);
  // dumpMaps("mapspvalT", mapspvalT, numsH);

  std::copy(dpstd::execution::make_sycl_policy<class Policy_7>(
                dpstd::execution::sycl),
            mapsp1T, mapsp1T + numsH, mapspkeyT);
  dpct::sequence(dpstd::execution::make_sycl_policy<class Policy_8>(
                     dpstd::execution::sycl),
                 mapspvalT, mapspvalT + numsH);
  // dumpMaps("mapspkeyT after copy", mapspkeyT, numsH);
  // dumpMaps("mapspvalT after sequence", mapspvalT, numsH);

  bool pass = true;
  for (int i = 0; i < numsH; ++i) {
    if (mapspkeyT[i] != valuep1) {
      std::cout << "Unexpected key: mapspkeyT[" << i << "] == " << mapspkeyT[i]
                << ", expected " << valuep1 << "\n";
      pass = false;
    }
    if (mapspvalT[i] != i) {
      std::cout << "Unexpected val: mapspvalT[" << i << "] == " << mapspvalT[i]
                << ", expected " << i << "\n";
      pass = false;
    }
  }
  std::cout << std::endl << (pass ? "Passed" : "Failed") << "\n";

  return 0;
}
