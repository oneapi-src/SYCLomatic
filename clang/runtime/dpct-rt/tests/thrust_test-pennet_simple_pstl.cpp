//===--- thrust_test-pennet_simple_pstl.cpp---------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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
  dpcpp -fno-sycl-unnamed-lambda thrust_test-pennet_simple_pstl.cpp

run:
  ./a.out

expected output:
i = 0, 101 9;
i = 1, 102 8;
i = 2, 103 7;
i = 3, 104 6;
i = 4, 105 5;
i = 5, 106 4;
i = 6, 107 3;
i = 7, 108 2;
i = 8, 109 1;
i = 9, 110 0;

done
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

int main() {

  int *mapsp1D, *mapspkeyD, *mapspvalD;
  int numsH = 10;

  mapsp1D = (int *)sycl::malloc_device(numsH * sizeof(int),
                                       dpct::get_current_device(),
                                       dpct::get_default_context());
  mapspkeyD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                         dpct::get_current_device(),
                                         dpct::get_default_context());
  mapspvalD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                         dpct::get_current_device(),
                                         dpct::get_default_context());

  dpct::device_ptr<int> mapsp1T(mapsp1D);
  dpct::device_ptr<int> mapspkeyT(mapspkeyD);
  dpct::device_ptr<int> mapspvalT(mapspvalD);

  for (int i = 0; i < numsH; ++i) {
    mapsp1T[i] = 100 + numsH - i;
  }
  // dumpMaps("mapsp1T", mapsp1T, numsH);

  // copy vector: mapsp1T -> mapspkeyT
  std::copy(dpstd::execution::make_sycl_policy<class Policy_44009e>(
                dpstd::execution::sycl),
            mapsp1T, mapsp1T + numsH, mapspkeyT);
  // dumpMaps("mapspkeyT after copy", mapspkeyT, numsH);

  // create a sequence of numbers in mapspvalT vector [0, 1, 2, ..., 9]
  dpct::sequence(dpstd::execution::make_sycl_policy<class Policy_9a9f11>(
                     dpstd::execution::sycl),
                 mapspvalT, mapspvalT + numsH);
  // dumpMaps("mapspvalT after sequence", mapspvalT, numsH);

  // sort both mapspkeyT and mapspvalT, so that the elements in mapspkeyT are in
  // smallest first order
  dpct::stable_sort_by_key(
      dpstd::execution::make_sycl_policy<class Policy_3b8d2e>(
          dpstd::execution::sycl),
      mapspkeyT, mapspkeyT + numsH, mapspvalT);

  for (int i = 0; i < numsH; ++i) {
    std::cout << "i = " << i << ", " << mapspkeyT[i] << " " << mapspvalT[i]
              << ";\n";
  }
  std::cout << std::endl << "done" << std::endl;
}
