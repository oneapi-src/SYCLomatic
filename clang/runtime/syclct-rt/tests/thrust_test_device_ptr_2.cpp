//===--- thrust_test_device_ptr_2.cpp---------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/*
Thrust test case:
package required:
install tbb and pstl library(eg. install intel parallel studio)
cd /path/to/tbb/bin &&  source  tbbvars.sh intel64
cd /path/to/pstl/bin && source   pstlvars.sh intel64 auto_pstlroot

build:
compute++   thrust_test_device_ptr_2.cpp
  -I/Path/to/ComputeCpp-CE-1.0.2-Ubuntu-16.04-x86_64/include/
  -std=c++11
  -L/Path/to/ComputeCpp-CE-1.0.2-Ubuntu-16.04-x86_64/lib
  -lComputeCpp   -I/path/to/syclct-install/include/
  -sycl-driver
*/
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

#include <syclct/syclct_thrust.hpp>

int main(void) {
  const int numsH = 100;
  const int value = -1;

  int *mapsp1H = new int[numsH];
  int *mapspkeyH = new int[numsH];
  int *mapspvalH = new int[numsH];

  std::fill(mapsp1H, mapsp1H + numsH, value);
  std::fill(mapspkeyH, mapspkeyH + numsH, value);
  std::fill(mapspvalH, mapspvalH + numsH, value);

  // cudaMalloc
  thrust::device_ptr<int> mapsp1D = thrust::device_malloc<int>(numsH);
  thrust::device_ptr<int> mapspkeyD = thrust::device_malloc<int>(numsH);
  thrust::device_ptr<int> mapspvalD = thrust::device_malloc<int>(numsH);

  // cudaMemcpy
  thrust::copy(mapsp1H, mapsp1H + numsH, mapsp1D);
  thrust::copy(mapspkeyH, mapspkeyH + numsH, mapspkeyD);
  thrust::copy(mapspvalH, mapspvalH + numsH, mapspvalD);

  // snapshot from Pennant
  thrust::device_ptr<int> mapsp1T(mapsp1D);
  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
  thrust::device_ptr<int> mapspvalT(mapspvalD);

  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
  thrust::sequence(mapspvalT, mapspvalT + numsH);

  for (int i = 0; i < numsH; ++i) {
    if (mapspkeyT[i] != value && mapspvalT[i] != i) {
      std::cout << "i = " << i << ", " << mapspkeyT[i] << " " << mapspvalT[i]
                << "; ";
    }
  }
  std::cout << std::endl << "done" << std::endl;

  return 0;
}
