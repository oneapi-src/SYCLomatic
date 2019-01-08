//===--- syclct_util.hpp -------------------------------*- C++ -*-----===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_UTIL_H
#define SYCLCT_UTIL_H

#include <CL/sycl.hpp>

namespace syclct {

inline double ll2d(long long int x) {
  static_assert(sizeof(double) == sizeof(unsigned long long int),
                "Mismatched type size");
  union {
    long long int input;
    double output;
  } data;
  data.input = x;
  return data.output;
}

inline long long int d2ll(double x) {
  static_assert(sizeof(double) == sizeof(unsigned long long int),
                "Mismatched type size");
  union {
    double input;
    long long int output;
  } data;
  data.input = x;
  return data.output;
}

} // namespace syclct

#endif // SYCLCT_UTIL_H
