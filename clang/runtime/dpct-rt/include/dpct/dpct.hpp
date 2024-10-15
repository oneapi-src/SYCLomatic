//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_HPP__
#define __DPCT_HPP__

#include <sycl/sycl.hpp>
#include <iostream>
#include <limits.h>
#include <math.h>

template <class... Args> class dpct_kernel_name;
template <int Arg> class dpct_kernel_scalar;

#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "memory.hpp"
#include "util.hpp"

#include "bindless_images.hpp"
#include "graph.hpp"

#define USE_DPCT_HELPER 1

#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#else
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_COMPATIBILITY_TEMP (900)

namespace dpct {
enum error_code { success = 0, default_error = 999 };
inline const char *get_error_dummy(int ec) {
  static const std::string Msg =
      "SYCL uses exceptions to report errors and does not use the error codes. "
      "You need to rewrite this code.";
  return Msg.c_str();
}
} // namespace dpct

#define DPCT_CHECK_ERROR(expr)                                                 \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return dpct::success;                                                    \
    } catch (std::exception const &e) {                                        \
      std::cerr << e.what() << std::endl;                                      \
      return dpct::default_error;                                              \
    }                                                                          \
  }()

#define DPCT_PI_F (3.14159274101257f)
#define DPCT_PI (3.141592653589793115998)

#endif // __DPCT_HPP__
