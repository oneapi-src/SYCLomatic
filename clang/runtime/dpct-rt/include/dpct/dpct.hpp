//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Top level header file for all helper functions
 * 
 * @copyright Copyright (C) Intel Corporation
 * 
 */

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

#if defined(_MSC_VER)
/**
 * @brief align for MSC.
 */
#define __dpct_align__(n) __declspec(align(n))
/**
 * @brief force inline for MSC.
 */
#define __dpct_inline__ __forceinline
#else
/**
 * @brief align for other compilers.
 */
#define __dpct_align__(n) __attribute__((aligned(n)))
/**
 * @brief force inline for other compilers.
 */
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
/**
 * @brief No inline for MSC.
 */
#define __dpct_noinline__ __declspec(noinline)
#else
/**
 * @brief No inline for other compilers.
 */
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_COMPATIBILITY_TEMP (900)

namespace dpct{
/**
 * @brief Error code which will be returned when exceptions happen.
 */
enum error_code { success = 0, default_error = 999 };
}
/**
 * @brief Functional macro to wrap a function call and catch the exceptions.
 */
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
/**
 * @brief Float value of Pi.
 */
#define DPCT_PI_F (3.14159274101257f)
/**
 * @brief Double value of Pi.
 */
#define DPCT_PI (3.141592653589793115998)

#endif // __DPCT_HPP__
