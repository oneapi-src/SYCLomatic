//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_HPP__
#define __DPCT_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <iostream>
#include <limits.h>
#include <math.h>
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_named_lambda|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <class... Args> class dpct_kernel_name;
template <int Arg> class dpct_kernel_scalar;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "memory.hpp"
#include "util.hpp"

// DPCT_LABEL_BEGIN|dpct_align_and_inline|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_noinline|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#if defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#else
#define __dpct_noinline__ __attribute__((noinline))
#endif
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_compatibility_temp|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#define DPCT_COMPATIBILITY_TEMP (600)
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|error_code|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
namespace dpct{
enum error_code { success = 0, default_error = 999 };
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|check_error_code|
// DPCT_DEPENDENCY_BEGIN
// Dpct|error_code
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_pi_f|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#define DPCT_PI_F (3.14159274101257f)
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dpct_pi|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#define DPCT_PI (3.141592653589793115998)
// DPCT_LABEL_END

#endif // __DPCT_HPP__
