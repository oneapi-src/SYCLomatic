//==---- c2s.hpp ----------------------------------*- C++ -*----------------==//
// C2S_LABEL_BEGIN|License|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// C2S_LABEL_END

#ifndef __C2S_HPP__
#define __C2S_HPP__
// C2S_COMMENT
// C2S_COMMENT Example1:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_EMPTY
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT Example2:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_BEGIN
// C2S_COMMENT // FileID|FeatureNameRef
// C2S_COMMENT [// FileID|FeatureNameRef]
// C2S_COMMENT ...
// C2S_COMMENT // C2S_DEPENDENCY_END
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT For header file including dependency, please use predefined feature name:
// C2S_COMMENT   local_include_dependency: c2s helper files
// C2S_COMMENT   non_local_include_dependency: other headler files

// C2S_LABEL_BEGIN|non_local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#include <CL/sycl.hpp>
#include <iostream>
#include <limits.h>
// C2S_LABEL_END

// C2S_LABEL_BEGIN|c2s_named_lambda|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
template <class... Args> class c2s_kernel_name;
template <int Arg> class c2s_kernel_scalar;
// C2S_LABEL_END

// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END
#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "util.hpp"

// C2S_LABEL_BEGIN|c2s_align_and_inline|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#if defined(_MSC_VER)
#define __c2s_align__(n) __declspec(align(n))
#define __c2s_inline__ __forceinline
#else
#define __c2s_align__(n) __attribute__((aligned(n)))
#define __c2s_inline__ __inline__ __attribute__((always_inline))
#endif
// C2S_LABEL_END

// C2S_LABEL_BEGIN|c2s_noinline|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#if defined(_MSC_VER)
#define __c2s_noinline__ __declspec(noinline)
#else
#define __c2s_noinline__ __attribute__((noinline))
#endif
// C2S_LABEL_END

// C2S_LABEL_BEGIN|c2s_compatibility_temp|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#define C2S_COMPATIBILITY_TEMP (600)
// C2S_LABEL_END

// C2S_LABEL_BEGIN|c2s_pi_f|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#define C2S_PI_F (3.14159274101257f)
// C2S_LABEL_END
// C2S_LABEL_BEGIN|c2s_pi|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#define C2S_PI (3.141592653589793115998)
// C2S_LABEL_END

#endif // __C2S_HPP__
