//==---- c2s.hpp ----------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __C2S_HPP__
#define __C2S_HPP__

#include <CL/sycl.hpp>
#include <iostream>
#include <limits.h>

template <class... Args> class c2s_kernel_name;
template <int Arg> class c2s_kernel_scalar;

#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "util.hpp"

#if defined(_MSC_VER)
#define __c2s_align__(n) __declspec(align(n))
#define __c2s_inline__ __forceinline
#else
#define __c2s_align__(n) __attribute__((aligned(n)))
#define __c2s_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __c2s_noinline__ __declspec(noinline)
#else
#define __c2s_noinline__ __attribute__((noinline))
#endif

#define C2S_COMPATIBILITY_TEMP (600)

#define C2S_PI_F (3.14159274101257f)
#define C2S_PI (3.141592653589793115998)

#endif // __C2S_HPP__
