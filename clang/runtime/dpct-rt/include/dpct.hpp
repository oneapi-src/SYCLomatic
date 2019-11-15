/******************************************************************************
*
* Copyright 2018 - 2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- dpct.hpp --------------------------------------*- C++ -*---===//

#ifndef __DPCT_HPP__
#define __DPCT_HPP__

#include <CL/sycl.hpp>
#include <iostream>
#include <limits.h>

#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "util.hpp"

#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#ifdef DPCT_NAMED_LAMBDA
template <class... Args> class dpct_kernel_name;
template <int Arg> class dpct_kernel_scalar;
#endif

#define DPCPP_COMPATIBILITY_TEMP (200)

#define DPCT_PI_F (3.14159274101257f)
#define DPCT_PI (3.141592653589793115998)

#endif // __DPCT_HPP__
