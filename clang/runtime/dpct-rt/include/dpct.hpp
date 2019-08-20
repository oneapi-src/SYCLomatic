/******************************************************************************
* INTEL CONFIDENTIAL
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

#ifndef DPCT_H
#define DPCT_H

#include <CL/sycl.hpp>
#include <iostream>

#define DPCPP_COMPATIBILITY_TEMP (200)

#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "util.hpp"

#if defined(_MSC_VER)
#define __sycl_align__(n) __declspec(align(n))
#else
#define __sycl_align__(n) __attribute__((aligned(n)))
#endif

template <class... Args> class dpct_kernel_name;
template <int Arg> class dpct_kernel_scalar;

#define DPCT_PI_F (3.14159274101257f)
#define DPCT_PI (3.141592653589793115998)

#include <limits.h>

#endif // DPCT_H
