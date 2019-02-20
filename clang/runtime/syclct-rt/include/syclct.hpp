/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
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

//===--- syclct.hpp --------------------------------------*- C++ -*---===//

#ifndef SYCLCT_H
#define SYCLCT_H

#include <CL/sycl.hpp>
#include <iostream>

// Todo: update this temply macro with the one available in sycl compiler.
#define SYCL_ARCH_TEMPLY_USED_MACRO

#include "syclct_atomic.hpp"
#include "syclct_device.hpp"
#include "syclct_kernel.hpp"
#include "syclct_memory.hpp"
#include "syclct_util.hpp"

#if defined(_MSC_VER) // MSVC
#define __sycl_align__(n) __declspec(align(n))
#else
#define __sycl_align__(n) __attribute__((aligned(n)))
#endif

template <class... Args> class syclct_kernel_name;

#endif // SYCLCT_H
