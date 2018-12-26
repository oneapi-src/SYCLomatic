//===--- syclct.hpp --------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_H
#define SYCLCT_H

#include <CL/sycl.hpp>
#include <iostream>

#include "syclct_atomic.hpp"
#include "syclct_device.hpp"
#include "syclct_kernel.hpp"
#include "syclct_memory.hpp"

#if defined(_MSC_VER) // MSVC
#define __sycl_align__(n) __declspec(align(n))
#else
#define __sycl_align__(n) __attribute__((aligned(n)))
#endif

template <class... Args> class syclct_kernel_name;

#endif // SYCLCT_H
