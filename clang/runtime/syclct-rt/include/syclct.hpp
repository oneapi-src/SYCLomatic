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

#include "syclct_device.hpp"
#include "syclct_memory.hpp"

#define PARALLEL_FOR_CONSTRUCT_GLOBAL_EXECUTION_RANGE(R1, R2) \
        (cl::sycl::range<3>(R1*R2))

#endif // SYCLCT_H
