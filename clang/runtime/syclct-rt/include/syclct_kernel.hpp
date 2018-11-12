//===--- syclct_kernel.hpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_KERNEL_H
#define SYCLCT_KERNEL_H

#include <CL/sycl.hpp>

struct sycl_kernel_function_info {
  int max_work_group_size = 0;
};

static void getSyclKernelFunctionInfo(sycl_kernel_function_info *KernelInfo,
                               const void *Func) {
  static cl::sycl::device Device;
  KernelInfo->max_work_group_size =
      Device.get_info<cl::sycl::info::device::max_work_group_size>();
}

#endif // !SYCLCT_KERNEL_H
