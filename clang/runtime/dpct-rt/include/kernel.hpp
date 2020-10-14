//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_KERNEL_HPP__
#define __DPCT_KERNEL_HPP__

#include <CL/sycl.hpp>

namespace dpct {

struct kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
}

} // namespace dpct
#endif // __DPCT_KERNEL_HPP__