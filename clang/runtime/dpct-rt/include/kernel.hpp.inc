//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_KERNEL_HPP__
#define __DPCT_KERNEL_HPP__

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END

namespace dpct {

// DPCT_LABEL_BEGIN|kernel_function_info|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
struct kernel_function_info {
  int max_work_group_size = 0;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_kernel_function_info|dpct
// DPCT_DEPENDENCY_BEGIN
// Kernel|kernel_function_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
}
// DPCT_LABEL_END

} // namespace dpct
#endif // __DPCT_KERNEL_HPP__