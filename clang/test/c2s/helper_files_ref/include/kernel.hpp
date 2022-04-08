//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __C2S_KERNEL_HPP__
#define __C2S_KERNEL_HPP__

#include <CL/sycl.hpp>

namespace c2s {

typedef void (*kernel_functor)(cl::sycl::queue &, const cl::sycl::nd_range<3> &,
                               unsigned int, void **, void **);

struct kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      c2s::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
}
static kernel_function_info get_kernel_function_info(const void *function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      c2s::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
  return kernel_info;
}

} // namespace c2s
#endif // __C2S_KERNEL_HPP__
