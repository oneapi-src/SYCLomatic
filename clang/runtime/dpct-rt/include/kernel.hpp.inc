// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_KERNEL_HPP__
#define __DPCT_KERNEL_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other headler files

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

// DPCT_LABEL_BEGIN|kernel_functor|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
typedef void (*kernel_functor)(cl::sycl::queue &, const cl::sycl::nd_range<3> &,
                               unsigned int, void **, void **);
// DPCT_LABEL_END

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
// Device|dev_mgr
// Device|dev_mgr_current_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
}
static kernel_function_info get_kernel_function_info(const void *function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
  return kernel_info;
}
// DPCT_LABEL_END

} // namespace dpct
#endif // __DPCT_KERNEL_HPP__
