// C2S_LABEL_BEGIN|License|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// C2S_LABEL_END

#ifndef __C2S_KERNEL_HPP__
#define __C2S_KERNEL_HPP__
// C2S_COMMENT
// C2S_COMMENT Example1:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_EMPTY
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT Example2:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_BEGIN
// C2S_COMMENT // FileID|FeatureNameRef
// C2S_COMMENT [// FileID|FeatureNameRef]
// C2S_COMMENT ...
// C2S_COMMENT // C2S_DEPENDENCY_END
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT For header file including dependency, please use predefined feature name:
// C2S_COMMENT   local_include_dependency: c2s helper files
// C2S_COMMENT   non_local_include_dependency: other headler files

// C2S_LABEL_BEGIN|non_local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#include <CL/sycl.hpp>
// C2S_LABEL_END
// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END

namespace c2s {

// C2S_LABEL_BEGIN|kernel_functor|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
typedef void (*kernel_functor)(cl::sycl::queue &, const cl::sycl::nd_range<3> &,
                               unsigned int, void **, void **);
// C2S_LABEL_END

// C2S_LABEL_BEGIN|kernel_function_info|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
struct kernel_function_info {
  int max_work_group_size = 0;
};
// C2S_LABEL_END

// C2S_LABEL_BEGIN|get_kernel_function_info|c2s
// C2S_DEPENDENCY_BEGIN
// Kernel|kernel_function_info
// Device|dev_mgr
// Device|dev_mgr_current_device
// C2S_DEPENDENCY_END
// C2S_CODE
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
// C2S_LABEL_END

} // namespace c2s
#endif // __C2S_KERNEL_HPP__
