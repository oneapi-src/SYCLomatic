//===--------------- RecommendLibrariesVersion.inc --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// - Name: Obj name
//   Feature: Feature name
//   SupportedVersion: Time stamp of component
//   LibType: Component type.
//   ReplacementText: The migrated Replacement Text
//   Description: Details of the feature.

#ifndef RECOMMENDLIBRARY
#define RECOMMENDLIBRARY
#endif
// Example:
// RECOMMENDLIBRARY(FreeQuery, "SYCL free function query", "2025", LibType::DPCPP, "this_work_item", "See https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_free_function_kernels.asciidoc for more details.")