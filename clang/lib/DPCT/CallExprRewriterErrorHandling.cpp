//===--------------- CallExprRewriterErrorHandling.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

void CallExprRewriterFactoryBase::initRewriterMapErrorHandling() {
  RewriterMap->merge(
  std::unordered_map<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
  {WARNING_FACTORY_ENTRY(
  "cudaGetErrorString",
  INSERT_AROUND_FACTORY(
  CALL_FACTORY_ENTRY("cudaGetErrorString",
                     CALL("cudaGetErrorString", ARG_WC(0))),
  "\"cudaGetErrorString is not supported\"/*", "*/"),
  Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED)

   WARNING_FACTORY_ENTRY(
   "cudaGetErrorName",
   INSERT_AROUND_FACTORY(
   CALL_FACTORY_ENTRY("cudaGetErrorName", CALL("cudaGetErrorName", ARG_WC(0))),
   "\"cudaGetErrorName is not supported\"/*", "*/"),
   Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED)

   CONDITIONAL_FACTORY_ENTRY(
   checkIsCallExprOnly(),
   WARNING_FACTORY_ENTRY(
   "cudaGetLastError", TOSTRING_FACTORY_ENTRY("cudaGetLastError", LITERAL("")),
   Diagnostics::FUNC_CALL_REMOVED, std::string("cudaGetLastError"),
   std::string("this call is redundant in SYCL.")),
   WARNING_FACTORY_ENTRY(
   "cudaGetLastError", TOSTRING_FACTORY_ENTRY("cudaGetLastError", LITERAL("0")),
   Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))

   CONDITIONAL_FACTORY_ENTRY(
   checkIsCallExprOnly(),
   WARNING_FACTORY_ENTRY(
   "cudaPeekAtLastError",
   TOSTRING_FACTORY_ENTRY("cudaPeekAtLastError", LITERAL("")),
   Diagnostics::FUNC_CALL_REMOVED, std::string("cudaPeekAtLastError"),
   std::string("this call is redundant in SYCL.")),
   WARNING_FACTORY_ENTRY(
   "cudaPeekAtLastError",
   TOSTRING_FACTORY_ENTRY("cudaPeekAtLastError", LITERAL("0")),
   Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))

  }));
}

} // namespace dpct
} // namespace clang
