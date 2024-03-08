//===--------------- APINamesErrorHandling.inc ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

WARNING_FACTORY_ENTRY(
    "cudnnGetErrorString",
    TOSTRING_FACTORY_ENTRY("cudnnGetErrorString",
                           LITERAL("\"<Placeholder string>\"")),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED,
    ARG("The call was replaced by a placeholder string"))

WARNING_FACTORY_ENTRY(
    "cudaGetErrorString",
    TOSTRING_FACTORY_ENTRY("cudaGetErrorString",
                           LITERAL("\"<Placeholder string>\"")),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED,
    ARG("The call was replaced by a placeholder string"))

WARNING_FACTORY_ENTRY(
    "cudaGetErrorName",
    TOSTRING_FACTORY_ENTRY("cudaGetErrorName",
                           LITERAL("\"<Placeholder string>\"")),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED,
    ARG("The call was replaced by a placeholder string"))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY(
        "cudaGetLastError",
        TOSTRING_FACTORY_ENTRY("cudaGetLastError", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED, std::string("cudaGetLastError"),
        std::string("this functionality is redundant in SYCL.")),
    WARNING_FACTORY_ENTRY(
        "cudaGetLastError",
        TOSTRING_FACTORY_ENTRY("cudaGetLastError", LITERAL("0")),
        Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY(
        "cudaPeekAtLastError",
        TOSTRING_FACTORY_ENTRY("cudaPeekAtLastError", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED, std::string("cudaPeekAtLastError"),
        std::string("this functionality is redundant in SYCL.")),
    WARNING_FACTORY_ENTRY(
        "cudaPeekAtLastError",
        TOSTRING_FACTORY_ENTRY("cudaPeekAtLastError", LITERAL("0")),
        Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))

WARNING_FACTORY_ENTRY(
    "ncclGetErrorString",
    TOSTRING_FACTORY_ENTRY("cublasGetStatusString",
                           LITERAL("\"<Placeholder string>\"")),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED,
    ARG("The call was replaced by a placeholder string"))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY(
        "ncclGetLastError",
        TOSTRING_FACTORY_ENTRY("ncclGetLastError", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED, std::string("ncclGetLastError"),
        std::string("this functionality is redundant in SYCL.")),
    WARNING_FACTORY_ENTRY(
        "ncclGetLastError",
        TOSTRING_FACTORY_ENTRY("ncclGetLastError", LITERAL("0")),
        Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY(
        "ncclCommGetAsyncError",
        TOSTRING_FACTORY_ENTRY("ncclCommGetAsyncError", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED, std::string("ncclCommGetAsyncError"),
        std::string("this functionality is redundant in SYCL.")),
    WARNING_FACTORY_ENTRY(
        "ncclCommGetAsyncError",
        TOSTRING_FACTORY_ENTRY("ncclCommGetAsyncError", LITERAL("0")),
        Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0))
