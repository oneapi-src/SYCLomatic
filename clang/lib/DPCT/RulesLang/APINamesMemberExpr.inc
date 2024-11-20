//===--------------- APINamesMemberExpr.inc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.device", MEM_BASE,
                          "get_device_id")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.hostPointer", MEM_BASE,
                          "get_host_pointer")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.devicePointer", MEM_BASE,
                          "get_device_pointer")

CUDA_MEMORY_TYPE_LITERAL_NUMBER_MIGRATION_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cudaPointerAttributes.type", MEM_BASE, "get_memory_type"))

CUDA_MEMORY_TYPE_LITERAL_NUMBER_MIGRATION_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cudaPointerAttributes.memoryType", MEM_BASE, "get_memory_type"))

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.isManaged", MEM_BASE,
                          "is_memory_shared")

WARNING_FACTORY_ENTRY(
    "cudaDeviceProp.totalConstMem",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        MEMBER_CALL_FACTORY_ENTRY("cudaDeviceProp.totalConstMem", MEM_BASE,
                                  "get_global_mem_size")),
    Diagnostics::UNCOMPATIBLE_DEVICE_PROP, std::string("totalConstMem"),
    std::string("get_global_mem_size"))
