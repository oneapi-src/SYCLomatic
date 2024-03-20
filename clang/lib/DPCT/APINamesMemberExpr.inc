//===--------------- APINamesMemberExpr.inc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.device", MEM_BASE, false,
                          "get_device_id")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.hostPointer", MEM_BASE, false,
                          "get_host_pointer")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.devicePointer", MEM_BASE,
                          false, "get_device_pointer")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.type", MEM_BASE, false,
                          "get_memory_type")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.memoryType", MEM_BASE, false,
                          "get_memory_type")

MEMBER_CALL_FACTORY_ENTRY("cudaPointerAttributes.isManaged", MEM_BASE, false,
                          "is_memory_shared")

//WARNING_FACTORY_ENTRY(
//    "cudaDeviceProp.totalConstMem",
//    FEATURE_REQUEST_FACTORY(
//        HelperFeatureEnum::device_ext,
//        MEMBER_REWRITE_ENTRY("cudaDeviceProp.totalConstMem",
//                             MEMBER_FACTORY(MEM_BASE, IS_ARROW,
//                                            MEM_CALL("get_global_mem_size")))),
//    Diagnostics::UNCOMPATIBLE_DEVICE_PROP, std::string("totalConstMem"),
//    std::string("get_global_mem_size"))
//

//MEM_EXPR_ENTRY("cudaPointerAttributes.d_buffers", MEM_BASE, false, "iter")
//
//MEM_EXPR_ENTRY("cudaPointerAttributes.Current", MEM_BASE, false, "first")
//
//MEM_EXPR_ENTRY("cudaPointerAttributes.Alternate", MEM_BASE, false, "second")
