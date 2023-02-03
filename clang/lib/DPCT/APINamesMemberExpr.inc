//===--------------- APINamesMemberExpr.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.device",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_device_id"))
)

MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.hostPointer",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_host_pointer"))
)

MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.devicePointer",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_device_pointer"))
)

MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.type",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_memory_type"))
)

MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.memoryType",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_memory_type"))
)

MEMBER_REWRITE_ENTRY(
    "cudaPointerAttributes.isManaged",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("is_memory_shared"))
)

WARNING_FACTORY_ENTRY(
    "cudaDeviceProp.totalConstMem",
    MEMBER_REWRITE_ENTRY(
        "cudaDeviceProp.totalConstMem",
        MEMBER_FACTORY(MEM_BASE, IS_ARROW, MEM_CALL("get_global_mem_size"))),
    Diagnostics::UNCOMPATIBLE_DEVICE_PROP, std::string("totalConstMem"),
    std::string("get_global_mem_size"))

MEMBER_REWRITE_ENTRY(
    "cub::DoubleBuffer.d_buffers",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, LITERAL("iter")))

MEMBER_REWRITE_ENTRY(
    "cub::DoubleBuffer.Current",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, LITERAL("input")))

MEMBER_REWRITE_ENTRY(
    "cub::DoubleBuffer.Alternate",
    MEMBER_FACTORY(MEM_BASE, IS_ARROW, LITERAL("output")))
