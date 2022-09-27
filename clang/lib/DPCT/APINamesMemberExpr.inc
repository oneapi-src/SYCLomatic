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
