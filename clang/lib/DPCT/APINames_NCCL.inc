//===--------------- APINames_NCCL.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface APIName: name of CUDA Interface API available in official document
*  APIName: name of CUDA API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated Commently.
*
*  Flag: Bit map for all kind of flag for this API.
*    default(all bits are 0): NO_FLAG
*    bit 0: API_CALL_REMOVED
*    bit 1: API_CALL_UNSUPPORTED
*  Priority: migration priority, eg. P0 - P4
*    P0:  should support in Beta release.
*    P4:  should support in Gold.
*
*  MigrationDesc: Comment[: Comment] | Unsupported: Comment
*    Comment:   this API has been migrated.
*    Unsupported:  this API can not be migrated currently.
*                  Must add Comment to explain why this API is Unsupported.
*    Comment:      extra comment on the migration.
*
****************************************************************************/


ENTRY(ncclGetVersion, ncclGetVersion, true, NO_FLAG, P4, "Successful")
ENTRY(ncclGetUniqueId, ncclGetUniqueId, true, NO_FLAG, P4, "Successful")
ENTRY(ncclCommInitRank, ncclCommInitRank, true, NO_FLAG, P4, "Successful")
ENTRY(ncclCommInitAll, ncclCommInitAll, false, NO_FLAG, P4, "comment")
ENTRY(ncclCommDestroy, ncclCommDestroy, true, NO_FLAG, P4, "Successful")
ENTRY(ncclCommAbort, ncclCommAbort, false, NO_FLAG, P4, "comment")
ENTRY(ncclCommGetAsyncError, ncclCommGetAsyncError, false, NO_FLAG, P4, "comment")
ENTRY(ncclCommCount, ncclCommCount, true, NO_FLAG, P4, "Successful")
ENTRY(ncclCommCuDevice, ncclCommCuDevice, true, NO_FLAG, P4, "Successful")
ENTRY(ncclCommUserRank, ncclCommUserRank, true, NO_FLAG, P4, "Successful")
ENTRY(ncclAllReduce, ncclAllReduce, true, NO_FLAG, P4, "Successful: DPCT1067")
ENTRY(ncclBroadcast, ncclBroadcast, true, NO_FLAG, P4, "Successful")
ENTRY(ncclReduce, ncclReduce, true, NO_FLAG, P4, "Successful: DPCT1067")
ENTRY(ncclAllGather, ncclAllGather, false, NO_FLAG, P4, "comment")
ENTRY(ncclReduceScatter, ncclReduceScatter, true, NO_FLAG, P4, "Successful: DPCT1067")
ENTRY(ncclGroupStart, ncclGroupStart, false, NO_FLAG, P4, "comment")
ENTRY(ncclGroupEnd, ncclGroupEnd, false, NO_FLAG, P4, "comment")
ENTRY(ncclSend, ncclSend, true, NO_FLAG, P4, "comment")
ENTRY(ncclRecv, ncclRecv, true, NO_FLAG, P4, "comment")

ENTRY(ncclGetLastError, ncclGetLastError, true, NO_FLAG, P4, "comment")
ENTRY(ncclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum, false, NO_FLAG, P4, "comment")
ENTRY(ncclRedOpDestroy, ncclRedOpDestroy, false, NO_FLAG, P4, "comment")
ENTRY(ncclBcast, ncclBcast, true, NO_FLAG, P4, "comment")
ENTRY(ncclGetErrorString, ncclGetErrorString, true, NO_FLAG, P4, "comment")
