//===--------------- APINames_NVTX.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface APIName: name of Interface API available in official document
*  APIName: name of API
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

ENTRY(nvtxNameCuDevice, nvtxNameCuDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuDevice, nvtxNameCuDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuContext, nvtxNameCuContextA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuContext, nvtxNameCuContextW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuStream, nvtxNameCuStreamA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuStream, nvtxNameCuStreamW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuEvent, nvtxNameCuEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuEvent, nvtxNameCuEventW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaDevice, nvtxNameCudaDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaDevice, nvtxNameCudaDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaStream, nvtxNameCudaStreamA, true, NO_FLAG, P4, "DPCT1026")
ENTRY(nvtxNameCudaStream, nvtxNameCudaStreamW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaEvent, nvtxNameCudaEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaEvent, nvtxNameCudaEventW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxInitialize, nvtxInitialize, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainMarkEx, nvtxDomainMarkEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMarkEx, nvtxMarkEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMark, nvtxMarkA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMark, nvtxMarkW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangeStartEx, nvtxDomainRangeStartEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStartEx, nvtxRangeStartEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStart, nvtxRangeStartA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStart, nvtxRangeStartW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangeEnd, nvtxDomainRangeEnd, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeEnd, nvtxRangeEnd, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangePushEx, nvtxDomainRangePushEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePushEx, nvtxRangePushEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePush, nvtxRangePushA, true, NO_FLAG, P4, "DPCT1026/DPCT1027")
ENTRY(nvtxRangePush, nvtxRangePushW, true, NO_FLAG, P4, "DPCT1026/DPCT1027")
ENTRY(nvtxDomainRangePop, nvtxDomainRangePop, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePop, nvtxRangePop, true, NO_FLAG, P4, "DPCT1026/DPCT1027")
ENTRY(nvtxDomainResourceCreate, nvtxDomainResourceCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainResourceDestroy, nvtxDomainResourceDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainNameCategory, nvtxDomainNameCategoryA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainNameCategory, nvtxDomainNameCategoryW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCategory, nvtxNameCategoryA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCategory, nvtxNameCategoryW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameOsThread, nvtxNameOsThreadA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameOsThread, nvtxNameOsThreadW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRegisterString, nvtxDomainRegisterStringA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRegisterString, nvtxDomainRegisterStringW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainCreate, nvtxDomainCreateA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainCreate, nvtxDomainCreateW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainDestroy, nvtxDomainDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserCreate, nvtxDomainSyncUserCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserDestroy, nvtxDomainSyncUserDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireStart, nvtxDomainSyncUserAcquireStart, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireFailed, nvtxDomainSyncUserAcquireFailed, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireSuccess, nvtxDomainSyncUserAcquireSuccess, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserReleasing, nvtxDomainSyncUserReleasing, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClDevice, nvtxNameClDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClDevice, nvtxNameClDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClContext, nvtxNameClContextA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClContext, nvtxNameClContextW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClCommandQueue, nvtxNameClCommandQueueA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClCommandQueue, nvtxNameClCommandQueueW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClMemObject, nvtxNameClMemObjectA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClMemObject, nvtxNameClMemObjectW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClSampler, nvtxNameClSamplerA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClSampler, nvtxNameClSamplerW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClProgram, nvtxNameClProgramA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClProgram, nvtxNameClProgramW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClEvent, nvtxNameClEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClEvent, nvtxNameClEventW, false, NO_FLAG, P4, "comment")
