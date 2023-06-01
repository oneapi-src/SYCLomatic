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

ENTRY(nvtxNameCuDeviceA, nvtxNameCuDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuDeviceW, nvtxNameCuDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuContextA, nvtxNameCuContextA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuContextW, nvtxNameCuContextW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuStreamA, nvtxNameCuStreamA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuStreamW, nvtxNameCuStreamW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuEventA, nvtxNameCuEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCuEventW, nvtxNameCuEventW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaDeviceA, nvtxNameCudaDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaDeviceW, nvtxNameCudaDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaStreamA, nvtxNameCudaStreamA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaStreamW, nvtxNameCudaStreamW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaEventA, nvtxNameCudaEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCudaEventW, nvtxNameCudaEventW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxInitialize, nvtxInitialize, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainMarkEx, nvtxDomainMarkEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMarkEx, nvtxMarkEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMarkA, nvtxMarkA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxMarkW, nvtxMarkW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangeStartEx, nvtxDomainRangeStartEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStartEx, nvtxRangeStartEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStartA, nvtxRangeStartA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeStartW, nvtxRangeStartW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangeEnd, nvtxDomainRangeEnd, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangeEnd, nvtxRangeEnd, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangePushEx, nvtxDomainRangePushEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePushEx, nvtxRangePushEx, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePushA, nvtxRangePushA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePushW, nvtxRangePushW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRangePop, nvtxDomainRangePop, false, NO_FLAG, P4, "comment")
ENTRY(nvtxRangePop, nvtxRangePop, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainResourceCreate, nvtxDomainResourceCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainResourceDestroy, nvtxDomainResourceDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainNameCategoryA, nvtxDomainNameCategoryA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainNameCategoryW, nvtxDomainNameCategoryW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCategoryA, nvtxNameCategoryA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameCategoryW, nvtxNameCategoryW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameOsThreadA, nvtxNameOsThreadA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameOsThreadW, nvtxNameOsThreadW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRegisterStringA, nvtxDomainRegisterStringA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainRegisterStringW, nvtxDomainRegisterStringW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainCreateA, nvtxDomainCreateA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainCreateW, nvtxDomainCreateW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainDestroy, nvtxDomainDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserCreate, nvtxDomainSyncUserCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserDestroy, nvtxDomainSyncUserDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireStart, nvtxDomainSyncUserAcquireStart, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireFailed, nvtxDomainSyncUserAcquireFailed, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserAcquireSuccess, nvtxDomainSyncUserAcquireSuccess, false, NO_FLAG, P4, "comment")
ENTRY(nvtxDomainSyncUserReleasing, nvtxDomainSyncUserReleasing, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClDeviceA, nvtxNameClDeviceA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClDeviceW, nvtxNameClDeviceW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClContextA, nvtxNameClContextA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClContextW, nvtxNameClContextW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClCommandQueueA, nvtxNameClCommandQueueA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClCommandQueueW, nvtxNameClCommandQueueW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClMemObjectA, nvtxNameClMemObjectA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClMemObjectW, nvtxNameClMemObjectW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClSamplerA, nvtxNameClSamplerA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClSamplerW, nvtxNameClSamplerW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClProgramA, nvtxNameClProgramA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClProgramW, nvtxNameClProgramW, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClEventA, nvtxNameClEventA, false, NO_FLAG, P4, "comment")
ENTRY(nvtxNameClEventW, nvtxNameClEventW, false, NO_FLAG, P4, "comment")
