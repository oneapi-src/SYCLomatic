//===---APINames_nvJPEG.inc-------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface APIName: name of library Interface API available in official document
*  APIName: name of library API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated successfully.
*
*  Flag: Bit map for all kind of flag for this API.
*    default(all bits are 0): NO_FLAG
*    bit 0: API_CALL_REMOVED
*    bit 1: API_CALL_UNSUPPORTED
*  Priority: migration priority, eg. P0 - P4
*    P0:  should support in Beta release.
*    P4:  should support in Gold.
*
*  MigrationDesc: Successful[: Comment] | Unsupported: Comment
*    Successful:   this API has been migrated.
*    Unsupported:  this API can not be migrated currently.
*                  Must add Comment to explain why this API is Unsupported.
*    Comment:      extra comment on the migration.
*
****************************************************************************/

ENTRY(nvjpegGetProperty, nvjpegGetProperty, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegCreate, nvjpegCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegCreateSimple, nvjpegCreateSimple, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegCreateEx, nvjpegCreateEx, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDestroy, nvjpegDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegSetDeviceMemoryPadding, nvjpegSetDeviceMemoryPadding, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegGetDeviceMemoryPadding, nvjpegGetDeviceMemoryPadding, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegSetPinnedMemoryPadding, nvjpegSetPinnedMemoryPadding, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegGetPinnedMemoryPadding, nvjpegGetPinnedMemoryPadding, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegJpegStateCreate, nvjpegJpegStateCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegJpegStateDestroy, nvjpegJpegStateDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegGetImageInfo, nvjpegGetImageInfo, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecode, nvjpegDecode, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodeBatchedInitialize, nvjpegDecodeBatchedInitialize, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodeBatched, nvjpegDecodeBatched, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodePhaseOne, nvjpegDecodePhaseOne, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodePhaseTwo, nvjpegDecodePhaseTwo, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodePhaseThree, nvjpegDecodePhaseThree, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseOne, nvjpegDecodeBatchedPhaseOne, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseTwo, nvjpegDecodeBatchedPhaseTwo, false, NO_FLAG, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseThree, nvjpegDecodeBatchedPhaseThree, false, NO_FLAG, P4, "comment")
