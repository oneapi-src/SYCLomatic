//===--- APINames_cuFFT.inc -----------------------------------------===//
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
ENTRY(cufftPlan1d, cufftPlan1d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftPlan2d, cufftPlan2d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftPlan3d, cufftPlan3d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftPlanMany, cufftPlanMany, true, NO_FLAG, P4, "comment")
ENTRY(cufftXtMakePlanMany, cufftXtMakePlanMany, false, NO_FLAG, P4, "comment")
ENTRY(cufftExecC2C, cufftExecC2C, true, NO_FLAG, P4, "Successful")
ENTRY(cufftExecZ2Z, cufftExecZ2Z, true, NO_FLAG, P4, "comment")
ENTRY(cufftExecR2C, cufftExecR2C, true, NO_FLAG, P4, "Successful")
ENTRY(cufftExecD2Z, cufftExecD2Z, true, NO_FLAG, P4, "comment")
ENTRY(cufftExecC2R, cufftExecC2R, true, NO_FLAG, P4, "comment")
ENTRY(cufftExecZ2D, cufftExecZ2D, true, NO_FLAG, P4, "comment")
ENTRY(cufftCreate, cufftCreate, true, NO_FLAG, P4, "comment")
ENTRY(cufftXtSetGPUs, cufftXtSetGPUs, false, NO_FLAG, P4, "comment")
ENTRY(cufftMakePlan1d, cufftMakePlan1d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftMakePlan2d, cufftMakePlan2d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftMakePlan3d, cufftMakePlan3d, true, NO_FLAG, P4, "Successful")
ENTRY(cufftMakePlanMany, cufftMakePlanMany, true, NO_FLAG, P4, "Successful")
ENTRY(cufftGetSize1d, cufftGetSize1d, false, NO_FLAG, P4, "comment")
ENTRY(cufftGetSize2d, cufftGetSize2d, false, NO_FLAG, P4, "comment")
ENTRY(cufftGetSize3d, cufftGetSize3d, false, NO_FLAG, P4, "comment")
ENTRY(cufftGetSizeMany, cufftGetSizeMany, false, NO_FLAG, P4, "comment")
ENTRY(cufftGetSize, cufftGetSize, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtSetWorkArea, cufftXtSetWorkArea, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtMalloc, cufftXtMalloc, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtMemcpy, cufftXtMemcpy, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtExecDescriptorC2C, cufftXtExecDescriptorC2C, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtExecDescriptorZ2Z, cufftXtExecDescriptorZ2Z, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtFree, cufftXtFree, false, NO_FLAG, P4, "comment")
ENTRY(cufftDestroy, cufftDestroy, true, NO_FLAG, P4, "comment")
ENTRY(cufftEstimateSize1d, cufftEstimateSize1d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimateSize2d, cufftEstimateSize2d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimateSize3d, cufftEstimateSize3d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimateSizeMany, cufftEstimateSizeMany, false, NO_FLAG, P4, "comment")
ENTRY(cufftSetAutoAllocation, cufftSetAutoAllocation, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimate1d, cufftEstimate1d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimate2d, cufftEstimate2d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimate3d, cufftEstimate3d, false, NO_FLAG, P4, "comment")
ENTRY(cufftEstimateMany, cufftEstimateMany, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtSetWorkAreaPolicy, cufftXtSetWorkAreaPolicy, false, NO_FLAG, P4, "comment")
ENTRY(cufftMakePlanMany64, cufftMakePlanMany64, true, NO_FLAG, P4, "Successful")
ENTRY(cufftGetSizeMany64, cufftGetSizeMany64, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtGetSizeMany, cufftXtGetSizeMany, false, NO_FLAG, P4, "comment")
ENTRY(cufftSetWorkArea, cufftSetWorkArea, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtExec, cufftXtExec, false, NO_FLAG, P4, "comment")
ENTRY(cufftXtExecDescriptor, cufftXtExecDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cufftSetStream, cufftSetStream, true, NO_FLAG, P4, "comment")
ENTRY(cufftGetVersion, cufftGetVersion, true, NO_FLAG, P4, "comment")
ENTRY(cufftGetProperty, cufftGetProperty, true, NO_FLAG, P4, "comment")
