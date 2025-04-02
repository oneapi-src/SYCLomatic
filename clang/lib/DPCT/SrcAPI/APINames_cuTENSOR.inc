//===------------------------ APINames_cuTENSOR.inc -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// clang-format off

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

// Helper Functions
ENTRY(cutensorCreate, cutensorCreate, false, NO_FLAG, P4, "comment")
ENTRY(cutensorDestroy, cutensorDestroy, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreateTensorDescriptor, cutensorCreateTensorDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorDestroyTensorDescriptor, cutensorDestroyTensorDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorGetErrorString, cutensorGetErrorString, false, NO_FLAG, P4, "comment")
ENTRY(cutensorGetVersion, cutensorGetVersion, false, NO_FLAG, P4, "comment")
ENTRY(cutensorGetCudartVersion, cutensorGetCudartVersion, false, NO_FLAG, P4, "comment")


// Element-wise Operations
ENTRY(cutensorCreateElementwiseTrinary, cutensorCreateElementwiseTrinary, false, NO_FLAG, P4, "comment")
ENTRY(cutensorElementwiseTrinaryExecute, cutensorElementwiseTrinaryExecute, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreateElementwiseBinary, cutensorCreateElementwiseBinary, false, NO_FLAG, P4, "comment")
ENTRY(cutensorElementwiseBinaryExecute, cutensorElementwiseBinaryExecute, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreatePermutation, cutensorCreatePermutation, false, NO_FLAG, P4, "comment")
ENTRY(cutensorPermute, cutensorPermute, false, NO_FLAG, P4, "comment")


// Contraction Operations
ENTRY(cutensorCreateContraction, cutensorCreateContraction, false, NO_FLAG, P4, "comment")
ENTRY(cutensorContract, cutensorContract, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreateContractionTrinary, cutensorCreateContractionTrinary, false, NO_FLAG, P4, "comment")
ENTRY(cutensorContractTrinary, cutensorContractTrinary, false, NO_FLAG, P4, "comment")


// Reduction Operations
ENTRY(cutensorCreateReduction, cutensorCreateReduction, false, NO_FLAG, P4, "comment")
ENTRY(cutensorReduce, cutensorReduce, false, NO_FLAG, P4, "comment")


// Generic Operation Functions
ENTRY(cutensorDestroyOperationDescriptor, cutensorDestroyOperationDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorOperationDescriptorGetAttribute, cutensorOperationDescriptorGetAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cutensorOperationDescriptorSetAttribute, cutensorOperationDescriptorSetAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreatePlanPreference, cutensorCreatePlanPreference, false, NO_FLAG, P4, "comment")
ENTRY(cutensorDestroyPlanPreference, cutensorDestroyPlanPreference, false, NO_FLAG, P4, "comment")
ENTRY(cutensorPlanPreferenceSetAttribute, cutensorPlanPreferenceSetAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cutensorEstimateWorkspaceSize, cutensorEstimateWorkspaceSize, false, NO_FLAG, P4, "comment")
ENTRY(cutensorCreatePlan, cutensorCreatePlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorDestroyPlan, cutensorDestroyPlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorPlanGetAttribute, cutensorPlanGetAttribute, false, NO_FLAG, P4, "comment")


// Cache-related Operations
ENTRY(cutensorHandleResizePlanCache, cutensorHandleResizePlanCache, false, NO_FLAG, P4, "comment")
ENTRY(cutensorHandleReadPlanCacheFromFile, cutensorHandleReadPlanCacheFromFile, false, NO_FLAG, P4, "comment")
ENTRY(cutensorHandleWritePlanCacheToFile, cutensorHandleWritePlanCacheToFile, false, NO_FLAG, P4, "comment")
ENTRY(cutensorReadKernelCacheFromFile, cutensorReadKernelCacheFromFile, false, NO_FLAG, P4, "comment")
ENTRY(cutensorWriteKernelCacheToFile, cutensorWriteKernelCacheToFile, false, NO_FLAG, P4, "comment")


// Logger Functions
ENTRY(cutensorLoggerSetCallback, cutensorLoggerSetCallback, false, NO_FLAG, P4, "comment")
ENTRY(cutensorLoggerSetFile, cutensorLoggerSetFile, false, NO_FLAG, P4, "comment")
ENTRY(cutensorLoggerOpenFile, cutensorLoggerOpenFile, false, NO_FLAG, P4, "comment")
ENTRY(cutensorLoggerSetLevel, cutensorLoggerSetLevel, false, NO_FLAG, P4, "comment")
ENTRY(cutensorLoggerSetMask, cutensorLoggerSetMask, false, NO_FLAG, P4, "comment")
ENTRY(cutensorLoggerForceDisable, cutensorLoggerForceDisable, false, NO_FLAG, P4, "comment")


// cuTENSORMg - General Operations
ENTRY(cutensorMgCreate, cutensorMgCreate, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroy, cutensorMgDestroy, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCreateTensorDescriptor, cutensorMgCreateTensorDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyTensorDescriptor, cutensorMgDestroyTensorDescriptor, false, NO_FLAG, P4, "comment")


// cuTENSORMg - Copy Operations
ENTRY(cutensorMgCreateCopyDescriptor, cutensorMgCreateCopyDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyCopyDescriptor, cutensorMgDestroyCopyDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCopyGetWorkspace, cutensorMgCopyGetWorkspace, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCreateCopyPlan, cutensorMgCreateCopyPlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyCopyPlan, cutensorMgDestroyCopyPlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCopy, cutensorMgCopy, false, NO_FLAG, P4, "comment")


// cuTENSORMg - Contraction Operations
ENTRY(cutensorMgCreateContractionDescriptor, cutensorMgCreateContractionDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyContractionDescriptor, cutensorMgDestroyContractionDescriptor, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCreateContractionFind, cutensorMgCreateContractionFind, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyContractionFind, cutensorMgDestroyContractionFind, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgContractionGetWorkspace, cutensorMgContractionGetWorkspace, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgCreateContractionPlan, cutensorMgCreateContractionPlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgDestroyContractionPlan, cutensorMgDestroyContractionPlan, false, NO_FLAG, P4, "comment")
ENTRY(cutensorMgContraction, cutensorMgContraction, false, NO_FLAG, P4, "comment")

// clang-format on
