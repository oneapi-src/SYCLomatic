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
ENTRY(cutensorCreate, cutensorCreate, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorDestroy, cutensorDestroy, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreateTensorDescriptor, cutensorCreateTensorDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorDestroyTensorDescriptor, cutensorDestroyTensorDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorGetErrorString, cutensorGetErrorString, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorGetVersion, cutensorGetVersion, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorGetCudartVersion, cutensorGetCudartVersion, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Element-wise Operations
ENTRY(cutensorCreateElementwiseTrinary, cutensorCreateElementwiseTrinary, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorElementwiseTrinaryExecute, cutensorElementwiseTrinaryExecute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreateElementwiseBinary, cutensorCreateElementwiseBinary, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorElementwiseBinaryExecute, cutensorElementwiseBinaryExecute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreatePermutation, cutensorCreatePermutation, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorPermute, cutensorPermute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Contraction Operations
ENTRY(cutensorCreateContraction, cutensorCreateContraction, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorContract, cutensorContract, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreateContractionTrinary, cutensorCreateContractionTrinary, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorContractTrinary, cutensorContractTrinary, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Reduction Operations
ENTRY(cutensorCreateReduction, cutensorCreateReduction, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorReduce, cutensorReduce, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Generic Operation Functions
ENTRY(cutensorDestroyOperationDescriptor, cutensorDestroyOperationDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorOperationDescriptorGetAttribute, cutensorOperationDescriptorGetAttribute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorOperationDescriptorSetAttribute, cutensorOperationDescriptorSetAttribute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreatePlanPreference, cutensorCreatePlanPreference, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorDestroyPlanPreference, cutensorDestroyPlanPreference, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorPlanPreferenceSetAttribute, cutensorPlanPreferenceSetAttribute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorEstimateWorkspaceSize, cutensorEstimateWorkspaceSize, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorCreatePlan, cutensorCreatePlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorDestroyPlan, cutensorDestroyPlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorPlanGetAttribute, cutensorPlanGetAttribute, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Cache-related Operations
ENTRY(cutensorHandleResizePlanCache, cutensorHandleResizePlanCache, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorHandleReadPlanCacheFromFile, cutensorHandleReadPlanCacheFromFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorHandleWritePlanCacheToFile, cutensorHandleWritePlanCacheToFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorReadKernelCacheFromFile, cutensorReadKernelCacheFromFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorWriteKernelCacheToFile, cutensorWriteKernelCacheToFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// Logger Functions
ENTRY(cutensorLoggerSetCallback, cutensorLoggerSetCallback, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorLoggerSetFile, cutensorLoggerSetFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorLoggerOpenFile, cutensorLoggerOpenFile, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorLoggerSetLevel, cutensorLoggerSetLevel, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorLoggerSetMask, cutensorLoggerSetMask, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorLoggerForceDisable, cutensorLoggerForceDisable, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// cuTENSORMg - General Operations
ENTRY(cutensorMgCreate, cutensorMgCreate, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroy, cutensorMgDestroy, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCreateTensorDescriptor, cutensorMgCreateTensorDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyTensorDescriptor, cutensorMgDestroyTensorDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// cuTENSORMg - Copy Operations
ENTRY(cutensorMgCreateCopyDescriptor, cutensorMgCreateCopyDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyCopyDescriptor, cutensorMgDestroyCopyDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCopyGetWorkspace, cutensorMgCopyGetWorkspace, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCreateCopyPlan, cutensorMgCreateCopyPlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyCopyPlan, cutensorMgDestroyCopyPlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCopy, cutensorMgCopy, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")


// cuTENSORMg - Contraction Operations
ENTRY(cutensorMgCreateContractionDescriptor, cutensorMgCreateContractionDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyContractionDescriptor, cutensorMgDestroyContractionDescriptor, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCreateContractionFind, cutensorMgCreateContractionFind, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyContractionFind, cutensorMgDestroyContractionFind, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgContractionGetWorkspace, cutensorMgContractionGetWorkspace, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgCreateContractionPlan, cutensorMgCreateContractionPlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgDestroyContractionPlan, cutensorMgDestroyContractionPlan, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")
ENTRY(cutensorMgContraction, cutensorMgContraction, false, API_CALL_UNSUPPORTED, P4, "DPCT1007")

// clang-format on
