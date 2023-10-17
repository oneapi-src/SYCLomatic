//===--------------- TypeNames.inc ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY_TYPE(Type, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Type: TypeName|TypeName*
*    TypeName : Data Type Name
*    TypeName*: pointer Data Type Name
*
*  ProcessedOrNot:  false|true
*    false: this Type has not been processed.
*    true:  this Type has been processed, need check MigrationDesc field to
*           see if it has be migrated Commently.
*
*  Flag: Bit map for all kind of flag for this API.
*    default(all bits are 0): NO_FLAG
*    bit 0: Type_REMOVED
*    bit 1: Type_UNSUPPORTED
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
// CUDA Driver Library
ENTRY_TYPE(CUexternalMemory, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUexternalSemaphore, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUgraph, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUgraphExec, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUgraphNode, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUgraphicsResource, false, NO_FLAG, P4, "comment")

// CUDA Runtime Library
ENTRY_TYPE(cudaMemcpy3DParms*, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUDA_ARRAY_DESCRIPTOR*, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cudaKernelNodeParams, false, NO_FLAG, P4, "comment")

// cuDNN Library
ENTRY_TYPE(cudnnNanPropagation_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cudnnReduceTensorIndices_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cudnnIndicesType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cudnnConvolutionMode_t, false, NO_FLAG, P4, "comment")

// NVML (NVIDIA Management Library)
ENTRY_TYPE(nvmlReturn_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlDevice_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlUnit_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEventSet_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPciInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEccErrorCounts_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlUtilization_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlMemory_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBAR1Memory_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlProcessInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlNvLinkUtilizationControl_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBridgeChipInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBridgeChipHierarchy_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlValue_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlSample_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlViolationTime_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBridgeChipType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlNvLinkUtilizationCountUnits_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlNvLinkUtilizationCountPktTypes_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlNvLinkCapability_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlNvLinkErrorCounter_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGpuTopologyLevel_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlSamplingType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPcieUtilCounter_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlValueType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPerfPolicyType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEnableState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBrandType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlTemperatureThresholds_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlTemperatureSensors_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlComputeMode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlMemoryErrorType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEccCounterType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlClockType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlClockId_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlDriverModel_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPstates_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGpuOperationMode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlInforomObject_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlReturn_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlMemoryLocation_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPageRetirementCause_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlRestrictedAPI_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlFieldValue_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlHwbcEntry_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlLedState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlUnitInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPSUInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlUnitFanInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlUnitFanSpeeds_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlFanState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlLedColor_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlAccountingStats_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEncoderSessionInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEncoderType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlFBCStats_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlFBCSessionInfo_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlFBCSessionType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlDetachGpuState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlPcieLinkState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlEventData_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGpuVirtualizationMode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlHostVgpuMode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuVmIdType_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuGuestInfoState_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGridLicenseFeatureCode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuInstanceUtilizationSample_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuProcessUtilizationSample_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlProcessUtilizationSample_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGridLicensableFeature_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlGridLicensableFeatures_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuVersion_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuMetadata_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuPgpuMetadata_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuPgpuCompatibility_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuVmCompatibility_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlVgpuPgpuCompatibilityLimitCode_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(nvmlBlacklistDeviceInfo_t, false, NO_FLAG, P4, "comment")

// cooperative groups
ENTRY_TYPE(cooperative_groups::__v1::experimental::block_tile_memory, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cooperative_groups::__v1::block_tile_memory, true, NO_FLAG, P4, "Successful")
ENTRY_TYPE(cooperative_groups::__v1::thread_group, true, NO_FLAG, P4, "Successful")
ENTRY_TYPE(cooperative_groups::__v1::cluster_group, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cooperative_groups::__v1::gird_group, true, NO_FLAG, P4, "Successful")
ENTRY_TYPE(cooperative_groups::__v1::multi_grid_group, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(cooperative_groups::__v1::thread_block_tile, true, NO_FLAG, P4, "Successful")
ENTRY_TYPE(cooperative_groups::__v1::coalesced_group, false, NO_FLAG, P4, "comment")

//CUmem
ENTRY_TYPE(CUmemGenericAllocationHandle, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUmemAllocationProp, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(CUmemAccessDesc, false, NO_FLAG, P4, "comment")

// cuRand
ENTRY_TYPE(curandStateScrambledSobol64_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateSobol64_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateScrambledSobol32_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateSobol32_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateMtgp32_t, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateScrambledSobol64, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateSobol64, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateScrambledSobol32, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateSobol32, false, NO_FLAG, P4, "comment")
ENTRY_TYPE(curandStateMtgp32, false, NO_FLAG, P4, "comment")
