//===--------------- APINames_Graph.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

ENTRY(cudaGraph_t, cudaGraph_t, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphExec_t, cudaGraphExec_t, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphNode_t, cudaGraphNode_t, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaDeviceGetGraphMemAttribute, cudaDeviceGetGraphMemAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cudaDeviceGraphMemTrim, cudaDeviceGraphMemTrim, false, NO_FLAG, P4, "comment")
ENTRY(cudaDeviceSetGraphMemAttribute, cudaDeviceSetGraphMemAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cudaGetCurrentGraphExec, cudaGetCurrentGraphExec, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddChildGraphNode, cudaGraphAddChildGraphNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddDependencies, cudaGraphAddDependencies, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphAddDependencies_v2, cudaGraphAddDependencies_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddEmptyNode, cudaGraphAddEmptyNode, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphAddEventRecordNode, cudaGraphAddEventRecordNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddEventWaitNode, cudaGraphAddEventWaitNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddExternalSemaphoresSignalNode, cudaGraphAddExternalSemaphoresSignalNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddExternalSemaphoresWaitNode, cudaGraphAddExternalSemaphoresWaitNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddHostNode, cudaGraphAddHostNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddKernelNode, cudaGraphAddKernelNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemAllocNode, cudaGraphAddMemAllocNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemFreeNode, cudaGraphAddMemFreeNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemcpyNode1D, cudaGraphAddMemcpyNode1D, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemcpyNodeFromSymbol, cudaGraphAddMemcpyNodeFromSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemcpyNodeToSymbol, cudaGraphAddMemcpyNodeToSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddMemsetNode, cudaGraphAddMemsetNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddNode, cudaGraphAddNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphAddNode_v2, cudaGraphAddNode_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphChildGraphNodeGetGraph, cudaGraphChildGraphNodeGetGraph, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphClone, cudaGraphClone, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphConditionalHandleCreate, cudaGraphConditionalHandleCreate, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphCreate, cudaGraphCreate, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphDebugDotPrint, cudaGraphDebugDotPrint, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphDestroy, cudaGraphDestroy, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphDestroyNode, cudaGraphDestroyNode, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphEventRecordNodeGetEvent, cudaGraphEventRecordNodeGetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphEventRecordNodeSetEvent, cudaGraphEventRecordNodeSetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphEventWaitNodeGetEvent, cudaGraphEventWaitNodeGetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphEventWaitNodeSetEvent, cudaGraphEventWaitNodeSetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecChildGraphNodeSetParams, cudaGraphExecChildGraphNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecDestroy, cudaGraphExecDestroy, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphExecEventRecordNodeSetEvent, cudaGraphExecEventRecordNodeSetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecEventWaitNodeSetEvent, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecExternalSemaphoresSignalNodeSetParams, cudaGraphExecExternalSemaphoresSignalNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecExternalSemaphoresWaitNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecGetFlags, cudaGraphExecGetFlags, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecHostNodeSetParams, cudaGraphExecHostNodeSetParams,false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecKernelNodeSetParams, cudaGraphExecKernelNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecMemcpyNodeSetParams, cudaGraphExecMemcpyNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecMemcpyNodeSetParams1D, cudaGraphExecMemcpyNodeSetParams1D, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecMemcpyNodeSetParamsFromSymbol, cudaGraphExecMemcpyNodeSetParamsFromSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecMemcpyNodeSetParamsToSymbol, cudaGraphExecMemcpyNodeSetParamsToSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecMemsetNodeSetParams, cudaGraphExecMemsetNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecNodeSetParams, cudaGraphExecNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExecUpdate, cudaGraphExecUpdate, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphExternalSemaphoresSignalNodeGetParams, cudaGraphExternalSemaphoresSignalNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExternalSemaphoresSignalNodeSetParams, cudaGraphExternalSemaphoresSignalNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExternalSemaphoresWaitNodeGetParams, cudaGraphExternalSemaphoresWaitNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphExternalSemaphoresWaitNodeSetParams, cudaGraphExternalSemaphoresWaitNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphGetEdges, cudaGraphGetEdges, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphGetEdges_v2, cudaGraphGetEdges_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphGetNodes, cudaGraphGetNodes, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphGetRootNodes, cudaGraphGetRootNodes, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphHostNodeGetParams, cudaGraphHostNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphHostNodeSetParams, cudaGraphHostNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphInstantiate, cudaGraphInstantiate, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphInstantiateWithFlags, cudaGraphInstantiateWithFlags, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphInstantiateWithParams, cudaGraphInstantiateWithParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeCopyAttributes, cudaGraphKernelNodeCopyAttributes, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeGetAttribute, cudaGraphKernelNodeGetAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeGetParams, cudaGraphKernelNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeSetAttribute, cudaGraphKernelNodeSetAttribute, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeSetEnabled, cudaGraphKernelNodeSetEnabled, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeSetGridDim, cudaGraphKernelNodeSetGridDim, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeSetParam, cudaGraphKernelNodeSetParam, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeSetParams, cudaGraphKernelNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphKernelNodeUpdatesApply, cudaGraphKernelNodeUpdatesApply, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphLaunch, cudaGraphLaunch, true, NO_FLAG, P4, "Successful/DPCT1119")
ENTRY(cudaGraphMemAllocNodeGetParams, cudaGraphMemAllocNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemFreeNodeGetParams, cudaGraphMemFreeNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemcpyNodeGetParams, cudaGraphMemcpyNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemcpyNodeSetParams, cudaGraphMemcpyNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemcpyNodeSetParams1D, cudaGraphMemcpyNodeSetParams1D, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemcpyNodeSetParamsFromSymbol, cudaGraphMemcpyNodeSetParamsFromSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemcpyNodeSetParamsToSymbol, cudaGraphMemcpyNodeSetParamsToSymbol, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemsetNodeGetParams, cudaGraphMemsetNodeGetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphMemsetNodeSetParams, cudaGraphMemsetNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeFindInClone, cudaGraphNodeFindInClone, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetDependencies, cudaGraphNodeGetDependencies, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetDependencies_v2, cudaGraphNodeGetDependencies_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetDependentNodes, cudaGraphNodeGetDependentNodes, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetDependentNodes_v2, cudaGraphNodeGetDependentNodes_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetEnabled, cudaGraphNodeGetEnabled, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeGetType, cudaGraphNodeGetType, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeSetEnabled, cudaGraphNodeSetEnabled, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphNodeSetParams, cudaGraphNodeSetParams, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphReleaseUserObject, cudaGraphReleaseUserObject, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphRemoveDependencies, cudaGraphRemoveDependencies, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphRemoveDependencies_v2, cudaGraphRemoveDependencies_v2, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphRetainUserObject, cudaGraphRetainUserObject, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphSetConditional, cudaGraphSetConditional, false, NO_FLAG, P4, "comment")
ENTRY(cudaGraphUpload, cudaGraphUpload, false, NO_FLAG, P4, "comment")
ENTRY(cudaUserObjectCreate, cudaUserObjectCreate, false, NO_FLAG, P4, "comment")
ENTRY(cudaUserObjectRelease, cudaUserObjectRelease, false, NO_FLAG, P4, "comment")
ENTRY(cudaUserObjectRetain, cudaUserObjectRetain, false, NO_FLAG, P4, "comment")