//===--------------- APINames_nvGRAPH.inc ---------------------------------===//
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
ENTRY(nvgraphGetProperty, nvgraphGetProperty, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphCreate, nvgraphCreate, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphDestroy, nvgraphDestroy, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphCreateGraphDescr, nvgraphCreateGraphDescr, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphDestroyGraphDescr, nvgraphDestroyGraphDescr, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSetGraphStructure, nvgraphSetGraphStructure, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphGetGraphStructure, nvgraphGetGraphStructure, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphConvertTopology, nvgraphConvertTopology, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphConvertGraph, nvgraphConvertGraph, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphAllocateEdgeData, nvgraphAllocateEdgeData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSetEdgeData, nvgraphSetEdgeData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphGetEdgeData, nvgraphGetEdgeData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphAllocateVertexData, nvgraphAllocateVertexData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSetVertexData, nvgraphSetVertexData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphGetVertexData, nvgraphGetVertexData, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphExtractSubgraphByVertex, nvgraphExtractSubgraphByVertex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphExtractSubgraphByEdge, nvgraphExtractSubgraphByEdge, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphWidestPath, nvgraphWidestPath, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSssp, nvgraphSssp, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSrSpmv, nvgraphSrSpmv, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphPagerank, nvgraphPagerank, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTriangleCount, nvgraphTriangleCount, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphStatusGetString, nvgraphStatusGetString, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphSpectralClustering, nvgraphSpectralClustering, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphAnalyzeClustering, nvgraphAnalyzeClustering, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalParameterInit, nvgraphTraversalParameterInit, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetDistancesIndex, nvgraphTraversalSetDistancesIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetDistancesIndex, nvgraphTraversalGetDistancesIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetPredecessorsIndex, nvgraphTraversalSetPredecessorsIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetPredecessorsIndex, nvgraphTraversalGetPredecessorsIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetEdgeMaskIndex, nvgraphTraversalSetEdgeMaskIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetEdgeMaskIndex, nvgraphTraversalGetEdgeMaskIndex, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetUndirectedFlag, nvgraphTraversalSetUndirectedFlag, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetUndirectedFlag, nvgraphTraversalGetUndirectedFlag, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetAlpha, nvgraphTraversalSetAlpha, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetAlpha, nvgraphTraversalGetAlpha, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalSetBeta, nvgraphTraversalSetBeta, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversalGetBeta, nvgraphTraversalGetBeta, false, NO_FLAG, P4, "comment")
ENTRY(nvgraphTraversal, nvgraphTraversal, false, NO_FLAG, P4, "comment")
