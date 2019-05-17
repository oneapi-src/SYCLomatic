/****************************************************************************
*
* ENTRY(APINAME,  ProcessedOrNot, Priority, MigrationDesc)
*  APIName: name of CUDA API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated successfully.
*
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
//cuGRAPH call api
ENTRY(nvgraphGetProperty, false, P4, "comment")
ENTRY(nvgraphCreate, false, P4, "comment")
ENTRY(nvgraphDestroy, false, P4, "comment")
ENTRY(nvgraphCreateGraphDescr, false, P4, "comment")
ENTRY(nvgraphDestroyGraphDescr, false, P4, "comment")
ENTRY(nvgraphSetGraphStructure, false, P4, "comment")
ENTRY(nvgraphGetGraphStructure, false, P4, "comment")
ENTRY(nvgraphConvertTopology, false, P4, "comment")
ENTRY(nvgraphConvertGraph, false, P4, "comment")
ENTRY(nvgraphAllocateEdgeData, false, P4, "comment")
ENTRY(nvgraphSetEdgeData, false, P4, "comment")
ENTRY(nvgraphGetEdgeData, false, P4, "comment")
ENTRY(nvgraphAllocateVertexData, false, P4, "comment")
ENTRY(nvgraphSetVertexData, false, P4, "comment")
ENTRY(nvgraphGetVertexData, false, P4, "comment")
ENTRY(nvgraphExtractSubgraphByVertex, false, P4, "comment")
ENTRY(nvgraphExtractSubgraphByEdge, false, P4, "comment")
ENTRY(nvgraphWidestPath, false, P4, "comment")
ENTRY(nvgraphSssp, false, P4, "comment")
ENTRY(nvgraphSrSpmv, false, P4, "comment")
ENTRY(nvgraphPagerank, false, P4, "comment")
ENTRY(nvgraphTriangleCount, false, P4, "comment")
ENTRY(nvgraphStatusGetString, false, P4, "comment")
ENTRY(nvgraphSpectralClustering, false, P4, "comment")
ENTRY(nvgraphAnalyzeClustering, false, P4, "comment")
ENTRY(nvgraphTraversalParameterInit, false, P4, "comment")
ENTRY(nvgraphTraversalSetDistancesIndex, false, P4, "comment")
ENTRY(nvgraphTraversalGetDistancesIndex, false, P4, "comment")
ENTRY(nvgraphTraversalSetPredecessorsIndex, false, P4, "comment")
ENTRY(nvgraphTraversalGetPredecessorsIndex, false, P4, "comment")
ENTRY(nvgraphTraversalSetEdgeMaskIndex, false, P4, "comment")
ENTRY(nvgraphTraversalGetEdgeMaskIndex, false, P4, "comment")
ENTRY(nvgraphTraversalSetUndirectedFlag, false, P4, "comment")
ENTRY(nvgraphTraversalGetUndirectedFlag, false, P4, "comment")
ENTRY(nvgraphTraversalSetAlpha, false, P4, "comment")
ENTRY(nvgraphTraversalGetAlpha, false, P4, "comment")
ENTRY(nvgraphTraversalSetBeta, false, P4, "comment")
ENTRY(nvgraphTraversalGetBeta, false, P4, "comment")
ENTRY(nvgraphTraversal, false, P4, "comment")
