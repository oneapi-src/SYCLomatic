//===---APINames_CUB.inc--------------------------------------------------===//
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
* ENTRY(cub::Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
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

/****************************************************************************
*
* ENTRY_MEMBER_FUNCTION(cub::ObjectName, Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  ObjectName: name of Object of underlying
*
*  Interface APIName: name of library Interface API available in official document
*  APIName: name of library API
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

//Warp Level
ENTRY(cub::ShuffleUp, cub::ShuffleUp, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ShuffleDown, cub::ShuffleDown, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ShuffleIndex, cub::ShuffleIndex, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, InclusiveSum, InclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, ExclusiveSum, ExclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, InclusiveScan, InclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, ExclusiveScan, ExclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, Scan, Scan, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, Broadcast, Broadcast, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, Sum, Sum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, HeadSegmentedSum, HeadSegmentedSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, TailSegmentedSum, TailSegmentedSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, HeadSegmentedReduce, HeadSegmentedReduce, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, TailSegmentedReduce, TailSegmentedReduce, false, NO_FLAG, P4, "Comment")

//Block Level
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, FlagHeads, FlagHeads, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, FlagTails, FlagTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, FlagHeadsAndTails, FlagHeadsAndTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, StripedToBlocked, StripedToBlocked, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, BlockedToStriped, BlockedToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, WarpStripedToBlocked, WarpStripedToBlocked, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, BlockedToWarpStriped, BlockedToWarpStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, ScatterToBlocked, ScatterToBlocked, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, ScatterToStriped, ScatterToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, ScatterToStripedGuarded, ScatterToStripedGuarded, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, ScatterToStripedFlagged, ScatterToStripedFlagged, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, InitHistogram, InitHistogram, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, Histogram, Histogram, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, Composite, Composite, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockLoad, Load, Load, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockStore, Store, Store, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, Sort, Sort, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, SortDescending, SortDescending, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, SortBlockedToStriped, SortBlockedToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, SortDescendingBlockedToStriped, SortDescendingBlockedToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockReduce, Sum, Sum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, ExclusiveSum, ExclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, ExclusiveScan, ExclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, InclusiveSum, InclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, InclusiveScan, InclusiveScan, true, NO_FLAG, P4, "Successful")

// Device Level
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, HistogramEven, HistogramEven, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, MultiHistogramEven, MultiHistogramEven, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, HistogramRange, HistogramRange, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, MultiHistogramRange, MultiHistogramRange, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DevicePartition, Flagged, Flagged, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DevicePartition, If, If, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, SortPairs, SortPairs, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, SortPairsDescending, SortPairsDescending, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, SortKeys, SortKeys, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, SortKeysDescending, SortKeysDescending, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, Reduce, Reduce, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, Sum, Sum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, Min, Min, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, ArgMin, ArgMin, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, Max, Max, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, ArgMax, ArgMax, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, ReduceByKey, ReduceByKey, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRunLengthEncode, Encode, Encode, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceRunLengthEncode, NonTrivialRuns, NonTrivialRuns, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, ExclusiveSum, ExclusiveSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, ExclusiveScan, ExclusiveScan, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, InclusiveSum, InclusiveSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, InclusiveScan, InclusiveScan, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, Flagged, Flagged, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, If, If, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, Unique, Unique, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSpmv, CsrMV, CsrMV, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, SortPairs, SortPairs, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, SortPairsDescending, SortPairsDescending, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, SortKeys, SortKeys, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, SortKeysDescending, SortKeysDescending, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, Reduce, Reduce, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, Sum, Sum, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, Min, Min, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, ArgMin, ArgMin, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, Max, Max, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, ArgMax, ArgMax, false, NO_FLAG, P4, "Comment")

// Thread and thread block I/O
ENTRY(cub::ThreadLoad, cub::ThreadLoad, true, NO_FLAG, P4, "Successful")
ENTRY(cub::ThreadStore, cub::ThreadStore, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LoadDirectBlocked, cub::LoadDirectBlocked, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LoadDirectBlockedVectorized, cub::LoadDirectBlockedVectorized, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectBlocked, cub::StoreDirectBlocked, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectBlockedVectorized, cub::StoreDirectBlockedVectorized, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LoadDirectStriped, cub::LoadDirectStriped, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectStriped, cub::StoreDirectStriped, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LoadDirectWarpStriped, cub::LoadDirectWarpStriped, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectWarpStriped, cub::StoreDirectWarpStriped, false, NO_FLAG, P4, "Comment")

// PTX intrinsics
ENTRY(cub::SHR_ADD, cub::SHR_ADD, false, NO_FLAG, P4, "Comment")
ENTRY(cub::SHL_ADD, cub::SHL_ADD, false, NO_FLAG, P4, "Comment")
ENTRY(cub::BFE, cub::BFE, false, NO_FLAG, P4, "Comment")
ENTRY(cub::BFI, cub::BFI, false, NO_FLAG, P4, "Comment")
ENTRY(cub::IADD3, cub::IADD3, false, NO_FLAG, P4, "Comment")
ENTRY(cub::PRMT, cub::PRMT, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ThreadExit, cub::ThreadExit, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ThreadTrap, cub::ThreadTrap, false, NO_FLAG, P4, "Comment")
ENTRY(cub::RowMajorTid, cub::RowMajorTid, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneId, cub::LaneId, false, NO_FLAG, P4, "Comment")
ENTRY(cub::WarpId, cub::WarpId, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskLt, cub::LaneMaskLt, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskLe, cub::LaneMaskLe, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskGt, cub::LaneMaskGt, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskGe, cub::LaneMaskGe, false, NO_FLAG, P4, "Comment")

// Device, kernel, and storage management
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, SetMaxCachedBytes, SetMaxCachedBytes, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, DeviceAllocate, DeviceAllocate, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, DeviceFree, DeviceFree, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, FreeAllCached, FreeAllCached, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Debug, cub::Debug, false, NO_FLAG, P4, "Comment")
ENTRY(cub::PtxVersion, cub::PtxVersion, false, NO_FLAG, P4, "Comment")
ENTRY(cub::SmVersion, cub::SmVersion, false, NO_FLAG, P4, "Comment")

// Fancy iterators
ENTRY_MEMBER_FUNCTION(cub::ArgIndexInputIterator, normalize, normalize, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexObjInputIterator, BindTexture, BindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexObjInputIterator, UnbindTexture, UnbindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexRefInputIterator, BindTexture, BindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexRefInputIterator, UnbindTexture, UnbindTexture, false, NO_FLAG, P4, "Comment")
