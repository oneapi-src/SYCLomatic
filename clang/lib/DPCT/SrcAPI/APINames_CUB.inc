//===--------------- APINames_CUB.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
* ENTRY_MEMBER_FUNCTION(Interface ObjectName, cub::ObjectName, Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface ObjectName: name of Object of underlying which user used.
*  ObjectName: name of Object of underlying which library real defined.
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
// clang-format off

//Warp Level
ENTRY(cub::ShuffleUp, cub::ShuffleUp, true, NO_FLAG, P4, "Successful")
ENTRY(cub::ShuffleDown, cub::ShuffleDown, true, NO_FLAG, P4, "Successful")
ENTRY(cub::ShuffleIndex, cub::ShuffleIndex, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, InclusiveSum, InclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, ExclusiveSum, ExclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, InclusiveScan, InclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, ExclusiveScan, ExclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, Scan, Scan, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpScan, cub::WarpScan, Broadcast, Broadcast, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, Sum, Sum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, HeadSegmentedSum, HeadSegmentedSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, TailSegmentedSum, TailSegmentedSum, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, HeadSegmentedReduce, HeadSegmentedReduce, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpReduce, cub::WarpReduce, TailSegmentedReduce, TailSegmentedReduce, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpExchange, cub::WarpExchange, BlockedToStriped, BlockedToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpExchange, cub::WarpExchange, StripedToBlocked, StripedToBlocked, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpExchange, cub::WarpExchange, ScatterToStriped, ScatterToStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpLoad, cub::WarpLoad, Load, Load, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::WarpStore, cub::WarpLoad, Store, Store, false, NO_FLAG, P4, "Comment")

//Block Level
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, SubtractLeft, SubtractLeft, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, SubtractLeftPartialTile, SubtractLeftPartialTile, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, SubtractRight, SubtractRight, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, SubtractRightPartialTile, SubtractRightPartialTile, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, FlagHeads, FlagHeads, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, FlagTails, FlagTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockAdjacentDifference, cub::BlockAdjacentDifference, FlagHeadsAndTails, FlagHeadsAndTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagHeads, FlagHeads, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagTails, FlagTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagHeadsAndTails, FlagHeadsAndTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, StripedToBlocked, StripedToBlocked, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, BlockedToStriped, BlockedToStriped, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, WarpStripedToBlocked, WarpStripedToBlocked, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, BlockedToWarpStriped, BlockedToWarpStriped, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, ScatterToBlocked, ScatterToBlocked, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, ScatterToStriped, ScatterToStriped, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, ScatterToStripedGuarded, ScatterToStripedGuarded, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockExchange, cub::BlockExchange, ScatterToStripedFlagged, ScatterToStripedFlagged, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, cub::BlockHistogram, InitHistogram, InitHistogram, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, cub::BlockHistogram, Histogram, Histogram, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockHistogram, cub::BlockHistogram, Composite, Composite, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockLoad, cub::BlockLoad, Load, Load, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockStore, cub::BlockStore, Store, Store, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, cub::BlockRadixSort, Sort, Sort, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, cub::BlockRadixSort, SortDescending, SortDescending, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, cub::BlockRadixSort, SortBlockedToStriped, SortBlockedToStriped, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixSort, cub::BlockRadixSort, SortDescendingBlockedToStriped, SortDescendingBlockedToStriped, true, NO_FLAG, P4, "Partial")
ENTRY_MEMBER_FUNCTION(cub::BlockReduce, cub::BlockReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockReduce, cub::BlockReduce, Sum, Sum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, cub::BlockScan, ExclusiveSum, ExclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, cub::BlockScan, ExclusiveScan, ExclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, cub::BlockScan, InclusiveSum, InclusiveSum, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockScan, cub::BlockScan, InclusiveScan, InclusiveScan, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::BlockShuffle, cub::BlockShuffle, Offset, Offset, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockShuffle, cub::BlockShuffle, Rotate, Rotate, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockShuffle, cub::BlockShuffle, Up, Up, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockShuffle, cub::BlockShuffle, Down, Down, true, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagHeads, FlagHeads, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagTails, FlagTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockDiscontinuity, cub::BlockDiscontinuity, FlagHeadsAndTails, FlagHeadsAndTails, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockMergeSortStrategy, cub::BlockMergeSortStrategy, get_linear_tid, get_linear_tid, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockMergeSortStrategy, cub::BlockMergeSortStrategy, Sort, Sort, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockMergeSortStrategy, cub::BlockMergeSortStrategy, StableSort, StableSort, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixRank, cub::BlockRadixRank, RankKeys, RankKeys, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixRankMatch, cub::BlockRadixRankMatch, RankKeys, RankKeys, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BlockRadixRankMatch, cub::BlockRadixRankMatch, CallBack, CallBack, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::RadixSortTwiddle, cub::RadixSortTwiddle, In, In, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::RadixSortTwiddle, cub::RadixSortTwiddle, Out, Out, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::RadixSortTwiddle, cub::RadixSortTwiddle, DefaultKey, DefaultKey, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::ShiftDigitExtractor, cub::ShiftDigitExtractor, Digit, Digit, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BFEDigitExtractor , cub::BFEDigitExtractor , Digit, Digit, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BFEDigitExtractor , cub::BFEDigitExtractor, ProcessFloatMinusZero, ProcessFloatMinusZero, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::BaseDigitExtractor , cub::BaseDigitExtractor, ProcessFloatMinusZero, ProcessFloatMinusZero, false, NO_FLAG, P4, "Comment")


// Device Level
ENTRY_MEMBER_FUNCTION(cub::DeviceAdjacentDifference, cub::DeviceAdjacentDifference, SubtractLeftCopy, SubtractLeftCopy, false, NO_FLAGE, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceAdjacentDifference, cub::DeviceAdjacentDifference, SubtractLeft, SubtractLeft, false, NO_FLAGE, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceAdjacentDifference, cub::DeviceAdjacentDifference, SubtractRightCopy, SubtractRightCopy, false, NO_FLAGE, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceAdjacentDifference, cub::DeviceAdjacentDifference, SubtractRight, SubtractRight, false, NO_FLAGE, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, cub::DeviceHistogram, HistogramEven, HistogramEven, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, cub::DeviceHistogram, MultiHistogramEven, MultiHistogramEven, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, cub::DeviceHistogram, HistogramRange, HistogramRange, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceHistogram, cub::DeviceHistogram, MultiHistogramRange, MultiHistogramRange, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DevicePartition, cub::DevicePartition, Flagged, Flagged, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DevicePartition, cub::DevicePartition, If, If, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, SortPairs, SortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, SortPairsCopy, SortPairsCopy, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, SortKeys, SortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, SortKeysCopy, SortKeysCopy, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, StableSortPairs, StableSortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceMergeSort, cub::DeviceMergeSort, StableSortKeys, StableSortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, cub::DeviceRadixSort, SortPairs, SortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, cub::DeviceRadixSort, SortPairsDescending, SortPairsDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, cub::DeviceRadixSort, SortKeys, SortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRadixSort, cub::DeviceRadixSort, SortKeysDescending, SortKeysDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, Sum, Sum, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, Min, Min, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, ArgMin, ArgMin, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, Max, Max, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, ArgMax, ArgMax, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceReduce, cub::DeviceReduce, ReduceByKey, ReduceByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRunLengthEncode, cub::DeviceRunLengthEncode, Encode, Encode, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceRunLengthEncode, cub::DeviceRunLengthEncode, NonTrivialRuns, NonTrivialRuns, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, ExclusiveSum, ExclusiveSum, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, ExclusiveScan, ExclusiveScan, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, InclusiveSum, InclusiveSum, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, InclusiveScan, InclusiveScan, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, InclusiveSumByKey, InclusiveSumByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, ExclusiveSumByKey, ExclusiveSumByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, InclusiveScanByKey, InclusiveScanByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceScan, cub::DeviceScan, ExclusiveScanByKey, ExclusiveScanByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, cub::DeviceSelect, Flagged, Flagged, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, cub::DeviceSelect, If, If, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, cub::DeviceSelect, Unique, Unique, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSelect, cub::DeviceSelect, UniqueByKey, UniqueByKey, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSpmv, cub::DeviceSpmv, CsrMV, CsrMV, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, cub::DeviceSegmentedRadixSort, SortPairs, SortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, cub::DeviceSegmentedRadixSort, SortPairsDescending, SortPairsDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, cub::DeviceSegmentedRadixSort, SortKeys, SortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedRadixSort, cub::DeviceSegmentedRadixSort, SortKeysDescending, SortKeysDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, Reduce, Reduce, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027/DPCT1091/DPCT1092")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, Sum, Sum, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027/DPCT1092")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, Min, Min, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027/DPCT1092")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, ArgMin, ArgMin, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, Max, Max, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027/DPCT1092")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedReduce, cub::DeviceSegmentedReduce, ArgMax, ArgMax, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, SortKeys, SortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, SortKeysDescending, SortKeysDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, StableSortKeys, StableSortKeys, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, StableSortKeysDescending, StableSortKeysDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, SortPairs, SortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, SortPairsDescending, SortPairsDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, StableSortPairs, StableSortPairs, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceSegmentedSort, cub::DeviceSegmentedSort, StableSortPairsDescending, StableSortPairsDescending, true, NO_FLAG, P4, "Successful: DPCT1026/DPCT1027")
ENTRY_MEMBER_FUNCTION(cub::DeviceCopy, cub::DeviceCopy, Batched, Batched, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceFor, cub::DeviceFor, Bulk, Bulk, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceFor, cub::DeviceFor, ForEachN, ForEachN, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceFor, cub::DeviceFor, ForEach, ForEach, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceFor, cub::DeviceFor, ForEachCopy, ForEachCopy, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceFor, cub::DeviceFor, ForEachCopyN, ForEachCopyN, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::DeviceMemcpy, cub::DeviceMemcpy, Batched, Batched, false, NO_FLAG, P4, "Comment")

// Grid Level
ENTRY_MEMBER_FUNCTION(cub::GridBarrier, cub::GridBarrier, Sync, Sync, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridBarrierLifetime, cub::GridBarrierLifetime, Sync, Sync, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridBarrierLifetime, cub::GridBarrierLifetime, HostReset, HostReset, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridBarrierLifetime, cub::GridBarrierLifetime, Setup, Setup, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, FillAndResetDrain, FillAndResetDrain, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, ResetDrain, ResetDrain, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, ResetFill, ResetFill, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, FillSize, FillSize, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, Drain, Drain, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, Fill, Fill, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridQueue, cub::GridQueue, AllocationSize, AllocationSize, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridEvenShare, cub::GridEvenShare, DispatchInit, DispatchInit, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::GridEvenShare, cub::GridEvenShare, BlockInit, BlockInit, false, NO_FLAG, P4, "Comment")

// Thread and thread block I/O
ENTRY(cub::ThreadLoad, cub::ThreadLoad, true, NO_FLAG, P4, "Successful")
ENTRY(cub::ThreadStore, cub::ThreadStore, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LoadDirectBlocked, cub::LoadDirectBlocked, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LoadDirectBlockedVectorized, cub::LoadDirectBlockedVectorized, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectBlocked, cub::StoreDirectBlocked, true, NO_FLAG, P4, "Successful")
ENTRY(cub::StoreDirectBlockedVectorized, cub::StoreDirectBlockedVectorized, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LoadDirectStriped, cub::LoadDirectStriped, true, NO_FLAG, P4, "Successful")
ENTRY(cub::StoreDirectStriped, cub::StoreDirectStriped, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LoadDirectWarpStriped, cub::LoadDirectWarpStriped, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StoreDirectWarpStriped, cub::StoreDirectWarpStriped, false, NO_FLAG, P4, "Comment")

// PTX intrinsics
ENTRY(cub::SHR_ADD, cub::SHR_ADD, true, NO_FLAG, P4, "Successful")
ENTRY(cub::SHL_ADD, cub::SHL_ADD, true, NO_FLAG, P4, "Successful")
ENTRY(cub::BFE, cub::BFE, true, NO_FLAG, P4, "Successful")
ENTRY(cub::BFI, cub::BFI, true, NO_FLAG, P4, "Successful")
ENTRY(cub::IADD3, cub::IADD3, true, NO_FLAG, P4, "Successful")
ENTRY(cub::PRMT, cub::PRMT, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ThreadExit, cub::ThreadExit, false, NO_FLAG, P4, "Comment")
ENTRY(cub::ThreadTrap, cub::ThreadTrap, false, NO_FLAG, P4, "Comment")
ENTRY(cub::RowMajorTid, cub::RowMajorTid, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LaneId, cub::LaneId, true, NO_FLAG, P4, "Successful")
ENTRY(cub::WarpId, cub::WarpId, true, NO_FLAG, P4, "Successful")
ENTRY(cub::LaneMaskLt, cub::LaneMaskLt, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskLe, cub::LaneMaskLe, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskGt, cub::LaneMaskGt, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LaneMaskGe, cub::LaneMaskGe, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Swap, cub::Swap, false, NO_FLAG, P4, "Comment")
ENTRY(cub::UpperBound, cub::UpperBound, false, NO_FLAG, P4, "Comment")
ENTRY(cub::WarpMask, cub::WarpMask, false, NO_FLAG, P4, "Comment")
ENTRY(cub::StableOddEvenSort, cub::StableOddEvenSort, false, NO_FLAG, P4, "Comment")
ENTRY(cub::SerialMerge, cub::SerialMerge, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Nominal8BItemsToItems, cub::Nominal8BItemsToItems, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Nominal4BItemsToItemsCombined, cub::Nominal4BItemsToItemsCombined, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Nominal4BItemsToItems, cub::Nominal4BItemsToItems, false, NO_FLAG, P4, "Comment")
ENTRY(cub::MidPoint, cub::MidPoint, false, NO_FLAG, P4, "Comment")
ENTRY(cub::MergePathSearch, cub::MergePathSearch, false, NO_FLAG, P4, "Comment")
ENTRY(cub::MergePath, cub::MergePath, false, NO_FLAG, P4, "Comment")
ENTRY(cub::MatchAny, cub::MatchAny, false, NO_FLAG, P4, "Comment")
ENTRY(cub::MakeBinaryFlip, cub::MakeBinaryFlip, false, NO_FLAG, P4, "Comment")
ENTRY(cub::LowerBound, cub::LowerBound, false, NO_FLAG, P4, "Comment")
ENTRY(cub::GetPerDeviceAttributeCache, cub::GetPerDeviceAttributeCache, false, NO_FLAG, P4, "Comment")
ENTRY(cub::DivideAndRoundUp, cub::DivideAndRoundUp, false, NO_FLAG, P4, "Comment")

// Device, kernel, and storage management
ENTRY_MEMBER_FUNCTION(cub::KernelConfig, cub::KernelConfig, Init, Init, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, cub::CachingDeviceAllocator, SetMaxCachedBytes, SetMaxCachedBytes, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, cub::CachingDeviceAllocator, DeviceAllocate, DeviceAllocate, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, cub::CachingDeviceAllocator, DeviceFree, DeviceFree, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::CachingDeviceAllocator, cub::CachingDeviceAllocator, FreeAllCached, FreeAllCached, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::ChainedPolicy, cub::ChainedPolicy, Invoke, Invoke, false, NO_FLAG, P4, "Comment")
ENTRY(cub::Debug, cub::Debug, true, NO_FLAG, P4, "Comment")
ENTRY(cub::MaxSmOccupancy, cub::MaxSmOccupancy, false, NO_FLAG, P4, "Comment")
ENTRY(cub::PtxVersion, cub::PtxVersion, true, NO_FLAG, P4, "Successful")
ENTRY(cub::SmVersion, cub::SmVersion, true, NO_FLAG, P4, "Successful")
ENTRY(cub::CurrentDevice, cub::CurrentDevice, true, NO_FLAG, P4, "Successful")
ENTRY(cub::DeviceCount, cub::DeviceCount, true, NO_FLAG, P4, "Successful")
ENTRY(cub::DeviceCountUncached, cub::DeviceCountUncached, true, NO_FLAG, P4, "Successful")
ENTRY(cub::DeviceCountCachedValue, cub::DeviceCountCachedValue, true, NO_FLAG, P4, "Successful")
ENTRY(cub::PtxVersionUncached, cub::PtxVersionUncached, true, NO_FLAG, P4, "Successful")
ENTRY(cub::SmVersionUncached, cub::SmVersionUncached, true, NO_FLAG, P4, "Successful")
ENTRY(cub::SyncStream, cub::SyncStream, true, NO_FLAG, P4, "Successful")

// Fancy iterators
ENTRY_MEMBER_FUNCTION(cub::ArgIndexInputIterator, cub::ArgIndexInputIterator, normalize, normalize, true, NO_FLAG, P4, "Successful")
ENTRY_MEMBER_FUNCTION(cub::TexObjInputIterator, cub::TexObjInputIterator, BindTexture, BindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexObjInputIterator, cub::TexObjInputIterator, UnbindTexture, UnbindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexRefInputIterator, cub::TexRefInputIterator, BindTexture, BindTexture, false, NO_FLAG, P4, "Comment")
ENTRY_MEMBER_FUNCTION(cub::TexRefInputIterator, cub::TexRefInputIterator, UnbindTexture, UnbindTexture, false, NO_FLAG, P4, "Comment")
ENTRY(cub::CountingInputIterator, cub::CountingInputIterator, true, NO_FLAG, P4, "Successful")
ENTRY(cub::TransformInputIterator, cub::TransformInputIterator, true, NO_FLAG, P4, "Successful")
ENTRY(cub::ConstantInputIterator, cub::ConstantInputIterator, true, NO_FLAG, P4, "Successful")
ENTRY(cub::DiscardOutputIterator, cub::DiscardOutputIterator, true, NO_FLAG, P4, "Successful")
ENTRY(cub::CacheModifiedInputIterator, cub::CacheModifiedInputIterator, false, NO_FLAG, P4, "Comment")
ENTRY(cub::CacheModifiedOutputIterator, cub::CacheModifiedOutputIterator, false, NO_FLAG, P4, "Comment")

// clang-format on
