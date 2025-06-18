// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// UNSUPPORTED: system-windows
// clang-format off

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::BlockedToStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).BlockedToStriped(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).blocked_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::BlockedToWarpStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).BlockedToWarpStriped(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).blocked_to_sub_group_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::ScatterToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).ScatterToBlocked(thread_data/*int(&)[4]*/, thread_rank/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).scatter_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_rank);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::ScatterToStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).ScatterToStriped(thread_data/*int(&)[4]*/, thread_rank/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).scatter_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_rank);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::StripedToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).StripedToBlocked(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).striped_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::WarpStripedToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).WarpStripedToBlocked(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).sub_group_striped_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockLoad::Load | FileCheck %s -check-prefix=CHECK_BLOCKLOAD_LOAD
// CHECK_BLOCKLOAD_LOAD:  CUDA API:
// CHECK_BLOCKLOAD_LOAD:    __shared__ typename cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>::TempStorage temp_storage;
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/, default_value/*int*/);
// CHECK_BLOCKLOAD_LOAD:  Is migrated to:
// CHECK_BLOCKLOAD_LOAD:    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data);
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data, end);
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data, end, default_value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockStore::Store | FileCheck %s -check-prefix=CHECK_BLOCKSTORE_STORE
// CHECK_BLOCKSTORE_STORE:  CUDA API:
// CHECK_BLOCKSTORE_STORE:    __shared__ typename cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>::TempStorage temp_storage;
// CHECK_BLOCKSTORE_STORE:    cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKSTORE_STORE:    cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
// CHECK_BLOCKSTORE_STORE:  Is migrated to:
// CHECK_BLOCKSTORE_STORE:    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK_BLOCKSTORE_STORE:    dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::blocked>(temp_storage).store(item_ct1, dst, thread_data);
// CHECK_BLOCKSTORE_STORE:    dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::blocked>(temp_storage).store(item_ct1, dst, thread_data, end);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockRadixSort::Sort | FileCheck %s -check-prefix=CHECK_BLOCKRADIXSORT_SORT
// CHECK_BLOCKRADIXSORT_SORT:  CUDA API:
// CHECK_BLOCKRADIXSORT_SORT:    __shared__ typename cub::BlockRadixSort<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKRADIXSORT_SORT:    cub::BlockRadixSort<int, 128, 4>(temp_storage).Sort(thread_data/*int(&)[4]*/);
// CHECK_BLOCKRADIXSORT_SORT:  Is migrated to:
// CHECK_BLOCKRADIXSORT_SORT:    dpct::group::group_radix_sort<int, 4>(temp_storage).sort(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockRadixSort::SortBlockedToStriped | FileCheck %s -check-prefix=CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED
// CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED:  CUDA API:
// CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED:    __shared__ typename cub::BlockRadixSort<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED:    cub::BlockRadixSort<int, 128, 4>(temp_storage).SortBlockedToStriped(thread_data/*int(&)[4]*/);
// CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED:  Is migrated to:
// CHECK_BLOCKRADIXSORT_SORTBLOCKEDTOSTRIPED:    dpct::group::group_radix_sort<int, 4>(temp_storage).sort_blocked_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockRadixSort::SortDescending | FileCheck %s -check-prefix=CHECK_BLOCKRADIXSORT_SORTDESCENDING
// CHECK_BLOCKRADIXSORT_SORTDESCENDING:  CUDA API:
// CHECK_BLOCKRADIXSORT_SORTDESCENDING:    __shared__ typename cub::BlockRadixSort<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKRADIXSORT_SORTDESCENDING:    cub::BlockRadixSort<int, 128, 4>(temp_storage).SortDescending(thread_data/*int(&)[4]*/);
// CHECK_BLOCKRADIXSORT_SORTDESCENDING:  Is migrated to:
// CHECK_BLOCKRADIXSORT_SORTDESCENDING:    dpct::group::group_radix_sort<int, 4>(temp_storage).sort_descending(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockRadixSort::SortDescendingBlockedToStriped | FileCheck %s -check-prefix=CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED
// CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED:  CUDA API:
// CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED:    __shared__ typename cub::BlockRadixSort<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED:    cub::BlockRadixSort<int, 128, 4>(temp_storage).SortDescendingBlockedToStriped(thread_data/*int(&)[4]*/);
// CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED:  Is migrated to:
// CHECK_BLOCKRADIXSORT_SORTDESCENDINGBLOCKEDTOSTRIPED:    dpct::group::group_radix_sort<int, 4>(temp_storage).sort_descending_blocked_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockReduce::Reduce | FileCheck %s -check-prefix=CHECK_BLOCKREDUCE_REDUCE
// CHECK_BLOCKREDUCE_REDUCE:  CUDA API:
// CHECK_BLOCKREDUCE_REDUCE:    __shared__ typename cub::BlockReduce<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKREDUCE_REDUCE:    cub::BlockReduce<int, 4>(temp_storage).Reduce(data/*int*/, cub::Sum()/*ReduceOp*/);
// CHECK_BLOCKREDUCE_REDUCE:  Is migrated to:
// CHECK_BLOCKREDUCE_REDUCE:    sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockReduce::Sum | FileCheck %s -check-prefix=CHECK_BLOCKREDUCE_SUM
// CHECK_BLOCKREDUCE_SUM:  CUDA API:
// CHECK_BLOCKREDUCE_SUM:    __shared__ typename cub::BlockReduce<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKREDUCE_SUM:    cub::BlockReduce<int, 4>(temp_storage).Sum(data/*int*/);
// CHECK_BLOCKREDUCE_SUM:  Is migrated to:
// CHECK_BLOCKREDUCE_SUM:    sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockScan::ExclusiveScan | FileCheck %s -check-prefix=CHECK_BLOCKSCAN_EXCLUSIVESCAN
// CHECK_BLOCKSCAN_EXCLUSIVESCAN:  CUDA API:
// CHECK_BLOCKSCAN_EXCLUSIVESCAN:    __shared__ typename cub::BlockScan<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKSCAN_EXCLUSIVESCAN:    cub::BlockScan<int, 4>(temp_storage).ExclusiveScan(input/*int*/, output/*int &*/, init/*int*/, cub::Sum()/*ScanOp*/);
// CHECK_BLOCKSCAN_EXCLUSIVESCAN:  Is migrated to:
// CHECK_BLOCKSCAN_EXCLUSIVESCAN:    output = sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), input, init, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockScan::ExclusiveSum | FileCheck %s -check-prefix=CHECK_BLOCKSCAN_EXCLUSIVESUM
// CHECK_BLOCKSCAN_EXCLUSIVESUM:  CUDA API:
// CHECK_BLOCKSCAN_EXCLUSIVESUM:    __shared__ typename cub::BlockScan<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKSCAN_EXCLUSIVESUM:    cub::BlockScan<int, 4>(temp_storage).ExclusiveSum(input/*int*/, output/*int &*/);
// CHECK_BLOCKSCAN_EXCLUSIVESUM:  Is migrated to:
// CHECK_BLOCKSCAN_EXCLUSIVESUM:    output = sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), input, 0, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockScan::InclusiveScan | FileCheck %s -check-prefix=CHECK_BLOCKSCAN_INCLUSIVESCAN
// CHECK_BLOCKSCAN_INCLUSIVESCAN:  CUDA API:
// CHECK_BLOCKSCAN_INCLUSIVESCAN:    __shared__ typename cub::BlockScan<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKSCAN_INCLUSIVESCAN:    cub::BlockScan<int, 4>(temp_storage).InclusiveScan(input/*int*/, output/*int &*/, cub::Sum()/*ScanOp*/);
// CHECK_BLOCKSCAN_INCLUSIVESCAN:  Is migrated to:
// CHECK_BLOCKSCAN_INCLUSIVESCAN:    output = sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), input, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockScan::InclusiveSum | FileCheck %s -check-prefix=CHECK_BLOCKSCAN_INCLUSIVESUM
// CHECK_BLOCKSCAN_INCLUSIVESUM:  CUDA API:
// CHECK_BLOCKSCAN_INCLUSIVESUM:    __shared__ typename cub::BlockScan<int, 4>::TempStorage temp_storage;
// CHECK_BLOCKSCAN_INCLUSIVESUM:    cub::BlockScan<int, 4>(temp_storage).InclusiveSum(input/*int*/, output/*int &*/);
// CHECK_BLOCKSCAN_INCLUSIVESUM:  Is migrated to:
// CHECK_BLOCKSCAN_INCLUSIVESUM:    output = sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), input, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockShuffle::Down | FileCheck %s -check-prefix=CHECK_BLOCKSHUFFLE_DOWN
// CHECK_BLOCKSHUFFLE_DOWN:  CUDA API:
// CHECK_BLOCKSHUFFLE_DOWN:    __shared__ typename cub::BlockShuffle<int, 128>::TempStorage temp_storage;
// CHECK_BLOCKSHUFFLE_DOWN:    cub::BlockShuffle<int, 128>(temp_storage).Down(input/*int*/, output/*int &*/);
// CHECK_BLOCKSHUFFLE_DOWN:  Is migrated to:
// CHECK_BLOCKSHUFFLE_DOWN:    dpct::group::group_shuffle<int, 128>(temp_storage).shuffle_left(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), input, output);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockShuffle::Offset | FileCheck %s -check-prefix=CHECK_BLOCKSHUFFLE_OFFSET
// CHECK_BLOCKSHUFFLE_OFFSET:  CUDA API:
// CHECK_BLOCKSHUFFLE_OFFSET:    __shared__ typename cub::BlockShuffle<int, 128>::TempStorage temp_storage;
// CHECK_BLOCKSHUFFLE_OFFSET:    cub::BlockShuffle<int, 128>(temp_storage).Offset(input/*int*/, output/*int &*/, distance/*int*/);
// CHECK_BLOCKSHUFFLE_OFFSET:  Is migrated to:
// CHECK_BLOCKSHUFFLE_OFFSET:    dpct::group::group_shuffle<int, 128>(temp_storage).select(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), input, output, distance);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockShuffle::Rotate | FileCheck %s -check-prefix=CHECK_BLOCKSHUFFLE_ROTATE
// CHECK_BLOCKSHUFFLE_ROTATE:  CUDA API:
// CHECK_BLOCKSHUFFLE_ROTATE:    __shared__ typename cub::BlockShuffle<int, 128>::TempStorage temp_storage;
// CHECK_BLOCKSHUFFLE_ROTATE:    cub::BlockShuffle<int, 128>(temp_storage).Rotate(input/*int*/, output/*int &*/, distance/*unsigned int*/);
// CHECK_BLOCKSHUFFLE_ROTATE:  Is migrated to:
// CHECK_BLOCKSHUFFLE_ROTATE:    dpct::group::group_shuffle<int, 128>(temp_storage).select2(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), input, output, distance);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockShuffle::Up | FileCheck %s -check-prefix=CHECK_BLOCKSHUFFLE_UP
// CHECK_BLOCKSHUFFLE_UP:  CUDA API:
// CHECK_BLOCKSHUFFLE_UP:    __shared__ typename cub::BlockShuffle<int, 128>::TempStorage temp_storage;
// CHECK_BLOCKSHUFFLE_UP:    cub::BlockShuffle<int, 128>(temp_storage).Up(input/*int*/, output/*int &*/);
// CHECK_BLOCKSHUFFLE_UP:  Is migrated to:
// CHECK_BLOCKSHUFFLE_UP:    dpct::group::group_shuffle<int, 128>(temp_storage).shuffle_right(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), input, output);
