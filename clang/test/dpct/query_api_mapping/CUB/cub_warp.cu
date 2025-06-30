// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// UNSUPPORTED: system-windows
// clang-format off
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpScan::InclusiveSum | FileCheck %s -check-prefix=CHECK_WARPSCAN_INCLUSIVESUM
// CHECK_WARPSCAN_INCLUSIVESUM:  CUDA API:
// CHECK_WARPSCAN_INCLUSIVESUM:    __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
// CHECK_WARPSCAN_INCLUSIVESUM:    cub::WarpScan<int>(temp_storage).InclusiveSum(thread_data/*int*/, thread_data/*int &*/);
// CHECK_WARPSCAN_INCLUSIVESUM:  Is migrated to:
// CHECK_WARPSCAN_INCLUSIVESUM:    thread_data = sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpScan::InclusiveScan | FileCheck %s -check-prefix=CHECK_WARPSCAN_INCLUSIVESCAN
// CHECK_WARPSCAN_INCLUSIVESCAN:  CUDA API:
// CHECK_WARPSCAN_INCLUSIVESCAN:    __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
// CHECK_WARPSCAN_INCLUSIVESCAN:    cub::WarpScan<int>(temp_storage).InclusiveScan(thread_data/*int*/, thread_data/*int &*/, cub::Sum()/*ScanOp*/);
// CHECK_WARPSCAN_INCLUSIVESCAN:  Is migrated to:
// CHECK_WARPSCAN_INCLUSIVESCAN:    thread_data = sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpScan::ExclusiveSum | FileCheck %s -check-prefix=CHECK_WARPSCAN_EXCLUSIVESUM
// CHECK_WARPSCAN_EXCLUSIVESUM:  CUDA API:
// CHECK_WARPSCAN_EXCLUSIVESUM:    __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
// CHECK_WARPSCAN_EXCLUSIVESUM:    cub::WarpScan<int>(temp_storage).ExclusiveSum(thread_data/*int*/, thread_data/*int &*/);
// CHECK_WARPSCAN_EXCLUSIVESUM:  Is migrated to:
// CHECK_WARPSCAN_EXCLUSIVESUM:    thread_data = sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpScan::ExclusiveScan | FileCheck %s -check-prefix=CHECK_WARPSCAN_EXCLUSIVESCAN
// CHECK_WARPSCAN_EXCLUSIVESCAN:  CUDA API:
// CHECK_WARPSCAN_EXCLUSIVESCAN:    __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
// CHECK_WARPSCAN_EXCLUSIVESCAN:    cub::WarpScan<int>(temp_storage).ExclusiveScan(thread_data/*int*/, thread_data/*int &*/, cub::Sum()/*ScanOp*/);
// CHECK_WARPSCAN_EXCLUSIVESCAN:  Is migrated to:
// CHECK_WARPSCAN_EXCLUSIVESCAN:    thread_data = sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpScan::Broadcast | FileCheck %s -check-prefix=CHECK_WARPSCAN_BROADCAST
// CHECK_WARPSCAN_BROADCAST:  CUDA API:
// CHECK_WARPSCAN_BROADCAST:    __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
// CHECK_WARPSCAN_BROADCAST:    cub::WarpScan<int>(temp_storage).Broadcast(thread_data/*int*/, 0/*unsigned int*/);
// CHECK_WARPSCAN_BROADCAST:  Is migrated to:
// CHECK_WARPSCAN_BROADCAST:    sycl::group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, 0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpReduce::Sum | FileCheck %s -check-prefix=CHECK_WARPREDUCE_SUM
// CHECK_WARPREDUCE_SUM:  CUDA API:
// CHECK_WARPREDUCE_SUM:    __shared__ typename cub::WarpReduce<int>::TempStorage temp_storage;
// CHECK_WARPREDUCE_SUM:    int result1 = cub::WarpReduce<int>(temp_storage).Sum(thread_data/*int*/);
// CHECK_WARPREDUCE_SUM:    int result2 = cub::WarpReduce<int>(temp_storage).Sum(thread_data/*int*/, valid_items/*int*/);
// CHECK_WARPREDUCE_SUM:  Is migrated to:
// CHECK_WARPREDUCE_SUM:    int result1 = sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::plus<>());
// CHECK_WARPREDUCE_SUM:    int result2 = dpct::group::reduce_over_partial_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, valid_items, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpReduce::Reduce | FileCheck %s -check-prefix=CHECK_WARPREDUCE_REDUCE
// CHECK_WARPREDUCE_REDUCE:  CUDA API:
// CHECK_WARPREDUCE_REDUCE:    __shared__ typename cub::WarpReduce<int>::TempStorage temp_storage;
// CHECK_WARPREDUCE_REDUCE:    int result1 = cub::WarpReduce<int>(temp_storage).Reduce(thread_data/*int*/, cub::Min()/*ReductionOp*/);
// CHECK_WARPREDUCE_REDUCE:    int result2 = cub::WarpReduce<int>(temp_storage).Reduce(thread_data/*int*/, cub::Min()/*ReductionOp*/, valid_items/*int*/);
// CHECK_WARPREDUCE_REDUCE:  Is migrated to:
// CHECK_WARPREDUCE_REDUCE:    int result1 = sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), thread_data, sycl::minimum<>());
// CHECK_WARPREDUCE_REDUCE:    int result2 = dpct::group::reduce_over_partial_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, valid_items, sycl::minimum<>());
