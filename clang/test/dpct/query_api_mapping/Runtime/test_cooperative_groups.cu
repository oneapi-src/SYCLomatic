// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --cuda-include-path="%cuda-path/include" -stop-on-parse-err --query-api-mapping=cooperative_groups::tiled_partition | FileCheck %s -check-prefix=CG_TILED_PARTITION
// CG_TILED_PARTITION: CUDA API:
// CG_TILED_PARTITION-NEXT:    cooperative_groups::thread_block cta =
// CG_TILED_PARTITION-NEXT:        cooperative_groups::this_thread_block();
// CG_TILED_PARTITION-NEXT:    cooperative_groups::tiled_partition<32>(cta);
// CG_TILED_PARTITION-NEXT:    cooperative_groups::tiled_partition<16>(cta);
// CG_TILED_PARTITION-NEXT: Is migrated to (with the option --use-experimental-features=logical-group --use-experimental-features=free-function-queries):
// CG_TILED_PARTITION-NEXT:    sycl::group<3> cta =
// CG_TILED_PARTITION-NEXT:        sycl::ext::oneapi::experimental::this_group<3>();
// CG_TILED_PARTITION-NEXT:    sycl::ext::oneapi::experimental::this_sub_group();
// CG_TILED_PARTITION-NEXT:    dpct::experimental::logical_group(sycl::ext::oneapi::experimental::this_nd_item<3>(), sycl::ext::oneapi::experimental::this_group<3>(), 16);


// RUN: dpct --cuda-include-path="%cuda-path/include" -stop-on-parse-err --query-api-mapping=cooperative_groups::thread_rank | FileCheck %s -check-prefix=CG_THREAD_RANK
// CG_THREAD_RANK: CUDA API:
// CG_THREAD_RANK-NEXT:   cooperative_groups::thread_block cta =
// CG_THREAD_RANK-NEXT:       cooperative_groups::this_thread_block();
// CG_THREAD_RANK-NEXT:   cta.thread_rank();
// CG_THREAD_RANK-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_THREAD_RANK-NEXT:   sycl::group<3> cta =
// CG_THREAD_RANK-NEXT:       sycl::ext::oneapi::experimental::this_group<3>();
// CG_THREAD_RANK-NEXT:   sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" -stop-on-parse-err --query-api-mapping=cooperative_groups::this_thread_block | FileCheck %s -check-prefix=CG_THIS_THREAD_BLOCK
// CG_THIS_THREAD_BLOCK: CUDA API:
// CG_THIS_THREAD_BLOCK-NEXT:   cooperative_groups::thread_block cta =
// CG_THIS_THREAD_BLOCK-NEXT:       cooperative_groups::this_thread_block();
// CG_THIS_THREAD_BLOCK-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_THIS_THREAD_BLOCK-NEXT:   sycl::group<3> cta =
// CG_THIS_THREAD_BLOCK-NEXT:       sycl::ext::oneapi::experimental::this_group<3>();

// RUN: dpct --cuda-include-path="%cuda-path/include" -stop-on-parse-err --query-api-mapping=cooperative_groups::reduce | FileCheck %s -check-prefix=CG_REDUCE
// CG_REDUCE: CUDA API:
// CG_REDUCE-NEXT:    cooperative_groups::reduce(
// CG_REDUCE-NEXT:        tile32 /* type group */, sdata[tid] /* type argument */,
// CG_REDUCE-NEXT:        cooperative_groups::plus<double>() /* type operator */);
// CG_REDUCE-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_REDUCE-NEXT:    sycl::reduce_over_group(sycl::ext::oneapi::experimental::this_sub_group(), sdata[tid], sycl::plus<double>());

