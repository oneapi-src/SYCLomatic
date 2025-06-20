// UNSUPPORTED: system-windows
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::exclusive_scan | FileCheck %s -check-prefix=CG_EXCLUSIVE_SCAN
// CG_EXCLUSIVE_SCAN: CUDA API:
// CG_EXCLUSIVE_SCAN-NEXT:   cooperative_groups::exclusive_scan(
// CG_EXCLUSIVE_SCAN-NEXT:       tile32 /* type group */, sdata[tid] /* type value */,
// CG_EXCLUSIVE_SCAN-NEXT:       cooperative_groups::plus<double>() /* type operator */);
// CG_EXCLUSIVE_SCAN-NEXT:   cooperative_groups::exclusive_scan(tile32 /* type group */,
// CG_EXCLUSIVE_SCAN-NEXT                                     sdata[tid] /* type value */);
// CG_EXCLUSIVE_SCAN: Is migrated to:
// CG_EXCLUSIVE_SCAN-NEXT:   sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<double>());
// CG_EXCLUSIVE_SCAN-NEXT:   sycl::exclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<>());


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block::num_threads | FileCheck %s -check-prefix=CG_TB_NUM_THREADS
// CG_TB_NUM_THREADS: CUDA API:
// CG_TB_NUM_THREADS-NEXT:   cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
// CG_TB_NUM_THREADS:  tb.num_threads();
// CG_TB_NUM_THREADS-NEXT: Is migrated to:
// CG_TB_NUM_THREADS-NEXT:   sycl::group<3> tb = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TB_NUM_THREADS:   sycl::ext::oneapi::this_work_item::get_work_group<3>().get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block::group_index | FileCheck %s -check-prefix=CG_TB_GROUP_INDEX
// CG_TB_GROUP_INDEX: CUDA API:
// CG_TB_GROUP_INDEX-NEXT:   cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
// CG_TB_GROUP_INDEX-NEXT:   tb.thread_index();
// CG_TB_GROUP_INDEX-NEXT: Is migrated to:
// CG_TB_GROUP_INDEX-NEXT:   sycl::group<3> tb = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TB_GROUP_INDEX-NEXT:   dpct::dim3(tb.get_local_id());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::this_thread | FileCheck %s -check-prefix=CG_THIS_THREAD
// CG_THIS_THREAD: CUDA API:
// CG_THIS_THREAD-NEXT:   auto thread = cooperative_groups::this_thread();
// CG_THIS_THREAD-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_THIS_THREAD-NEXT:   auto thread = dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 1);


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::this_grid | FileCheck %s -check-prefix=CG_THIS_GRID
// CG_THIS_GRID: CUDA API:
// CG_THIS_GRID: cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_THIS_GRID: Is migrated to (with the option --use-experimental-features=root-group):
// CG_THIS_GRID: sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::sync | FileCheck %s -check-prefix=CG_SYNC
// CG_SYNC: CUDA API:
// CG_SYNC-NEXT:  cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
// CG_SYNC-NEXT:   cooperative_groups::thread_block_tile<32> tbt32 =
// CG_SYNC-NEXT:       cooperative_groups::tiled_partition<32>(tb);
// CG_SYNC-NEXT:   tb.sync();
// CG_SYNC-NEXT:   tbt32.sync();
// CG_SYNC-NEXT:   cooperative_groups::sync(tb);
// CG_SYNC-NEXT:   cooperative_groups::sync(tbt32);
// CG_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_SYNC-NEXT:   auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CG_SYNC-NEXT:   sycl::group<3> tb = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_SYNC-NEXT:   sycl::sub_group tbt32 =
// CG_SYNC-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_SYNC-NEXT:   item_ct1.barrier();
// CG_SYNC-NEXT:   sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
// CG_SYNC-NEXT:   item_ct1.barrier();
// CG_SYNC-NEXT:   sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::plus | FileCheck %s -check-prefix=CG_PLUS
// CG_PLUS: CUDA API:
// CG_PLUS-NEXT:   cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::plus<double>()/*cg::plus<T>*/);
// CG_PLUS-NEXT: Is migrated to:
// CG_PLUS-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<double>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::less | FileCheck %s -check-prefix=CG_LESS
// CG_LESS: CUDA API:
// CG_LESS-NEXT:   cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::less<double>()/*cg::less<T>*/);
// CG_LESS-NEXT: Is migrated to:
// CG_LESS-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::minimum<double>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::inclusive_scan | FileCheck %s -check-prefix=CG_INCLUSIVE_SCAN
// CG_INCLUSIVE_SCAN: CUDA API:
// CG_INCLUSIVE_SCAN-NEXT:  cooperative_groups::inclusive_scan(
// CG_INCLUSIVE_SCAN-NEXT:      tile32 /* type group */, sdata[tid] /* type value */,
// CG_INCLUSIVE_SCAN-NEXT:      cooperative_groups::plus<double>() /* type operator */);
// CG_INCLUSIVE_SCAN:  cooperative_groups::inclusive_scan(tile32 /* type group */,
// CG_INCLUSIVE_SCAN-NEXT:                                      sdata[tid] /* type value */);
// CG_INCLUSIVE_SCAN: Is migrated to:
// CG_INCLUSIVE_SCAN-NEXT:  sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<double>());
// CG_INCLUSIVE_SCAN: sycl::inclusive_scan_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::thread_rank | FileCheck %s -check-prefix=CG_GG_THREAD_RANK
// CG_GG_THREAD_RANK: CUDA API:
// CG_GG_THREAD_RANK-NEXT:  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_THREAD_RANK-NEXT:  grid.thread_rank() /* grid_group::thread_rank */;
// CG_GG_THREAD_RANK-NEXT: Is migrated to:
// CG_GG_THREAD_RANK-NEXT:  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_THREAD_RANK-NEXT:  grid.thread_rank() /* grid_group::thread_rank */;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::sync | FileCheck %s -check-prefix=CG_GG_SYNC
// CG_GG_SYNC: CUDA API:
// CG_GG_SYNC-NEXT:  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_SYNC-NEXT:   grid.sync();
// CG_GG_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=root-group):
// CG_GG_SYNC-NEXT:   sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();
// CG_GG_SYNC-NEXT:   sycl::group_barrier(grid);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::size | FileCheck %s -check-prefix=CG_GG_SIZE
// CG_GG_SIZE: CUDA API:
// CG_GG_SIZE-NEXT:  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_SIZE-NEXT:   grid.size();
// CG_GG_SIZE-NEXT: Is migrated to (with the option --use-experimental-features=root-group):
// CG_GG_SIZE-NEXT:   sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();
// CG_GG_SIZE-NEXT:   grid.get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::num_threads | FileCheck %s -check-prefix=CG_GG_NUM_THREADS
// CG_GG_NUM_THREADS: CUDA API:
// CG_GG_NUM_THREADS-NEXT:  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_NUM_THREADS-NEXT:  grid.num_threads();
// CG_GG_NUM_THREADS-NEXT:Is migrated to (with the option --use-experimental-features=root-group):
// CG_GG_NUM_THREADS-NEXT:  sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();
// CG_GG_NUM_THREADS-NEXT:  grid.get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::num_blocks | FileCheck %s -check-prefix=CG_GG_NUM_BLOCKS
// CG_GG_NUM_BLOCKS: CUDA API:
// CG_GG_NUM_BLOCKS-NEXT:   cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_NUM_BLOCKS-NEXT:   grid.num_blocks();
// CG_GG_NUM_BLOCKS-NEXT: Is migrated to (with the option --use-experimental-features=root-group):
// CG_GG_NUM_BLOCKS-NEXT:   sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();
// CG_GG_NUM_BLOCKS-NEXT:   grid.get_group_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::grid_group::block_rank | FileCheck %s -check-prefix=CG_GG_BLOCK_RANK
// CG_GG_BLOCK_RANK: CUDA API:
// CG_GG_BLOCK_RANK-NEXT:   cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// CG_GG_BLOCK_RANK-NEXT:   grid.block_rank();
// CG_GG_BLOCK_RANK-NEXT: Is migrated to (with the option --use-experimental-features=root-group):
// CG_GG_BLOCK_RANK-NEXT:   sycl::ext::oneapi::experimental::root_group grid = sycl::ext::oneapi::this_work_item::get_nd_item<3>().ext_oneapi_get_root_group();
// CG_GG_BLOCK_RANK-NEXT:   grid.get_group_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::greater | FileCheck %s -check-prefix=CG_GREATER
// CG_GREATER: CUDA API:
// CG_GREATER-NEXT:   cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::greater<double>()/*cg::greater<T>*/);
// CG_GREATER-NEXT: Is migrated to:
// CG_GREATER-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::maximum<double>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::coalesced_threads | FileCheck %s -check-prefix=CG_COALESCED_THREADS
// CG_COALESCED_THREADS: CUDA API:
// CG_COALESCED_THREADS-NEXT:   cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
// CG_COALESCED_THREADS-NEXT: Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CG_COALESCED_THREADS-NEXT:   sycl::ext::oneapi::experimental::opportunistic_group active = sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::coalesced_group::thread_rank | FileCheck %s -check-prefix=CG_CG_THREAD_RANK
// CG_CG_THREAD_RANK:CUDA API:
// CG_CG_THREAD_RANK-NEXT:  cooperative_groups::coalesced_group active =
// CG_CG_THREAD_RANK-NEXT:      cooperative_groups::coalesced_threads();
// CG_CG_THREAD_RANK-NEXT:  active.thread_rank();
// CG_CG_THREAD_RANK-NEXT: Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CG_CG_THREAD_RANK-NEXT:  sycl::ext::oneapi::experimental::opportunistic_group active =
// CG_CG_THREAD_RANK-NEXT:      sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();
// CG_CG_THREAD_RANK-NEXT:  active.get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::coalesced_group::sync | FileCheck %s -check-prefix=CG_CG_SYNC
// CG_CG_SYNC: CUDA API:
// CG_CG_SYNC-NEXT:   cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
// CG_CG_SYNC-NEXT:   active.sync();
// CG_CG_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CG_CG_SYNC-NEXT:   sycl::ext::oneapi::experimental::opportunistic_group active = sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();
// CG_CG_SYNC-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::coalesced_group::size | FileCheck %s -check-prefix=CG_CG_SIZE
// CG_CG_SIZE: CUDA API:
// CG_CG_SIZE-NEXT:  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
// CG_CG_SIZE-NEXT:   active.size();
// CG_CG_SIZE-NEXT: Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CG_CG_SIZE-NEXT:   sycl::ext::oneapi::experimental::opportunistic_group active = sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();
// CG_CG_SIZE-NEXT:   active.get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::coalesced_group::shfl | FileCheck %s -check-prefix=CG_CG_SHFL
// CG_CG_SHFL: CUDA API:
// CG_CG_SHFL-NEXT:   cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
// CG_CG_SHFL-NEXT:  active.shfl(0, 0);
// CG_CG_SHFL-NEXT:Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CG_CG_SHFL-NEXT:  sycl::ext::oneapi::experimental::opportunistic_group active = sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();
// CG_CG_SHFL-NEXT:  sycl::select_from_group(active, 0, 0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::bit_xor | FileCheck %s -check-prefix=CG_BIT_XOR
// CG_BIT_XOR: CUDA API:
// CG_BIT_XOR-NEXT:  cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::bit_xor<int>()/*cg::bit_xor<T>*/);
// CG_BIT_XOR-NEXT: Is migrated to:
// CG_BIT_XOR-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::bit_xor<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::bit_or  | FileCheck %s -check-prefix=CG_BIT_OR
// CG_BIT_OR: CUDA API:
// CG_BIT_OR-NEXT:   cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::bit_or<int>()/*cg::bit_or<T>*/);
// CG_BIT_OR-NEXT: Is migrated to:
// CG_BIT_OR-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::bit_or<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::bit_and | FileCheck %s -check-prefix=CG_BIT_AND
// CG_BIT_AND: CUDA API:
// CG_BIT_AND-NEXT:   cooperative_groups::reduce(tile32 /*thread_block_tile<32>*/, sdata[tid]/*data*/, cooperative_groups::bit_and<int>()/*cg::bit_and<T>*/);
// CG_BIT_AND-NEXT: Is migrated to:
// CG_BIT_AND-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::bit_and<int>());
