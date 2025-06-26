// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::tiled_partition | FileCheck %s -check-prefix=CG_TILED_PARTITION
// CG_TILED_PARTITION: CUDA API:
// CG_TILED_PARTITION-NEXT:    cooperative_groups::thread_block cta =
// CG_TILED_PARTITION-NEXT:        cooperative_groups::this_thread_block();
// CG_TILED_PARTITION-NEXT:    cooperative_groups::tiled_partition<32>(cta);
// CG_TILED_PARTITION-NEXT:    cooperative_groups::tiled_partition<16>(cta);
// CG_TILED_PARTITION-NEXT: Is migrated to (with the option --use-experimental-features=logical-group --use-experimental-features=free-function-queries):
// CG_TILED_PARTITION-NEXT:    sycl::group<3> cta =
// CG_TILED_PARTITION-NEXT:        sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TILED_PARTITION-NEXT:    sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TILED_PARTITION-NEXT:    dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_rank | FileCheck %s -check-prefix=CG_THREAD_RANK
// CG_THREAD_RANK: CUDA API:
// CG_THREAD_RANK-NEXT:   cooperative_groups::thread_block cta =
// CG_THREAD_RANK-NEXT:       cooperative_groups::this_thread_block();
// CG_THREAD_RANK-NEXT:   cta.thread_rank();
// CG_THREAD_RANK-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_THREAD_RANK-NEXT:   sycl::group<3> cta =
// CG_THREAD_RANK-NEXT:       sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_THREAD_RANK-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::this_thread_block | FileCheck %s -check-prefix=CG_THIS_THREAD_BLOCK
// CG_THIS_THREAD_BLOCK: CUDA API:
// CG_THIS_THREAD_BLOCK-NEXT:   cooperative_groups::thread_block cta =
// CG_THIS_THREAD_BLOCK-NEXT:       cooperative_groups::this_thread_block();
// CG_THIS_THREAD_BLOCK-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_THIS_THREAD_BLOCK-NEXT:   sycl::group<3> cta =
// CG_THIS_THREAD_BLOCK-NEXT:       sycl::ext::oneapi::this_work_item::get_work_group<3>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::reduce | FileCheck %s -check-prefix=CG_REDUCE
// CG_REDUCE: CUDA API:
// CG_REDUCE-NEXT:    cooperative_groups::reduce(
// CG_REDUCE-NEXT:        tile32 /* type group */, sdata[tid] /* type argument */,
// CG_REDUCE-NEXT:        cooperative_groups::plus<double>() /* type operator */);
// CG_REDUCE-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// CG_REDUCE-NEXT:    sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), sdata[tid], sycl::plus<double>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_group::thread_rank | FileCheck %s -check-prefix=CG_TG_THREAD_RANK
// CG_TG_THREAD_RANK: CUDA API:
// CG_TG_THREAD_RANK-NEXT:   cooperative_groups::thread_group tg = cooperative_groups::this_thread_block();
// CG_TG_THREAD_RANK-NEXT:   tg.thread_rank();
// CG_TG_THREAD_RANK-NEXT: Is migrated to:
// CG_TG_THREAD_RANK-NEXT:   cooperative_groups::thread_group tg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TG_THREAD_RANK-NEXT:   tg.get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_group::sync | FileCheck %s -check-prefix=CG_TG_SYNC
// CG_TG_SYNC: CUDA API:
// CG_TG_SYNC-NEXT:    cooperative_groups::thread_group tg = cooperative_groups::this_thread_block();
// CG_TG_SYNC-NEXT:    tg.sync();
// CG_TG_SYNC-NEXT: Is migrated to:
// CG_TG_SYNC-NEXT:     cooperative_groups::thread_group tg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TG_SYNC-NEXT:     tg.barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_group::size | FileCheck %s -check-prefix=CG_TG_SIZE
// CG_TG_SIZE: CUDA API:
// CG_TG_SIZE-NEXT:   cooperative_groups::thread_group tg = cooperative_groups::this_thread_block();
// CG_TG_SIZE-NEXT:   tg.size();
// CG_TG_SIZE-NEXT: Is migrated to:
// CG_TG_SIZE-NEXT:   cooperative_groups::thread_group tg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TG_SIZE-NEXT:   tg.get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_group::num_threads | FileCheck %s -check-prefix=CG_TG_NUM_THREADS
// CG_TG_NUM_THREADS: CUDA API:
// CG_TG_NUM_THREADS-NEXT:   cooperative_groups::thread_group tg = cooperative_groups::this_thread_block();
// CG_TG_NUM_THREADS-NEXT:   tg.num_threads();
// CG_TG_NUM_THREADS-NEXT: Is migrated to:
// CG_TG_NUM_THREADS-NEXT:   cooperative_groups::thread_group tg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TG_NUM_THREADS-NEXT:   tg.get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_group::get_type | FileCheck %s -check-prefix=CG_TG_GET_TYPE
// CG_TG_GET_TYPE: CUDA API:
// CG_TG_GET_TYPE-NEXT:   cooperative_groups::thread_group tg = cooperative_groups::this_thread_block();
// CG_TG_GET_TYPE-NEXT:   tg.get_type();
// CG_TG_GET_TYPE-NEXT: Is migrated to:
// CG_TG_GET_TYPE-NEXT:   cooperative_groups::thread_group tg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TG_GET_TYPE-NEXT:   tg.get_type();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::thread_rank | FileCheck %s -check-prefix=CG_TBT_THREAD_RANK
// CG_TBT_THREAD_RANK: CUDA API:
// CG_TBT_THREAD_RANK-NEXT:  cooperative_groups::thread_block block =
// CG_TBT_THREAD_RANK-NEXT:      cooperative_groups::this_thread_block();
// CG_TBT_THREAD_RANK-NEXT:  cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_THREAD_RANK-NEXT:      cooperative_groups::tiled_partition<32>(block);
// CG_TBT_THREAD_RANK-NEXT:  cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_THREAD_RANK-NEXT:      cooperative_groups::tiled_partition<16>(block);
// CG_TBT_THREAD_RANK-NEXT:  ctile32.thread_rank();
// CG_TBT_THREAD_RANK-NEXT:  ctile16.thread_rank();
// CG_TBT_THREAD_RANK-NEXT: Is migrated to:
// CG_TBT_THREAD_RANK-NEXT:   sycl::group<3> block =
// CG_TBT_THREAD_RANK-NEXT:       sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TBT_THREAD_RANK-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_THREAD_RANK-NEXT:      sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_THREAD_RANK-NEXT:  cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_THREAD_RANK-NEXT:      cooperative_groups::tiled_partition<16>(block);
// CG_TBT_THREAD_RANK-NEXT:  sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_id();
// CG_TBT_THREAD_RANK-NEXT:  ctile16.thread_rank();

// RUN: dpct --cuda-include-path="%cuda-path/include"  --query-api-mapping=cooperative_groups::thread_block_tile::sync  | FileCheck %s -check-prefix=CG_TBT_SYNC
// CG_TBT_SYNC:  CUDA API:
// CG_TBT_SYNC-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 = cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SYNC-NEXT:  ctile32.sync();
// CG_TBT_SYNC-NEXT:Is migrated to:
// CG_TBT_SYNC-NEXT:  sycl::sub_group ctile32 = sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SYNC-NEXT:  sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::size | FileCheck %s -check-prefix=CG_TBT_SIZE
// CG_TBT_SIZE: CUDA API:
// CG_TBT_SIZE-NEXT:  cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_SIZE-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SIZE-NEXT:   cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_SIZE-NEXT:       cooperative_groups::tiled_partition<16>(block);
// CG_TBT_SIZE:   ctile32.size();
// CG_TBT_SIZE-NEXT:   ctile16.size();
// CG_TBT_SIZE-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_TBT_SIZE-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_SIZE-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SIZE-NEXT:   dpct::experimental::logical_group ctile16 =
// CG_TBT_SIZE-NEXT:       dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
// CG_TBT_SIZE:   sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_range();
// CG_TBT_SIZE-NEXT:   ctile16.get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::shfl_xor | FileCheck %s -check-prefix=CG_TBT_SHFL_XOR
// CG_TBT_SHFL_XOR: CUDA API:
// CG_TBT_SHFL_XOR-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_SHFL_XOR-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SHFL_XOR-NEXT:   cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_SHFL_XOR-NEXT:       cooperative_groups::tiled_partition<16>(block);
// CG_TBT_SHFL_XOR:   ctile32.shfl_xor(1, 0);
// CG_TBT_SHFL_XOR-NEXT:   ctile16.shfl_xor(1, 0);
// CG_TBT_SHFL_XOR-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_TBT_SHFL_XOR-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_SHFL_XOR-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SHFL_XOR-NEXT:   dpct::experimental::logical_group ctile16 =
// CG_TBT_SHFL_XOR-NEXT:       dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
// CG_TBT_SHFL_XOR:  sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0);
// CG_TBT_SHFL_XOR-NEXT:   dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0, 16);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::shfl_up | FileCheck %s -check-prefix=CG_TBT_SHFL_UP
// CG_TBT_SHFL_UP: CUDA API:
// CG_TBT_SHFL_UP-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_SHFL_UP-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SHFL_UP-NEXT:   cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_SHFL_UP-NEXT:       cooperative_groups::tiled_partition<16>(block);
// CG_TBT_SHFL_UP-NEXT:   ctile32.shfl_down(1, 0);
// CG_TBT_SHFL_UP-NEXT:   ctile16.shfl_down(1, 0);
// CG_TBT_SHFL_UP-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_TBT_SHFL_UP-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_SHFL_UP-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SHFL_UP-NEXT:   dpct::experimental::logical_group ctile16 =
// CG_TBT_SHFL_UP-NEXT:       dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
// CG_TBT_SHFL_UP-NEXT:   sycl::shift_group_left(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0);
// CG_TBT_SHFL_UP-NEXT:   dpct::shift_sub_group_left(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0, 16);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::shfl_down | FileCheck %s -check-prefix=CG_TBT_SHFL_DOWN
// CG_TBT_SHFL_DOWN: CUDA API:
// CG_TBT_SHFL_DOWN-NEXT:  cooperative_groups::thread_block_tile<32> ctile32 = cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SHFL_DOWN-NEXT:   cooperative_groups::thread_block_tile<16> ctile16 = cooperative_groups::tiled_partition<16>(block);
// CG_TBT_SHFL_DOWN-NEXT:   ctile32.shfl(1, 0);
// CG_TBT_SHFL_DOWN-NEXT:   ctile16.shfl(1, 0);
// CG_TBT_SHFL_DOWN-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_TBT_SHFL_DOWN-NEXT:   sycl::sub_group ctile32 = sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SHFL_DOWN-NEXT:   dpct::experimental::logical_group ctile16 = dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
// CG_TBT_SHFL_DOWN-NEXT:   sycl::select_from_group(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0);
// CG_TBT_SHFL_DOWN-NEXT:   dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0, 16);


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::shfl | FileCheck %s -check-prefix=CG_TBT_SHFL
// CG_TBT_SHFL: CUDA API:
// CG_TBT_SHFL-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_SHFL-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_SHFL-NEXT:   cooperative_groups::thread_block_tile<16> ctile16 =
// CG_TBT_SHFL-NEXT:       cooperative_groups::tiled_partition<16>(block);
// CG_TBT_SHFL:   ctile32.shfl(1, 0);
// CG_TBT_SHFL-NEXT:   ctile16.shfl(1, 0);
// CG_TBT_SHFL-NEXT: Is migrated to (with the option --use-experimental-features=logical-group):
// CG_TBT_SHFL-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_SHFL-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_SHFL-NEXT:   dpct::experimental::logical_group ctile16 =
// CG_TBT_SHFL-NEXT:       dpct::experimental::logical_group(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
// CG_TBT_SHFL:   sycl::select_from_group(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0);
// CG_TBT_SHFL-NEXT:   dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), 1, 0, 16);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::num_threads | FileCheck %s -check-prefix=CG_TBT_NUM_THREADS
// CG_TBT_NUM_THREADS: CUDA API:
// CG_TBT_NUM_THREADS-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_NUM_THREADS-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_NUM_THREADS-NEXT:   ctile32.num_threads();
// CG_TBT_NUM_THREADS-NEXT: Is migrated to:
// CG_TBT_NUM_THREADS-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_NUM_THREADS-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_NUM_THREADS-NEXT:   sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::meta_group_size | FileCheck %s -check-prefix=CG_TBT_META_GROUP_SIZE
// CG_TBT_META_GROUP_SIZE: CUDA API:
// CG_TBT_META_GROUP_SIZE-NEXT:   cooperative_groups::thread_block_tile<32> ctile32 =
// CG_TBT_META_GROUP_SIZE-NEXT:       cooperative_groups::tiled_partition<32>(block);
// CG_TBT_META_GROUP_SIZE-NEXT:   ctile32.meta_group_size();
// CG_TBT_META_GROUP_SIZE-NEXT: Is migrated to:
// CG_TBT_META_GROUP_SIZE-NEXT:   sycl::sub_group ctile32 =
// CG_TBT_META_GROUP_SIZE-NEXT:       sycl::ext::oneapi::this_work_item::get_sub_group();
// CG_TBT_META_GROUP_SIZE-NEXT:   sycl::ext::oneapi::this_work_item::get_sub_group().get_group_linear_range();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block_tile::meta_group_rank | FileCheck %s -check-prefix=CG_TBT_META_GROUP_RANK
// CG_TBT_META_GROUP_RANK: CUDA API:
// CG_TBT_META_GROUP_RANK-NEXT:   ctile32.meta_group_rank();// thread_block_tile<tile size>::meta_group_rank
// CG_TBT_META_GROUP_RANK-NEXT: Is migrated to:
// CG_TBT_META_GROUP_RANK-NEXT:   sycl::ext::oneapi::this_work_item::get_sub_group().get_group_linear_id();// thread_block_tile<tile size>::meta_group_rank

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block::thread_index | FileCheck %s -check-prefix=CG_TB_THREAD_INDEX
// CG_TB_THREAD_INDEX: CUDA API:
// CG_TB_THREAD_INDEX-NEXT:    cooperative_groups::thread_block tb =
// CG_TB_THREAD_INDEX-NEXT:       cooperative_groups::this_thread_block();
// CG_TB_THREAD_INDEX-NEXT:     tb.sync();
// CG_TB_THREAD_INDEX-NEXT: Is migrated to:
// CG_TB_THREAD_INDEX-NEXT:     sycl::group<3> tb =
// CG_TB_THREAD_INDEX-NEXT:       sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TB_THREAD_INDEX-NEXT:     sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block::sync | FileCheck %s -check-prefix=CG_TB_SYNC
// CG_TB_SYNC: CUDA API:
// CG_TB_SYNC-NEXT:   cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
// CG_TB_SYNC-NEXT:   tb.sync();
// CG_TB_SYNC-NEXT: Is migrated to:
// CG_TB_SYNC-NEXT:   sycl::group<3> tb = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TB_SYNC-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::thread_block::size | FileCheck %s -check-prefix=CG_TB_SIZE
// CG_TB_SIZE: CUDA API:
// CG_TB_SIZE-NEXT:   cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
// CG_TB_SIZE-NEXT:   tb.size();
// CG_TB_SIZE-NEXT: Is migrated to:
// CG_TB_SIZE-NEXT:   sycl::group<3> tb = sycl::ext::oneapi::this_work_item::get_work_group<3>();
// CG_TB_SIZE-NEXT:   sycl::ext::oneapi::this_work_item::get_work_group<3>().get_local_linear_range();
