// UNSUPPORTED: system-windows
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.7, v12.8, v12.9
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.7, cuda-12.8, cuda-12.9

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::experimental::this_thread_block 
// CG_EXP_THIS_THREAD_BLOCK: CUDA API:
// CG_EXP_THIS_THREAD_BLOCK-NEXT:  cooperative_groups::thread_block thb = cooperative_groups::experimental::this_thread_block(shared/*cooperative_groups::experimental::block_tile_memory*/);
// CG_EXP_THIS_THREAD_BLOCK-NEXT: Is migrated to:
// CG_EXP_THIS_THREAD_BLOCK-NEXT:  sycl::group<3> thb = sycl::ext::oneapi::this_work_item::get_work_group<3>();


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cooperative_groups::experimental::tiled_partition
// CG_EXP_THIS_TILED_PARTITION: CUDA API:
// CG_EXP_THIS_TILED_PARTITION-NEXT:    cooperative_groups::thread_block_tile<32> tbt32 = cooperative_groups::experimental::tiled_partition<32>(tb/*cooperative_groups::thread_block*/);
// CG_EXP_THIS_TILED_PARTITION-NEXT: Is migrated to:
// CG_EXP_THIS_TILED_PARTITION-NEXT:    sycl::sub_group tbt32 = sycl::ext::oneapi::this_work_item::get_sub_group();
