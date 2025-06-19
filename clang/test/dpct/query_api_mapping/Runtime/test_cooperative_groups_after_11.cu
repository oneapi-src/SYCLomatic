// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0

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
