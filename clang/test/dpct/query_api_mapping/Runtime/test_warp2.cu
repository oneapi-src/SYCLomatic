// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_add_sync | FileCheck %s -check-prefix=__reduce_add_sync
// __reduce_add_sync: CUDA API:
// __reduce_add_sync-NEXT:   r = __reduce_add_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_add_sync-NEXT: Is migrated to:
// __reduce_add_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_and_sync | FileCheck %s -check-prefix=__reduce_and_sync
// __reduce_and_sync: CUDA API:
// __reduce_and_sync-NEXT:   r = __reduce_and_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_and_sync-NEXT: Is migrated to:
// __reduce_and_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::bit_and<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_max_sync | FileCheck %s -check-prefix=__reduce_max_sync
// __reduce_max_sync: CUDA API:
// __reduce_max_sync-NEXT:   r = __reduce_max_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_max_sync-NEXT: Is migrated to:
// __reduce_max_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::maximum());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_min_sync | FileCheck %s -check-prefix=__reduce_min_sync
// __reduce_min_sync: CUDA API:
// __reduce_min_sync-NEXT:   r = __reduce_min_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_min_sync-NEXT: Is migrated to:
// __reduce_min_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::minimum());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_or_sync | FileCheck %s -check-prefix=__reduce_or_sync
// __reduce_or_sync: CUDA API:
// __reduce_or_sync-NEXT:   r = __reduce_or_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_or_sync-NEXT: Is migrated to:
// __reduce_or_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::bit_or<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__reduce_xor_sync | FileCheck %s -check-prefix=__reduce_xor_sync
// __reduce_xor_sync: CUDA API:
// __reduce_xor_sync-NEXT:   r = __reduce_xor_sync(mask /*unsigned int*/, value /*unsigned int*/);
// __reduce_xor_sync-NEXT: Is migrated to:
// __reduce_xor_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), value, sycl::bit_xor<>());
