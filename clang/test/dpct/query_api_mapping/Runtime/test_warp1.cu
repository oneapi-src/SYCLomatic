// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__any | FileCheck %s -check-prefix=__any
// __any: CUDA API:
// __any-NEXT:   r = __any(pred /*int*/);
// __any-NEXT: Is migrated to:
// __any-NEXT:   r = sycl::any_of_group(item_ct1.get_sub_group(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__all | FileCheck %s -check-prefix=__all
// __all: CUDA API:
// __all-NEXT:   r = __all(pred /*int*/);
// __all-NEXT: Is migrated to:
// __all-NEXT:   r = sycl::all_of_group(item_ct1.get_sub_group(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ballot | FileCheck %s -check-prefix=__ballot
// __ballot: CUDA API:
// __ballot-NEXT:   r = __ballot(pred /*int*/);
// __ballot-NEXT: Is migrated to:
// __ballot-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), pred ? (0x1 << item_ct1.get_sub_group().get_local_linear_id()) : 0, sycl::ext::oneapi::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_down | FileCheck %s -check-prefix=__shfl_down
// __shfl_down: CUDA API:
// __shfl_down-NEXT:   r = __shfl_down(var /*unsigned int*/, delta /*unsigned int*/, width /*int*/);
// __shfl_down-NEXT: Is migrated to:
// __shfl_down-NEXT:   r = dpct::shift_sub_group_left(item_ct1.get_sub_group(), var, delta, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl | FileCheck %s -check-prefix=__shfl
// __shfl: CUDA API:
// __shfl-NEXT:   r = __shfl(var /*unsigned int*/, src_lane /*int*/, width /*int*/);
// __shfl-NEXT: Is migrated to:
// __shfl-NEXT:   r = dpct::select_from_sub_group(item_ct1.get_sub_group(), var, src_lane, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_up | FileCheck %s -check-prefix=__shfl_up
// __shfl_up: CUDA API:
// __shfl_up-NEXT:   r = __shfl_up(var /*unsigned int*/, delta /*unsigned int*/, width /*int*/);
// __shfl_up-NEXT: Is migrated to:
// __shfl_up-NEXT:   r = dpct::shift_sub_group_right(item_ct1.get_sub_group(), var, delta, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_xor | FileCheck %s -check-prefix=__shfl_xor
// __shfl_xor: CUDA API:
// __shfl_xor-NEXT:   r = __shfl_xor(var /*unsigned int*/, lane /*int*/, width /*int*/);
// __shfl_xor-NEXT: Is migrated to:
// __shfl_xor-NEXT:   r = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__activemask | FileCheck %s -check-prefix=__activemask
// __activemask: CUDA API:
// __activemask-NEXT:   r = __activemask();
// __activemask-NEXT: Is migrated to:
// __activemask-NEXT:   /*
// __activemask-NEXT:   DPCT1086:0: __activemask() is migrated to 0xffffffff. You may need to adjust the code.
// __activemask-NEXT:   */
// __activemask-NEXT:   r = 0xffffffff;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__all_sync | FileCheck %s -check-prefix=__all_sync
// __all_sync: CUDA API:
// __all_sync-NEXT:   r = __all_sync(mask /*unsigned int*/, pred /*int*/);
// __all_sync-NEXT: Is migrated to:
// __all_sync-NEXT:   r = sycl::all_of_group(item_ct1.get_sub_group(), (~mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) || pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__any_sync | FileCheck %s -check-prefix=__any_sync
// __any_sync: CUDA API:
// __any_sync-NEXT:   r = __any_sync(mask /*unsigned int*/, pred /*int*/);
// __any_sync-NEXT: Is migrated to:
// __any_sync-NEXT:   r = sycl::any_of_group(item_ct1.get_sub_group(), (mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) && pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ballot_sync | FileCheck %s -check-prefix=__ballot_sync
// __ballot_sync: CUDA API:
// __ballot_sync-NEXT:   r = __ballot_sync(mask /*unsigned int*/, pred /*int*/);
// __ballot_sync-NEXT: Is migrated to:
// __ballot_sync-NEXT:   r = sycl::reduce_over_group(item_ct1.get_sub_group(), (mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) && pred ? (0x1 << item_ct1.get_sub_group().get_local_linear_id()) : 0, sycl::ext::oneapi::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__match_all_sync | FileCheck %s -check-prefix=__match_all_sync
// __match_all_sync: CUDA API:
// __match_all_sync-NEXT:   r = __match_all_sync(mask /*unsigned int*/, value /*int*/, pred /*int*/);
// __match_all_sync-NEXT: Is migrated to:
// __match_all_sync-NEXT:   r = dpct::match_all_over_sub_group(item_ct1.get_sub_group(), mask, value, pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__match_any_sync | FileCheck %s -check-prefix=__match_any_sync
// __match_any_sync: CUDA API:
// __match_any_sync-NEXT:   r = __match_any_sync(mask /*unsigned int*/, value /*int*/);
// __match_any_sync-NEXT: Is migrated to:
// __match_any_sync-NEXT:   r = dpct::match_any_over_sub_group(item_ct1.get_sub_group(), mask, value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_down_sync | FileCheck %s -check-prefix=__shfl_down_sync
// __shfl_down_sync: CUDA API:
// __shfl_down_sync-NEXT:   r = __shfl_down_sync(mask /*unsigned int*/, var /*unsigned int*/,
// __shfl_down_sync-NEXT:                        delta /*unsigned int*/, width /*int*/);
// __shfl_down_sync-NEXT: Is migrated to:
// __shfl_down_sync-NEXT:   r = dpct::shift_sub_group_left(item_ct1.get_sub_group(), var, delta, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_sync | FileCheck %s -check-prefix=__shfl_sync
// __shfl_sync: CUDA API:
// __shfl_sync-NEXT:   r = __shfl_sync(mask /*unsigned int*/, var /*unsigned int*/, src_lane /*int*/,
// __shfl_sync-NEXT:                   width /*int*/);
// __shfl_sync-NEXT: Is migrated to:
// __shfl_sync-NEXT:   r = dpct::select_from_sub_group(item_ct1.get_sub_group(), var, src_lane, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_up_sync | FileCheck %s -check-prefix=__shfl_up_sync
// __shfl_up_sync: CUDA API:
// __shfl_up_sync-NEXT:   r = __shfl_up_sync(mask /*unsigned int*/, var /*unsigned int*/,
// __shfl_up_sync-NEXT:                      delta /*unsigned int*/, width /*int*/);
// __shfl_up_sync-NEXT: Is migrated to:
// __shfl_up_sync-NEXT:   r = dpct::shift_sub_group_right(item_ct1.get_sub_group(), var, delta, width);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shfl_xor_sync | FileCheck %s -check-prefix=__shfl_xor_sync
// __shfl_xor_sync: CUDA API:
// __shfl_xor_sync-NEXT:   r = __shfl_xor_sync(mask /*unsigned int*/, var /*unsigned int*/, lane /*int*/,
// __shfl_xor_sync-NEXT:                       width /*int*/);
// __shfl_xor_sync-NEXT: Is migrated to:
// __shfl_xor_sync-NEXT:   r = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane, width);
