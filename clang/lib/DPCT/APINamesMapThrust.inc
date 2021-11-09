//===---APINamesMapThrust.inc-------------------------------------------===//
//
// Copyright (C) 2020 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/****************************************************************************
* ENTRY_HOST(thrustAPIName, targetAPIName, ExececutionPolicy)
*   Specifies the mapping for 'thrustAPIName'.
*   Only expected in host code
*
* ENTRY_DEVICE(thrustAPIName, targetAPIName, ExececutionPolicy)
*   Specifies the mapping for 'thrustAPIName'.
*   Expected in device code
*
* ENTRY_BOTH(thrustAPIName, targetAPIName, ExececutionPolicy)
*   Specifies the mapping for 'thrustAPIName'.
*   Expected in both host and device code
*
****************************************************************************/
ENTRY_HOST("copy", "std::copy", "oneapi::dpl::execution::sycl")
ENTRY_HOST("copy_n", "std::copy_n", "oneapi::dpl::execution::sycl")
ENTRY_HOST("sequence", MapNames::getDpctNamespace() + "iota", "oneapi::dpl::execution::sycl")
ENTRY_HOST("stable_sort_by_key", MapNames::getDpctNamespace(true) + "stable_sort", "oneapi::dpl::execution::sycl")
ENTRY_HOST("sort", "oneapi::dpl::sort", "oneapi::dpl::execution::sycl")
ENTRY_HOST("transform", "std::transform", "oneapi::dpl::execution::sycl")
ENTRY_HOST("transform_if", MapNames::getDpctNamespace() + "transform_if", "oneapi::dpl::execution::sycl")
ENTRY_HOST("reduce", "std::reduce", "oneapi::dpl::execution::sycl")
ENTRY_HOST("device_free", MapNames::getDpctNamespace() + "free_device", "")
ENTRY_HOST("device_malloc", MapNames::getDpctNamespace() + "malloc_device", "")
ENTRY_BOTH("raw_pointer_cast", MapNames::getDpctNamespace() + "get_raw_pointer", "")
ENTRY_BOTH("complex", "std::complex", "")
ENTRY_BOTH("exp", "std::exp", "")
ENTRY_BOTH("log", "std::log", "")
ENTRY_BOTH("copy_if", "std::copy_if", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("make_counting_iterator", MapNames::getDpctNamespace() + "make_counting_iterator", "")
ENTRY_BOTH("count_if", "std::count_if", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("fill", "std::fill", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("fill_n", "std::fill_n", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("device_pointer_cast", MapNames::getDpctNamespace() + "get_device_pointer", "")
ENTRY_BOTH("for_each", "std::for_each", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("max", "std::max", "")
ENTRY_BOTH("min", "std::min", "")
ENTRY_BOTH("inner_product", MapNames::getDpctNamespace() + "inner_product", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("sort_by_key", MapNames::getDpctNamespace() + "sort", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("get", "std::get", "")
ENTRY_BOTH("make_constant_iterator", MapNames::getDpctNamespace() + "make_constant_iterator", "")
ENTRY_BOTH("make_tuple", "std::make_tuple", "")
ENTRY_BOTH("make_zip_iterator", "oneapi::dpl::make_zip_iterator", "")
ENTRY_BOTH("exclusive_scan_by_key", "oneapi::dpl::exclusive_scan_by_segment", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("inclusive_scan_by_key", "oneapi::dpl::inclusive_scan_by_segment", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("partition_point", MapNames::getDpctNamespace() + "partition_point", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("binary_search", "oneapi::dpl::binary_search", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("lower_bound", "oneapi::dpl::lower_bound", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("upper_bound", "oneapi::dpl::upper_bound", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("equal_range", "oneapi::dpl::equal_range", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("count", "std::count", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("generate", "std::generate", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("generate_n", "std::generate_n", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("merge", "std::merge", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("uninitialized_fill", "std::uninitialized_fill", "")
ENTRY_BOTH("unique", "std::unique", "")
ENTRY_BOTH("exclusive_scan", "std::exclusive_scan", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("reduce_by_key", "oneapi::dpl::reduce_by_segment", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("max_element", "std::max_element", "")
ENTRY_BOTH("min_element", "std::min_element", "")
ENTRY_BOTH("make_permutation_iterator", "oneapi::dpl::make_permutation_iterator", "")
ENTRY_BOTH("make_transform_iterator", "oneapi::dpl::make_transform_iterator", "")
ENTRY_BOTH("make_discard_iterator", "oneapi::dpl::discard_iterator", "")
