//===--------------- APINamesMapThrust.inc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
ENTRY_HOST("thrust::copy", "std::copy", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::copy_n", "std::copy_n", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::sequence", MapNames::getDpctNamespace() + "iota", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::sort", "oneapi::dpl::sort", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::transform_if", MapNames::getDpctNamespace() + "transform_if", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::reduce", "std::reduce", "oneapi::dpl::execution::sycl")
ENTRY_HOST("thrust::device_free", MapNames::getDpctNamespace() + "free_device", "")
ENTRY_HOST("thrust::device_malloc", MapNames::getDpctNamespace() + "malloc_device", "")
ENTRY_BOTH("thrust::raw_pointer_cast", MapNames::getDpctNamespace() + "get_raw_pointer", "")
ENTRY_BOTH("thrust::complex", "std::complex", "")
ENTRY_BOTH("thrust::make_counting_iterator", MapNames::getDpctNamespace() + "make_counting_iterator", "")
ENTRY_BOTH("thrust::count_if", "std::count_if", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::fill", "std::fill", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::fill_n", "std::fill_n", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::device_pointer_cast", MapNames::getDpctNamespace() + "get_device_pointer", "")
ENTRY_BOTH("thrust::max", "std::max", "")
ENTRY_BOTH("thrust::min", "std::min", "")
ENTRY_BOTH("thrust::make_constant_iterator", MapNames::getDpctNamespace() + "make_constant_iterator", "")
ENTRY_BOTH("thrust::make_tuple", "std::make_tuple", "")
ENTRY_BOTH("thrust::tie", "std::tie", "")
ENTRY_BOTH("thrust::partition_point", MapNames::getDpctNamespace() + "partition_point", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::binary_search", "oneapi::dpl::binary_search", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::lower_bound", "oneapi::dpl::lower_bound", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::upper_bound", "oneapi::dpl::upper_bound", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::equal_range", "oneapi::dpl::equal_range", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::count", "std::count", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::generate", "std::generate", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::generate_n", "std::generate_n", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::merge", "std::merge", "oneapi::dpl::execution::sycl")
ENTRY_BOTH("thrust::uninitialized_fill", "std::uninitialized_fill", "")
ENTRY_BOTH("thrust::unique", "std::unique", "")
ENTRY_BOTH("thrust::max_element", "std::max_element", "")
ENTRY_BOTH("thrust::min_element", "std::min_element", "")
ENTRY_BOTH("thrust::make_discard_iterator", "oneapi::dpl::discard_iterator", "")
