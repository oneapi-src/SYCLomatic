//===---APINamesMapThrust.inc-------------------------------------------===//
//
// Copyright (C) 2020 - 2020 Intel Corporation. All rights reserved.
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
ENTRY_HOST("copy", "std::copy", "dpstd::execution::sycl")
ENTRY_HOST("copy_n", "std::copy_n", "dpstd::execution::sycl")
ENTRY_HOST("sequence", "dpct::sequence", "dpstd::execution::sycl")
ENTRY_HOST("stable_sort_by_key", "dpct::stable_sort_by_key", "dpstd::execution::sycl")
ENTRY_HOST("transform", "std::transform", "dpstd::execution::sycl")
ENTRY_HOST("reduce", "std::reduce", "dpstd::execution::sycl")
ENTRY_HOST("device_free", "dpct::device_free", "")
ENTRY_HOST("device_malloc", "dpct::device_malloc", "")
ENTRY_BOTH("raw_pointer_cast", "dpct::raw_pointer_cast", "")
ENTRY_BOTH("complex", "std::complex", "")
ENTRY_BOTH("exp", "std::exp", "")
ENTRY_BOTH("log", "std::log", "")
ENTRY_BOTH("copy_if", "std::copy_if", "dpstd::execution::sycl")
ENTRY_BOTH("make_counting_iterator", "dpct::make_counting_iterator", "")
ENTRY_BOTH("count_if", "std::count_if", "dpstd::execution::sycl")
ENTRY_BOTH("fill", "std::fill", "dpstd::execution::sycl")
ENTRY_BOTH("fill_n", "std::fill_n", "dpstd::execution::sycl")
ENTRY_BOTH("device_pointer_cast", "dpct::device_pointer_cast", "")
ENTRY_BOTH("for_each", "std::for_each", "dpstd::execution::sycl")
ENTRY_BOTH("max", "std::max", "")
ENTRY_BOTH("min", "std::min", "")
ENTRY_BOTH("inner_product", "dpct::inner_product", "dpstd::execution::sycl")
ENTRY_BOTH("sort_by_key", "dpct::sort_by_key", "dpstd::execution::sycl")
ENTRY_BOTH("get", "std::get", "")
ENTRY_BOTH("make_constant_iterator", "dpct::make_constant_iterator", "")
ENTRY_BOTH("make_tuple", "std::make_tuple", "")
ENTRY_BOTH("make_zip_iterator", "dpstd::make_zip_iterator", "")
ENTRY_BOTH("transform_reduce", "std::transform_reduce", "dpstd::execution::sycl")
