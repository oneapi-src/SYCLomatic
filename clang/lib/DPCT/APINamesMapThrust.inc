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
* ENTRY_DEVICE(thrustAPIName, targetAPIName)
*   Specifies the mapping for 'thrustAPIName'.
*   Expected in device code
*
* ENTRY_BOTH(thrustAPIName, targetAPIName)
*   Specifies the mapping for 'thrustAPIName'.
*   Expected in both host and device code
*
****************************************************************************/
ENTRY_HOST("copy", "std::copy", "dpstd::execution::sycl")
ENTRY_HOST("copy", "std::copy", "dpstd::execution::sycl")
ENTRY_HOST("copy_n", "std::copy_n", "dpstd::execution::sycl")
ENTRY_HOST("sequence", "dpct::sequence", "dpstd::execution::sycl")
ENTRY_HOST("stable_sort_by_key", "dpct::stable_sort_by_key", "dpstd::execution::sycl")
ENTRY_HOST("reduce", "std::reduce", "dpstd::execution::sycl")
ENTRY_HOST("device_free", "dpct::device_free", "")
ENTRY_HOST("device_malloc", "dpct::device_malloc", "")
ENTRY_BOTH("raw_pointer_cast", "dpct::raw_pointer_cast")
ENTRY_BOTH("complex", "std::complex")
ENTRY_BOTH("exp", MapNames::getClNamespace() + "::exp")
ENTRY_BOTH("log", MapNames::getClNamespace() + "::log")
