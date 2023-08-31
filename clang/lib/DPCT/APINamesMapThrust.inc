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
ENTRY_HOST("thrust::device_malloc", MapNames::getDpctNamespace() + "malloc_device", "")
ENTRY_BOTH("thrust::complex", "std::complex", "")
ENTRY_BOTH("thrust::device_pointer_cast", MapNames::getDpctNamespace() + "get_device_pointer", "")
ENTRY_BOTH("thrust::max", "std::max", "")
ENTRY_BOTH("thrust::min", "std::min", "")
ENTRY_BOTH("thrust::make_tuple", "std::make_tuple", "")
ENTRY_BOTH("thrust::tie", "std::tie", "")
