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
ENTRY_HOST("thrust::device_new", MapNames::getDpctNamespace() + "device_new", "")
ENTRY_HOST("thrust::device_delete", MapNames::getDpctNamespace() + "device_delete", "")
ENTRY_HOST("thrust::malloc", MapNames::getDpctNamespace() + "malloc", "")
ENTRY_HOST("thrust::free", MapNames::getDpctNamespace() + "free", "")
ENTRY_BOTH("thrust::device_pointer_cast", MapNames::getDpctNamespace() + "get_device_pointer", "")
ENTRY_BOTH("thrust::get_temporary_buffer", MapNames::getDpctNamespace() + "get_temporary_allocation", "")
