//===--------------- APINames_removed.inc ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ENTRY(cublasCreate_v2,                 "this call is redundant in DPC++.")
ENTRY(cublasDestroy_v2,                "this call is redundant in DPC++.")

ENTRY(cublasInit,                      "this call is redundant in DPC++.")
ENTRY(cublasShutdown,                  "this call is redundant in DPC++.")
ENTRY(cublasGetError,                  "this call is redundant in DPC++.")

ENTRY(cusolverDnCreate,                "this call is redundant in DPC++.")
ENTRY(cusolverDnDestroy,               "this call is redundant in DPC++.")
ENTRY(cusolverDnSpotrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnDpotrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnCpotrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnZpotrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnSpotri_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnDpotri_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnCpotri_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnZpotri_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnSgetrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnDgetrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnCgetrf_bufferSize,     "this call is redundant in DPC++.")
ENTRY(cusolverDnZgetrf_bufferSize,     "this call is redundant in DPC++.")

ENTRY(cudaDeviceSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaThreadSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaDeviceSetCacheConfig,        "DPC++ currently does not support setting cache config on devices.")
ENTRY(cudaDeviceGetCacheConfig,        "DPC++ currently does not support setting cache config on devices.")

ENTRY(cudaEventCreate,                 "this call is redundant in DPC++.")
ENTRY(cudaEventCreateWithFlags,        "this call is redundant in DPC++.")
ENTRY(cudaEventDestroy,                "this call is redundant in DPC++.")
ENTRY(cuEventDestroy_v2,               "this call is redundant in DPC++.")
ENTRY(cuEventCreate,                   "this call is redundant in DPC++.")

ENTRY(cudaStreamAttachMemAsync,        "SYCL currently does not support associating USM with a specific queue.")
ENTRY(cuStreamAttachMemAsync,          "SYCL currently does not support associating USM with a specific queue.")
ENTRY(cudaStreamQuery,                 "SYCL currently does not support query operations on queues.")
ENTRY(cudaStreamBeginCapture,          "SYCL currently does not support capture operations on queues.")
ENTRY(cudaStreamEndCapture,            "SYCL currently does not support capture operations on queues.")
ENTRY(cudaStreamIsCapturing,           "SYCL currently does not support capture operations on queues.")

ENTRY(cudaHostRegister,                "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cudaHostUnregister,              "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cuMemHostRegister_v2,            "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cuMemHostUnregister,             "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")