//===--------------- APINames_removed.inc ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ENTRY(cublasCreate_v2,                 "this call is redundant in SYCL.")
ENTRY(cublasDestroy_v2,                "this call is redundant in SYCL.")

ENTRY(cublasInit,                      "this call is redundant in SYCL.")
ENTRY(cublasShutdown,                  "this call is redundant in SYCL.")
ENTRY(cublasGetError,                  "this call is redundant in SYCL.")

ENTRY(cusolverDnCreate,                "this call is redundant in SYCL.")
ENTRY(cusolverDnDestroy,               "this call is redundant in SYCL.")
ENTRY(cusolverDnSpotrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnDpotrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnCpotrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnZpotrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnSpotri_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnDpotri_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnCpotri_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnZpotri_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnSgetrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnDgetrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnCgetrf_bufferSize,     "this call is redundant in SYCL.")
ENTRY(cusolverDnZgetrf_bufferSize,     "this call is redundant in SYCL.")

ENTRY(cudaDeviceSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaThreadSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaDeviceSetCacheConfig,        "SYCL currently does not support setting cache config on devices.")
ENTRY(cudaDeviceGetCacheConfig,        "SYCL currently does not support setting cache config on devices.")

ENTRY(cudaEventCreate,                 "this call is redundant in SYCL.")
ENTRY(cudaEventCreateWithFlags,        "this call is redundant in SYCL.")
ENTRY(cudaEventDestroy,                "this call is redundant in SYCL.")
ENTRY(cuEventDestroy_v2,               "this call is redundant in SYCL.")
ENTRY(cuEventCreate,                   "this call is redundant in SYCL.")

ENTRY(cudaStreamAttachMemAsync,        "SYCL currently does not support associating USM with a specific queue.")
ENTRY(cuStreamAttachMemAsync,          "SYCL currently does not support associating USM with a specific queue.")
ENTRY(cudaStreamQuery,                 "SYCL currently does not support query operations on queues.")
ENTRY(cudaStreamBeginCapture,          "SYCL currently does not support capture operations on queues.")
ENTRY(cudaStreamEndCapture,            "SYCL currently does not support capture operations on queues.")
ENTRY(cudaStreamIsCapturing,           "SYCL currently does not support capture operations on queues.")

ENTRY(cudaHostRegister,                "SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cudaHostUnregister,              "SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cuMemHostRegister_v2,            "SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cuMemHostUnregister,             "SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")