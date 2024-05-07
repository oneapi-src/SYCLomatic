//===--------------- APINames_removed.inc ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ENTRY(cublasInit,                      "this functionality is redundant in SYCL.")
ENTRY(cublasShutdown,                  "this functionality is redundant in SYCL.")
ENTRY(cublasGetError,                  "this functionality is redundant in SYCL.")

ENTRY(cudaDeviceSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaThreadSetLimit,              "SYCL currently does not support setting resource limits on devices.")
ENTRY(cudaDeviceSetCacheConfig,        "SYCL currently does not support setting cache config on devices.")
ENTRY(cudaDeviceGetCacheConfig,        "SYCL currently does not support setting cache config on devices.")
ENTRY(cudaCtxResetPersistingL2Cache,   "SYCL currently does not support setting cache config on devices.")
ENTRY(cuCtxResetPersistingL2Cache,     "SYCL currently does not support setting cache config on devices.")

ENTRY(cuCtxSetLimit,                   "SYCL currently does not support setting resource limits on devices.")
ENTRY(cuCtxGetLimit,                   "SYCL currently does not support getting resource limits on devices.")
ENTRY(cuCtxSetCacheConfig,             "SYCL currently does not support setting cache config on devices.")

ENTRY(cudaEventCreate,                 "this functionality is redundant in SYCL.")
ENTRY(cudaEventCreateWithFlags,        "this functionality is redundant in SYCL.")
ENTRY(cudaEventDestroy,                "this functionality is redundant in SYCL.")
ENTRY(cuEventDestroy_v2,               "this functionality is redundant in SYCL.")
ENTRY(cuEventCreate,                   "this functionality is redundant in SYCL.")

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

ENTRY(cudaProfilerStart,               "SYCL currently does not support this function. Remove the profiler API will not impact the outcome.")
ENTRY(cudaProfilerStop,                "SYCL currently does not support this function. Remove the profiler API will not impact the outcome.")
