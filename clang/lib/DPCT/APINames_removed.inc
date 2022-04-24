//===---APINames_removed.inc-------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
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

ENTRY(cudaDeviceSetLimit,              "DPC++ currently does not support setting resource limits on devices.")
ENTRY(cudaThreadSetLimit,              "DPC++ currently does not support setting resource limits on devices.")
ENTRY(cudaDeviceSetCacheConfig,        "DPC++ currently does not support setting cache config on devices.")
ENTRY(cudaDeviceGetCacheConfig,        "DPC++ currently does not support setting cache config on devices.")

ENTRY(cudaEventCreate,                 "this call is redundant in DPC++.")
ENTRY(cudaEventCreateWithFlags,        "this call is redundant in DPC++.")
ENTRY(cudaEventDestroy,                "this call is redundant in DPC++.")
ENTRY(cuEventCreate,                 "this call is redundant in DPC++.")

ENTRY(cudaStreamAttachMemAsync,        "DPC++ currently does not support associating USM with a specific queue.")
ENTRY(cudaStreamQuery,                 "DPC++ currently does not support query operations on queues.")
ENTRY(cudaStreamBeginCapture,          "DPC++ currently does not support capture operations on queues.")
ENTRY(cudaStreamEndCapture,            "DPC++ currently does not support capture operations on queues.")
ENTRY(cudaStreamIsCapturing,           "DPC++ currently does not support capture operations on queues.")

ENTRY(cudaHostRegister,                "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
ENTRY(cudaHostUnregister,              "DPC++ currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.")
