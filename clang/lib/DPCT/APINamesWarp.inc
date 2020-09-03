//===---APINamesWarp.inc---------------------------------------------===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
ENTRY_WARP("__all", "sycl::ONEAPI::all_of")
ENTRY_WARP("__all_sync", "sycl::ONEAPI::all_of")
ENTRY_WARP("__any", "sycl::ONEAPI::any_of")
ENTRY_WARP("__any_sync", "sycl::ONEAPI::any_of")
ENTRY_WARP("__shfl", "shuffle")
ENTRY_WARP("__shfl_sync", "shuffle")
ENTRY_WARP("__shfl_up", "shuffle_up")
ENTRY_WARP("__shfl_up_sync", "shuffle_up")
ENTRY_WARP("__shfl_down", "shuffle_down")
ENTRY_WARP("__shfl_down_sync", "shuffle_down")
ENTRY_WARP("__shfl_xor", "shuffle_xor")
ENTRY_WARP("__shfl_xor_sync", "shuffle_xor")

ENTRY_WARP("__activemask", "Unsupported: No such API in DPC++")
ENTRY_WARP("__ballot", "Unsupported: No such API in DPC++")
ENTRY_WARP("__ballot_sync", "Unsupported: No such API in DPC++")
