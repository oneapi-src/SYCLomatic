//===--- MapNames.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"

#include <map>

using namespace clang;
using namespace clang::cu2sycl;

// DeviceProp names mapping.
const std::map<std::string, std::string> DevicePropVarRule::PropNamesMap{
    {"clockRate", "max_clock_frequency"},
    {"computeMode", "mode"},
    {"major", "major_version"},
    {"minor", "minor_version"},
    {"multiProcessorCount", "max_compute_units"},
    {"multiProcessorCount", "max_compute_units"},
    {"name", "name"}
    // ...
};

// Enum constants name mapping.
const std::map<std::string, std::string> EnumConstantRule::EnumNamesMap{
    // enum cudaComputeMode
    {"cudaComputeModeDefault", "compute_mode::default_"},
    {"cudaComputeModeExclusive", "compute_mode::exclusive"},
    {"cudaComputeModeProhibited", "compute_mode::prohibited"},
    {"cudaComputeModeExclusiveProcess", "compute_mode::exclusive_process"},
    // ...
    // enum cudaDeviceAttr
    // ...
    {"cudaDevAttrHostNativeAtomicSupported", "is_native_atomic_supported"},
    // ...
};
