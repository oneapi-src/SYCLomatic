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
using namespace clang::syclct;

// Type names mapping.
const std::map<std::string, std::string> TypeInVarDeclRule::TypeNamesMap{
    {"cudaDeviceProp", "syclct::sycl_device_info"},
    {"cudaError_t", "int"},
    {"cudaError", "int"},
    {"dim3", "cl::sycl::range<3>"},
    {"int2", "cl::sycl::int2"},
    // ...
};

// DeviceProp names mapping.
const std::map<std::string, std::string> DevicePropVarRule::PropNamesMap{
    {"clockRate", "max_clock_frequency"},
    {"computeMode", "mode"},
    {"major", "major_version"},
    {"minor", "minor_version"},
    {"integrated", "get_integrated"},
    {"multiProcessorCount", "max_compute_units"},
    {"name", "name"}
    // ...
};

// DeviceProp names mapping.
const std::map<std::string, std::string> SyclStyleVectorRule::MemberNamesMap{
    {"x", "x()"}, {"y", "y()"}, {"z", "z()"},
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
    // enum cudaMemcpyKind
    {"cudaMemcpyHostToHost", "host_to_host"},
    {"cudaMemcpyHostToDevice", "host_to_device"},
    {"cudaMemcpyDeviceToHost", "device_to_host"},
    {"cudaMemcpyDeviceToDevice", "device_to_device"},
    {"cudaMemcpyDefault", "automatic"},
    // ...
};
