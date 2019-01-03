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

#include "MapNames.h"
#include "ASTTraversal.h"
#include "SaveNewFiles.h"

#include <map>

using namespace clang;
using namespace clang::syclct;

// Type names mapping.
const std::map<std::string, std::string> MapNames::TypeNamesMap{
    {"cudaDeviceProp", "syclct::sycl_device_info"},
    {"cudaError_t", "int"},
    {"cudaError", "int"},
    {"dim3", "cl::sycl::range<3>"},
    {"int2", "cl::sycl::int2"},
    {"struct int2", "cl::sycl::int2"},
    {"double2", "cl::sycl::double2"},
    {"struct double2", "cl::sycl::double2"},
    // ...
};

// Atomic function names mapping
const std::unordered_map<std::string, std::string>
    AtomicFunctionRule::AtomicFuncNamesMap{
        {"atomicAdd", "syclct::atomic_fetch_add"},
        {"atomicSub", "syclct::atomic_fetch_sub"},
        {"atomicAnd", "syclct::atomic_fetch_and"},
        {"atomicOr", "syclct::atomic_fetch_or"},
        {"atomicXor", "syclct::atomic_fetch_xor"},
        {"atomicMin", "syclct::atomic_fetch_min"},
        {"atomicMax", "syclct::atomic_fetch_max"},
        {"atomicExch", "syclct::atomic_exchange"},
        {"atomicCAS", "syclct::atomic_compare_exchange_strong"},
    };

// CUDA dim3 dot member funciton names mapping.
const std::map<std::string, std::string> MapNames::Dim3MemberNamesMap{
    {"x", "[0]"}, {"y", "[1]"}, {"z", "[2]"},
    // ...
};

// CUDA dim3 pointer member funciton names mapping.
const std::map<std::string, std::string> MapNames::Dim3MemberPointerNamesMap{
    {"x", "operator[](0)"}, {"y", "operator[](1)"}, {"z", "operator[](2)"},
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
    {"name", "name"},
    {"totalGlobalMem", "global_mem_size"},
    // ...
};

// DeviceProp names mapping.
const std::map<std::string, std::string>
    VectorTypeMemberAccessRule::MemberNamesMap{
        {"x", "x()"}, {"y", "y()"}, {"z", "z()"}, {"w", "w()"},
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

// Math function names translation.
const std::map<std::string, std::string> MathFunctionsRule::FunctionNamesMap{
    // Cuda's max to sycl's max. all have integer types and double types.
    // See "4.13.4 Integer functions" and "4.13.5 Common functions"

    // <SYCL/sycl_math_builtins_common.h>
    {"abs", "cl::sycl::abs"},
    {"max", "cl::sycl::max"},
    {"min", "cl::sycl::min"},

    // <SYCL/sycl_math_builtins_floating_point.h>
    {"acos", "cl::sycl::acos"},
    {"acosh", "cl::sycl::acosh"},
    {"acospi", "cl::sycl::acospi"},
    {"asin", "cl::sycl::asin"},
    {"asinh", "cl::sycl::asinh"},
    {"atan", "cl::sycl::atan"},
    {"atanh", "cl::sycl::atanh"},
    {"cbrt", "cl::sycl::cbrt"},
    {"ceil", "cl::sycl::ceil"},
    {"cos", "cl::sycl::cos"},
    {"cosh", "cl::sycl::cosh"},
    {"cospi", "cl::sycl::cospi"},
    {"erfc", "cl::sycl::erfc"},
    {"erf", "cl::sycl::erf"},
    {"exp", "cl::sycl::exp"},
    {"exp2", "cl::sycl::exp2"},
    {"exp10", "cl::sycl::exp10"},
    {"expm1", "cl::sycl::expm1"},
    {"fabs", "cl::sycl::fabs"},
    {"floor", "cl::sycl::floor"},
    {"lgamma", "cl::sycl::lgamma"},
    {"log", "cl::sycl::log"},
    {"log2", "cl::sycl::log2"},
    {"log10", "cl::sycl::log10"},
    {"log1p", "cl::sycl::log1p"},
    {"logb", "cl::sycl::logb"},
    {"rint", "cl::sycl::rint"},
    {"round", "cl::sycl::round"},
    {"rsqrt", "cl::sycl::rsqrt"},
    {"sin", "cl::sycl::sin"},
    {"sinh", "cl::sycl::sinh"},
    {"sinpi", "cl::sycl::sinpi"},
    {"sqrt", "cl::sycl::sqrt"},
    {"tan", "cl::sycl::tan"},
    {"tanh", "cl::sycl::tanh"},
    {"tanpi", "cl::sycl::tanpi"},
    {"tgamma", "cl::sycl::tgamma"},
    {"trunc", "cl::sycl::trunc"},

    // Not in CUDA API but in SYCL API
    // {"asinpi", "cl::sycl::asinpi"},
    // {"atanpi", "cl::sycl::atanpi"},
    // {"clamp", "cl::sycl::clamp"},
    // {"degrees", "cl::sycl::degrees"},
    // {"mix", "cl::sycl::mix"},
    // {"radians", "cl::sycl::radians"},
    // {"sign", "cl::sycl::sign"},
    // {"smoothstep", "cl::sycl::smoothstep"},
    // {"step", "cl::sycl::step"},
};

// cudaFuncAttributes names translation
const std::map<std::string, std::string>
    KernelFunctionInfoRule::AttributesNamesMap{
        {"maxThreadsPerBlock", "max_work_group_size"},
    };

std::map<std::string, bool> TranslationStatistics::TranslationTable{
#define ENTRY(APINAME, VALUE) {#APINAME, VALUE},
#include "APINames.inc"
#undef ENTRY
};

bool TranslationStatistics::IsTranslated(const std::string &APIName) {
  auto Search = TranslationTable.find(APIName);
  if (Search != TranslationTable.end()) {
    return Search->second;
  } else {
    llvm::errs() << "[NOTE] Find new API\"" << APIName
                 << "\" , please update translated API database.\n";
    std::exit(TranslationError);
  }
}

std::vector<std::string> TranslationStatistics::GetAllAPINames(void) {
  std::vector<std::string> AllAPINames;
  for (const auto &APIName : TranslationTable) {
    AllAPINames.push_back(APIName.first);
  }

  return AllAPINames;
}
