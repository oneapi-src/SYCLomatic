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
const MapNames::MapTy MapNames::TypeNamesMap{
    {"cudaDeviceProp", "syclct::sycl_device_info"},
    {"cudaError_t", "int"},
    {"cudaError", "int"},
    {"dim3", "cl::sycl::range<3>"},
    {"int2", "cl::sycl::int2"},
    {"struct int2", "cl::sycl::int2"},
    {"double2", "cl::sycl::double2"},
    {"struct double2", "cl::sycl::double2"},
    {"cudaEvent_t", "cl::sycl::event"},
    {"cudaStream_t", "cl::sycl::queue"},
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
const MapNames::MapTy MapNames::Dim3MemberNamesMap{
    {"x", "[0]"}, {"y", "[1]"}, {"z", "[2]"},
    // ...
};

const MapNames::MapTy MapNames::MacrosMap{
    {"__CUDA_ARCH__", "DPCPP_COMPATIBILITY_TEMP"}, /**/
    {"__NVCC__", "DPCPP_COMPATIBILITY_TEMP"},      /**/
    {"__CUDACC__", "DPCPP_COMPATIBILITY_TEMP"},
    //...
};

// DeviceProp names mapping.
const MapNames::MapTy DevicePropVarRule::PropNamesMap{
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
const MapNames::MapTy
    VectorTypeMemberAccessRule::MemberNamesMap{
        {"x", "x()"}, {"y", "y()"}, {"z", "z()"}, {"w", "w()"},
        // ...
    };

// Enum constants name mapping.
const MapNames::MapTy EnumConstantRule::EnumNamesMap{
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

// Math function names migration.
const MapNames::MapTy MathFunctionsRule::FunctionNamesMap{
    // Cuda's max to sycl's max. all have integer types and double types.
    // See "4.13.4 Integer functions" and "4.13.5 Common functions"

    // <SYCL/sycl_math_builtins_common.h>
    {"abs", "cl::sycl::abs"}, /* for Int type => abs, float, double => fabs*/
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

    // Double precision intrinisics
    {"__dadd_rd", ""},
    {"__dadd_rn", ""},
    {"__dadd_ru", ""},
    {"__dadd_rz", ""},
    {"__dsub_rd", ""},
    {"__dsub_rn", ""},
    {"__dsub_ru", ""},
    {"__dsub_rz", ""},
    {"__dmul_rd", ""},
    {"__dmul_rn", ""},
    {"__dmul_ru", ""},
    {"__dmul_rz", ""},
    {"__ddiv_rd", ""},
    {"__ddiv_rn", ""},
    {"__ddiv_ru", ""},
    {"__ddiv_rz", ""},
    {"__drcp_rd", "cl::sycl::recip"},
    {"__drcp_rn", "cl::sycl::recip"},
    {"__drcp_ru", "cl::sycl::recip"},
    {"__drcp_rz", "cl::sycl::recip"},
    {"__dsqrt_rd", "cl::sycl::sqrt"},
    {"__dsqrt_rn", "cl::sycl::sqrt"},
    {"__dsqrt_ru", "cl::sycl::sqrt"},
    {"__dsqrt_rz", "cl::sycl::sqrt"},
    {"__fma_rd", "cl::sycl::fma"},
    {"__fma_rn", "cl::sycl::fma"},
    {"__fma_ru", "cl::sycl::fma"},
    {"__fma_rz", "cl::sycl::fma"},

    // Single precision intrinisics
    {"__fadd_rd", ""},
    {"__fadd_rn", ""},
    {"__fadd_ru", ""},
    {"__fadd_rz", ""},
    {"__fsub_rd", ""},
    {"__fsub_rn", ""},
    {"__fsub_ru", ""},
    {"__fsub_rz", ""},
    {"__fmul_rd", ""},
    {"__fmul_rn", ""},
    {"__fmul_ru", ""},
    {"__fmul_rz", ""},
    {"__fdiv_rd", ""},
    {"__fdiv_rn", ""},
    {"__fdiv_ru", ""},
    {"__fdiv_rz", ""},
    {"__frcp_rd", "cl::sycl::recip"},
    {"__frcp_rn", "cl::sycl::recip"},
    {"__frcp_rn", "cl::sycl::recip"},
    {"__frcp_rz", "cl::sycl::recip"},
    {"__fsqrt_rd", "cl::sycl::sqrt"},
    {"__fsqrt_rn", "cl::sycl::sqrt"},
    {"__fsqrt_ru", "cl::sycl::sqrt"},
    {"__fsqrt_rz", "cl::sycl::sqrt"},
    {"__fmaf_rd", "cl::sycl::fma"},
    {"__fmaf_rn", "cl::sycl::fma"},
    {"__fmaf_ru", "cl::sycl::fma"},
    {"__fmaf_rz", "cl::sycl::fma"},
    {"__cosf", "cl::sycl::cos"},
    {"__exp10f", "cl::sycl::exp10"},
    {"__expf", "cl::sycl::exp"},
    {"__fdividef", "cl::sycl::divide"},
    {"__frsqrt_rn", "cl::sycl::rsqrt"},
    {"__log10f", "cl::sycl::log10"},
    {"__log2f", "cl::sycl::log2"},
    {"__logf", "cl::sycl::log"},
    {"__powf", "cl::sycl::pow"},
    {"__sincosf", ""},
    {"__sinf", "cl::sycl::sin"},
    {"__tanf", "cl::sycl::tan"},

    {"fmax", "cl::sycl::fmax"},
    {"fmin", "cl::sycl::fmin"},
    {"ceil", "cl::sycl::ceil"},
    {"floor", "cl::sycl::floor"},
    {"nan", "cl::sycl::nan"},
    {"fma", "cl::sycl::fma"},

    {"fmaxf", "cl::sycl::fmax"},
    {"fminf", "cl::sycl::fmin"},
    {"ceilf", "cl::sycl::ceil"},
    {"floorf", "cl::sycl::floor"},
    {"nanf", "cl::sycl::nan"},
    {"fmaf", "cl::sycl::fma"},

    // Type casting intrinsics
    {"__int_as_float", ""},
    {"__float_as_int", ""},
    {"__uint_as_float", ""},
    {"__float_as_uint", ""},
    {"__longlong_as_double", ""},
    {"__double_as_longlong", ""},

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

// cudaFuncAttributes names migration
const MapNames::MapTy KernelFunctionInfoRule::AttributesNamesMap{
    {"maxThreadsPerBlock", "max_work_group_size"},
};

std::map<std::string, bool> TranslationStatistics::TranslationTable{
#define ENTRY(APINAME, VALUE, TARGET, COMMENT) {#APINAME, VALUE},
#include "APINames.inc"
#undef ENTRY
};

bool TranslationStatistics::IsTranslated(const std::string &APIName) {
  auto Search = TranslationTable.find(APIName);
  if (Search != TranslationTable.end()) {
    return Search->second;
  } else {
    llvm::errs() << "[NOTE] Find new API\"" << APIName
                 << "\" , please update migrated API database.\n";
    std::exit(MigrationError);
  }
}

std::vector<std::string> TranslationStatistics::GetAllAPINames(void) {
  std::vector<std::string> AllAPINames;
  for (const auto &APIName : TranslationTable) {
    AllAPINames.push_back(APIName.first);
  }

  return AllAPINames;
}
