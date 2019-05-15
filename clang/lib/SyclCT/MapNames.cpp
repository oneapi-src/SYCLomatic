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
    {"__half", "cl::sycl::half"},
    {"__half2", "cl::sycl::half2"},
    {"half", "cl::sycl::half"},
    {"half2", "cl::sycl::half2"},
    {"cudaEvent_t", "cl::sycl::event"},
    {"cudaStream_t", "cl::sycl::queue"},
    {"char1", "char"},
    {"char2", "cl::sycl::char2"},
    {"char3", "cl::sycl::char3"},
    {"char4", "cl::sycl::char4"},
    {"double1", "double"},
    {"double2", "cl::sycl::double2"},
    {"double3", "cl::sycl::double3"},
    {"double4", "cl::sycl::double4"},
    {"float1", "float"},
    {"float2", "cl::sycl::float2"},
    {"float3", "cl::sycl::float3"},
    {"float4", "cl::sycl::float4"},
    {"int1", "int"},
    {"int2", "cl::sycl::int2"},
    {"int3", "cl::sycl::int3"},
    {"int4", "cl::sycl::int4"},
    {"long1", "long"},
    {"long2", "cl::sycl::long2"},
    {"long3", "cl::sycl::long3"},
    {"long4", "cl::sycl::long4"},
    {"longlong1", "long long"},
    {"longlong2", "cl::sycl::longlong2"},
    {"longlong3", "cl::sycl::longlong3"},
    {"longlong4", "cl::sycl::longlong4"},
    {"short1", "short"},
    {"short2", "cl::sycl::short2"},
    {"short3", "cl::sycl::short3"},
    {"short4", "cl::sycl::short4"},
    {"uchar1", "unsigned char"},
    {"uchar2", "cl::sycl::uchar2"},
    {"uchar3", "cl::sycl::uchar3"},
    {"uchar4", "cl::sycl::uchar4"},
    {"uint1", "unsigned int"},
    {"uint2", "cl::sycl::uint2"},
    {"uint3", "cl::sycl::uint3"},
    {"uint4", "cl::sycl::uint4"},
    {"ulong1", "unsigned long"},
    {"ulong2", "cl::sycl::ulong2"},
    {"ulong3", "cl::sycl::ulong3"},
    {"ulong4", "cl::sycl::ulong4"},
    {"ulonglong1", "unsigned long long"},
    {"ulonglong2", "cl::sycl::ulonglong2"},
    {"ulonglong3", "cl::sycl::ulonglong3"},
    {"ulonglong4", "cl::sycl::ulonglong4"},
    {"ushort1", "unsigned short"},
    {"ushort2", "cl::sycl::ushort2"},
    {"ushort3", "cl::sycl::ushort3"},
    {"ushort4", "cl::sycl::ushort4"},
    {"cublasHandle_t", "cl::sycl::queue"},
    {"cublasStatus_t", "int"},
    {"cuComplex", "std::complex<float>"},
    {"cuDoubleComplex", "std::complex<double>"},
    // ...
};

// BLAS function names mapping
const MapNames::MapTy MapNames::BLASFunctionNamesMap{
    {"cublasSgemm_v2", "mkl::Sgemm"},
    {"cublasDgemm_v2", "mkl::Dgemm"},
    {"cublasCgemm_v2", "mkl::Cgemm"},
    {"cublasZgemm_v2", "mkl::Zgemm"},
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
const MapNames::MapTy VectorTypeMemberAccessRule::MemberNamesMap{
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

const MapNames::MapTy MathFunctionsRule::HalfFunctionNamesMap{
    // Half Arithmetic Functions
    {"__h2div", "/"},
    {"__hadd_sat", StringLiteralUnsupported},
    {"__hdiv", "/"},
    {"__hfma", "cl::sycl::fma"},
    {"__hfma_sat", StringLiteralUnsupported},
    {"__hmul", "*"},
    {"__hmul_sat", StringLiteralUnsupported},
    {"__hneg", "-"},
    {"__hsub", "-"},
    {"__hsub_sat", StringLiteralUnsupported},

    // Half2 Arithmetic Functions
    {"__hadd2_sat", StringLiteralUnsupported},
    {"__hfma2", "cl::sycl::fma"},
    {"__hfma2_sat", StringLiteralUnsupported},
    {"__hmul2", "*"},
    {"__hmul2_sat", StringLiteralUnsupported},
    {"__hneg2", "-"},
    {"__hsub2", "-"},
    {"__hsub2_sat", StringLiteralUnsupported},

    // Half Comparison Functions
    {"__heq", "=="},
    {"__hequ", StringLiteralUnsupported},
    {"__hge", ">="},
    {"__hgeu", StringLiteralUnsupported},
    {"__hgt", ">"},
    {"__hgtu", StringLiteralUnsupported},
    {"__hisinf", "cl::sycl::isinf"},
    {"__hisnan", "cl::sycl::isnan"},
    {"__hle", "<="},
    {"__hleu", StringLiteralUnsupported},
    {"__hlt", "<"},
    {"__hltu", StringLiteralUnsupported},
    {"__hne", "!="},
    {"__hneu", StringLiteralUnsupported},

    // Half2 Comparison Functions
    {"__hbeq2", StringLiteralUnsupported},
    {"__hbequ2", StringLiteralUnsupported},
    {"__hbge2", StringLiteralUnsupported},
    {"__hbgeu2", StringLiteralUnsupported},
    {"__hbgt2", StringLiteralUnsupported},
    {"__hbgtu2", StringLiteralUnsupported},
    {"__hble2", StringLiteralUnsupported},
    {"__hbleu2", StringLiteralUnsupported},
    {"__hblt2", StringLiteralUnsupported},
    {"__hbltu2", StringLiteralUnsupported},
    {"__hbne2", StringLiteralUnsupported},
    {"__hbneu2", StringLiteralUnsupported},
    {"__heq2", StringLiteralUnsupported},
    {"__hequ2", StringLiteralUnsupported},
    {"__hge2", StringLiteralUnsupported},
    {"__hgeu2", StringLiteralUnsupported},
    {"__hgt2", StringLiteralUnsupported},
    {"__hgtu2", StringLiteralUnsupported},
    {"__hisnan2", StringLiteralUnsupported},
    {"__hle2", StringLiteralUnsupported},
    {"__hleu2", StringLiteralUnsupported},
    {"__hlt2", StringLiteralUnsupported},
    {"__hltu2", StringLiteralUnsupported},
    {"__hne2", StringLiteralUnsupported},
    {"__hneu2", StringLiteralUnsupported},

    // Half Math Functions
    {"hceil", "cl::sycl::ceil"},
    {"hcos", "cl::sycl::cos"},
    {"hexp", "cl::sycl::exp"},
    {"hexp10", "cl::sycl::exp10"},
    {"hexp2", "cl::sycl::exp2"},
    {"hfloor", "cl::sycl::floor"},
    {"hlog", "cl::sycl::log"},
    {"hlog10", "cl::sycl::log10"},
    {"hlog2", "cl::sycl::log2"},
    {"hrcp", StringLiteralUnsupported},
    {"hrint", "cl::sycl::rint"},
    {"hrsqrt", "cl::sycl::rsqrt"},
    {"hsin", "cl::sycl::sin"},
    {"hsqrt", "cl::sycl::sqrt"},
    {"htrunc", "cl::sycl::trunc"},

    // Half2 Math Functions
    {"h2ceil", "cl::sycl::ceil"},
    {"h2cos", "cl::sycl::cos"},
    {"h2exp", "cl::sycl::exp"},
    {"h2exp10", "cl::sycl::exp10"},
    {"h2exp2", "cl::sycl::exp2"},
    {"h2floor", "cl::sycl::floor"},
    {"h2log", "cl::sycl::log"},
    {"h2log10", "cl::sycl::log10"},
    {"h2log2", "cl::sycl::log2"},
    {"h2rcp", StringLiteralUnsupported},
    {"h2rint", "cl::sycl::rint"},
    {"h2rsqrt", "cl::sycl::rsqrt"},
    {"h2sin", "cl::sycl::sin"},
    {"h2sqrt", "cl::sycl::sqrt"},
    {"h2trunc", "cl::sycl::trunc"},
};

const MapNames::MapTy MathFunctionsRule::SingleDoubleFunctionNamesMap{
    // Single Precision Mathematical Functions
    {"log", "cl::sycl::log"},
    {"logf", "cl::sycl::log"},
    {"acosf", "cl::sycl::acos"},
    {"acoshf", "cl::sycl::acosh"},
    {"asinf", "cl::sycl::asin"},
    {"asinhf", "cl::sycl::asinh"},
    {"atan2f", "cl::sycl::atan2"},
    {"atanf", "cl::sycl::atan"},
    {"atanhf", "cl::sycl::atanh"},
    {"cbrtf", "cl::sycl::cbrt"},
    {"ceilf", "cl::sycl::ceil"},
    {"copysignf", "cl::sycl::copysign"},
    {"cosf", "cl::sycl::cos"},
    {"coshf", "cl::sycl::cosh"},
    {"cospif", "cl::sycl::cospi"},
    {"cyl_bessel_i0f", StringLiteralUnsupported},
    {"cyl_bessel_i1f", StringLiteralUnsupported},
    {"erfcf", "cl::sycl::erfc"},
    {"erfcinvf", StringLiteralUnsupported},
    {"erfcxf", StringLiteralUnsupported},
    {"erff", "cl::sycl::erf"},
    {"erfinvf", StringLiteralUnsupported},
    {"exp10f", "cl::sycl::exp10"},
    {"exp2f", "cl::sycl::exp2"},
    {"expf", "cl::sycl::exp"},
    {"expm1f", "cl::sycl::expm1"},
    {"fabsf", "cl::sycl::fabs"},
    {"fdimf", "cl::sycl::fdim"},
    {"fdividef", "cl::sycl::native::divide"},
    {"floorf", "cl::sycl::floor"},
    {"fmaf", "cl::sycl::fma"},
    {"fmaxf", "cl::sycl::fmax"},
    {"fminf", "cl::sycl::fmin"},
    {"fmodf", "cl::sycl::fmod"},
    {"frexpf", "cl::sycl::frexp"},
    {"hypotf", "cl::sycl::hypot"},
    {"ilogbf", "cl::sycl::ilogb"},
    {"isfinite", "cl::sycl::isfinite"},
    {"isinf", "cl::sycl::isinf"},
    {"isnan", "cl::sycl::isnan"},
    {"j0f", StringLiteralUnsupported},
    {"j1f", StringLiteralUnsupported},
    {"jnf", StringLiteralUnsupported},
    {"ldexpf", "cl::sycl::ldexp"},
    {"lgammaf", "cl::sycl::lgamma"},
    {"llrintf", "cl::sycl::rint"},
    {"llroundf", "cl::sycl::round"},
    {"log10f", "cl::sycl::log10"},
    {"log1pf", "cl::sycl::log1p"},
    {"log2f", "cl::sycl::log2"},
    {"logbf", "cl::sycl::logb"},
    {"lrintf", "cl::sycl::rint"},
    {"lroundf", "cl::sycl::round"},
    {"modff", "cl::sycl::modf"},
    {"nanf", "cl::sycl::nan"},
    {"nearbyintf", "cl::sycl::floor"}, // nearbyintf(x) => cl::sycl::floor(x + 0.5)
    {"nextafterf", "cl::sycl::nextafter"},
    {"norm3df", StringLiteralUnsupported},
    {"norm4df", StringLiteralUnsupported},
    {"normcdff", StringLiteralUnsupported},
    {"normcdfinvf", StringLiteralUnsupported},
    {"normf", StringLiteralUnsupported},
    {"powf", "cl::sycl::pow"},
    {"rcbrtf", StringLiteralUnsupported},
    {"remainderf", "cl::sycl::remainder"},
    {"remquof", "cl::sycl::remquo"},
    {"rhypotf", "cl::sycl::hypot"},    // rhypotf(x, y) => 1 / cl::sycl::hypot(x, y)
    {"rintf", "cl::sycl::rint"},
    {"rnorm3df", StringLiteralUnsupported},
    {"rnorm4df", StringLiteralUnsupported},
    {"rnormf", StringLiteralUnsupported},
    {"roundf", "cl::sycl::round"},
    {"rsqrtf", "cl::sycl::rsqrt"},
    {"scalblnf", StringLiteralUnsupported},
    {"scalbnf", StringLiteralUnsupported},
    {"signbit", "cl::sycl::signbit"},
    {"sincosf", "cl::sycl::sincos"},   // sincospif(x, &y, &z) => y = cl::sycl::sincos(x, &z)
    {"sincospif", "cl::sycl::sincos"}, // sincospif(x, &y, &z) => y = cl::sycl::sincos(x * SYCLCT_PI_F, &z)
    {"sinf", "cl::sycl::sin"},
    {"sinhf", "cl::sycl::sinh"},
    {"sinpif", "cl::sycl::sinpi"},
    {"sqrtf", "cl::sycl::sqrt"},
    {"tanf", "cl::sycl::tan"},
    {"tanhf", "cl::sycl::tanh"},
    {"tgammaf", "cl::sycl::tgamma"},
    {"truncf", "cl::sycl::trunc"},
    {"y0f", StringLiteralUnsupported},
    {"y1f", StringLiteralUnsupported},
    {"ynf", StringLiteralUnsupported},

    // Double Precision Mathematical Functions
    {"acos", "cl::sycl::acos"},
    {"acosh", "cl::sycl::acosh"},
    {"asin", "cl::sycl::asin"},
    {"asinh", "cl::sycl::asinh"},
    {"atan2", "cl::sycl::atan2"},
    {"atan", "cl::sycl::atan"},
    {"atanh", "cl::sycl::atanh"},
    {"cbrt", "cl::sycl::cbrt"},
    {"ceil", "cl::sycl::ceil"},
    {"copysign", "cl::sycl::copysign"},
    {"cos", "cl::sycl::cos"},
    {"cosh", "cl::sycl::cosh"},
    {"cospi", "cl::sycl::cospi"},
    {"cyl_bessel_i0", StringLiteralUnsupported},
    {"cyl_bessel_i1", StringLiteralUnsupported},
    {"erfc", "cl::sycl::erfc"},
    {"erfcinv", StringLiteralUnsupported},
    {"erfcx", StringLiteralUnsupported},
    {"erf", "cl::sycl::erf"},
    {"erfinv", StringLiteralUnsupported},
    {"exp10", "cl::sycl::exp10"},
    {"exp2", "cl::sycl::exp2"},
    {"exp", "cl::sycl::exp"},
    {"expm1", "cl::sycl::expm1"},
    {"fabs", "cl::sycl::fabs"},
    {"fdim", "cl::sycl::fdim"},
    {"floor", "cl::sycl::floor"},
    {"fma", "cl::sycl::fma"},
    {"fmax", "cl::sycl::fmax"},
    {"fmin", "cl::sycl::fmin"},
    {"fmod", "cl::sycl::fmod"},
    {"frexp", "cl::sycl::frexp"},
    {"hypot", "cl::sycl::hypot"},
    {"ilogb", "cl::sycl::ilogb"},
    {"j0", StringLiteralUnsupported},
    {"j1", StringLiteralUnsupported},
    {"jn", StringLiteralUnsupported},
    {"ldexp", "cl::sycl::ldexp"},
    {"lgamma", "cl::sycl::lgamma"},
    {"llrint", "cl::sycl::rint"},
    {"llround", "cl::sycl::round"},
    {"log10", "cl::sycl::log10"},
    {"log1p", "cl::sycl::log1p"},
    {"log2", "cl::sycl::log2"},
    {"logb", "cl::sycl::logb"},
    {"lrint", "cl::sycl::rint"},
    {"lround", "cl::sycl::round"},
    {"modf", "cl::sycl::modf"},
    {"nan", "cl::sycl::nan"},
    {"nearbyint", "cl::sycl::floor"}, // nearbyint(x) => cl::sycl::floor(x + 0.5)
    {"nextafter", "cl::sycl::nextafter"},
    {"norm", StringLiteralUnsupported},
    {"norm3d", StringLiteralUnsupported},
    {"norm4d", StringLiteralUnsupported},
    {"normcdf", StringLiteralUnsupported},
    {"normcdfinv", StringLiteralUnsupported},
    {"pow", "cl::sycl::pow"},
    {"rcbrt", StringLiteralUnsupported},
    {"remainder", "cl::sycl::remainder"},
    {"remquo", "cl::sycl::remquo"},
    {"rhypot", "cl::sycl::hypot"}, // rhypot(x, y) => 1 / cl::sycl::hypot(x, y)
    {"rint", "cl::sycl::rint"},
    {"rnorm3d", StringLiteralUnsupported},
    {"rnorm4d", StringLiteralUnsupported},
    {"rnorm", StringLiteralUnsupported},
    {"round", "cl::sycl::round"},
    {"rsqrt", "cl::sycl::rsqrt"},
    {"scalbln", StringLiteralUnsupported},
    {"scalbn", StringLiteralUnsupported},
    {"sincos", "cl::sycl::sincos"},   // sincospi(x, &y, &z) => y = cl::sycl::sincos(x, &z)
    {"sincospi", "cl::sycl::sincos"}, // sincospi(x, &y, &z) => y = cl::sycl::sincos(x * SYCLCT_PI, &z)
    {"sin", "cl::sycl::sin"},
    {"sinh", "cl::sycl::sinh"},
    {"sinpi", "cl::sycl::sinpi"},
    {"sqrt", "cl::sycl::sqrt"},
    {"tan", "cl::sycl::tan"},
    {"tanh", "cl::sycl::tanh"},
    {"tgamma", "cl::sycl::tgamma"},
    {"trunc", "cl::sycl::trunc"},
    {"y0", StringLiteralUnsupported},
    {"y1", StringLiteralUnsupported},
    {"yn", StringLiteralUnsupported},

    // Single precision intrinisics
    {"__cosf", "cl::sycl::cos"},
    {"__exp10f", "cl::sycl::exp10"},
    {"__expf", "cl::sycl::exp"},
    {"__fadd_rd", "+"},
    {"__fadd_rn", "+"},
    {"__fadd_ru", "+"},
    {"__fadd_rz", "+"},
    {"__fdiv_rd", "/"},
    {"__fdiv_rn", "/"},
    {"__fdiv_ru", "/"},
    {"__fdiv_rz", "/"},
    {"__fdividef", "cl::sycl::native::divide"},
    {"__fmaf_rd", "cl::sycl::fma"},
    {"__fmaf_rn", "cl::sycl::fma"},
    {"__fmaf_ru", "cl::sycl::fma"},
    {"__fmaf_rz", "cl::sycl::fma"},
    {"__fmul_rd", "*"},
    {"__fmul_rn", "*"},
    {"__fmul_ru", "*"},
    {"__fmul_rz", "*"},
    {"__frcp_rd", "cl::sycl::native::recip"},
    {"__frcp_rn", "cl::sycl::native::recip"},
    {"__frcp_ru", "cl::sycl::native::recip"},
    {"__frcp_rz", "cl::sycl::native::recip"},
    {"__frsqrt_rn", "cl::sycl::rsqrt"},
    {"__fsqrt_rd", "cl::sycl::sqrt"},
    {"__fsqrt_rn", "cl::sycl::sqrt"},
    {"__fsqrt_ru", "cl::sycl::sqrt"},
    {"__fsqrt_rz", "cl::sycl::sqrt"},
    {"__fsub_rd", "-"},
    {"__fsub_rn", "-"},
    {"__fsub_ru", "-"},
    {"__fsub_rz", "-"},
    {"__log10f", "cl::sycl::log10"},
    {"__log2f", "cl::sycl::log2"},
    {"__logf", "cl::sycl::log"},
    {"__powf", "cl::sycl::pow"},
    {"__saturatef", StringLiteralUnsupported},
    {"__sincosf", "cl::sycl::sincos"},
    {"__sinf", "cl::sycl::sin"},
    {"__tanf", "cl::sycl::tan"},

    // Double precision intrinisics
    {"__dadd_rd", "+"},
    {"__dadd_rn", "+"},
    {"__dadd_ru", "+"},
    {"__dadd_rz", "+"},
    {"__ddiv_rd", "/"},
    {"__ddiv_rn", "/"},
    {"__ddiv_ru", "/"},
    {"__ddiv_rz", "/"},
    {"__dmul_rd", "*"},
    {"__dmul_rn", "*"},
    {"__dmul_ru", "*"},
    {"__dmul_rz", "*"},
    {"__drcp_rd", StringLiteralUnsupported},
    {"__drcp_rn", StringLiteralUnsupported},
    {"__drcp_ru", StringLiteralUnsupported},
    {"__drcp_rz", StringLiteralUnsupported},
    {"__dsqrt_rd", "cl::sycl::sqrt"},
    {"__dsqrt_rn", "cl::sycl::sqrt"},
    {"__dsqrt_ru", "cl::sycl::sqrt"},
    {"__dsqrt_rz", "cl::sycl::sqrt"},
    {"__dsub_rd", "-"},
    {"__dsub_rn", "-"},
    {"__dsub_ru", "-"},
    {"__dsub_rz", "-"},
    {"__fma_rd", "cl::sycl::fma"},
    {"__fma_rn", "cl::sycl::fma"},
    {"__fma_ru", "cl::sycl::fma"},
    {"__fma_rz", "cl::sycl::fma"},
};

const MapNames::MapTy MathFunctionsRule::TypecastFunctionNamesMap{
    //  Half Precision Conversion And Data Movement
    {"__float22half2_rn", ""},
    {"__float2half", ""},
    {"__float2half2_rn", ""},
    {"__float2half_rd", ""},
    {"__float2half_rn", ""},
    {"__float2half_ru", ""},
    {"__float2half_rz", ""},
    {"__floats2half2_rn", ""},
    {"__half22float2", ""},
    {"__half2float", ""},
    {"__half2half2", ""},
    {"__half2int_rd", ""},
    {"__half2int_rn", ""},
    {"__half2int_ru", ""},
    {"__half2int_rz", ""},
    {"__half2ll_rd", ""},
    {"__half2ll_rn", ""},
    {"__half2ll_ru", ""},
    {"__half2ll_rz", ""},
    {"__half2short_rd", ""},
    {"__half2short_rn", ""},
    {"__half2short_ru", ""},
    {"__half2short_rz", ""},
    {"__half2uint_rd", ""},
    {"__half2uint_rn", ""},
    {"__half2uint_ru", ""},
    {"__half2uint_rz", ""},
    {"__half2ull_rd", ""},
    {"__half2ull_rn", ""},
    {"__half2ull_ru", ""},
    {"__half2ull_rz", ""},
    {"__half2ushort_rd", ""},
    {"__half2ushort_rn", ""},
    {"__half2ushort_ru", ""},
    {"__half2ushort_rz", ""},
    {"__half_as_short", "syclct::bit_cast<cl::sycl::half, short>"},
    {"__half_as_ushort", "syclct::bit_cast<cl::sycl::half, unsigned short>"},
    {"__halves2half2", ""},
    {"__high2float", ""},
    {"__high2half", ""},
    {"__high2half2", ""},
    {"__highs2half2", ""},
    {"__int2half_rd", ""},
    {"__int2half_rn", ""},
    {"__int2half_ru", ""},
    {"__int2half_rz", ""},
    {"__ll2half_rd", ""},
    {"__ll2half_rn", ""},
    {"__ll2half_ru", ""},
    {"__ll2half_rz", ""},
    {"__low2float", ""},
    {"__low2half", ""},
    {"__low2half2", ""},
    {"__lowhigh2highlow", ""},
    {"__lows2half2", ""},
    {"__shfl_down_sync", StringLiteralUnsupported},
    {"__shfl_sync", StringLiteralUnsupported},
    {"__shfl_up_sync", StringLiteralUnsupported},
    {"__shfl_xor_sync", StringLiteralUnsupported},
    {"__short2half_rd", ""},
    {"__short2half_rn", ""},
    {"__short2half_ru", ""},
    {"__short2half_rz", ""},
    {"__short_as_half", "syclct::bit_cast<short, cl::sycl::half>"},
    {"__uint2half_rd", ""},
    {"__uint2half_rn", ""},
    {"__uint2half_ru", ""},
    {"__uint2half_rz", ""},
    {"__ull2half_rd", ""},
    {"__ull2half_rn", ""},
    {"__ull2half_ru", ""},
    {"__ull2half_rz", ""},
    {"__ushort2half_rd", ""},
    {"__ushort2half_rn", ""},
    {"__ushort2half_ru", ""},
    {"__ushort2half_rz", ""},
    {"__ushort_as_half", "syclct::bit_cast<unsigned short, cl::sycl::half>"},

    // Type Casting Intrinsics
    {"__double2float_rd", ""},
    {"__double2float_rn", ""},
    {"__double2float_ru", ""},
    {"__double2float_rz", ""},
    {"__double2hiint", StringLiteralUnsupported},  // TODO
    {"__double2int_rd", ""},
    {"__double2int_rn", ""},
    {"__double2int_ru", ""},
    {"__double2int_rz", ""},
    {"__double2ll_rd", ""},
    {"__double2ll_rn", ""},
    {"__double2ll_ru", ""},
    {"__double2ll_rz", ""},
    {"__double2loint", StringLiteralUnsupported}, // TODO
    {"__double2uint_rd", ""},
    {"__double2uint_rn", ""},
    {"__double2uint_ru", ""},
    {"__double2uint_rz", ""},
    {"__double2ull_rd", ""},
    {"__double2ull_rn", ""},
    {"__double2ull_ru", ""},
    {"__double2ull_rz", ""},
    {"__double_as_longlong", "syclct::bit_cast<double, long long>"},
    {"__float2int_rd", ""},
    {"__float2int_rn", ""},
    {"__float2int_ru", ""},
    {"__float2int_rz", ""},
    {"__float2ll_rd", ""},
    {"__float2ll_rn", ""},
    {"__float2ll_ru", ""},
    {"__float2ll_rz", ""},
    {"__float2uint_rd", ""},
    {"__float2uint_rn", ""},
    {"__float2uint_ru", ""},
    {"__float2uint_rz", ""},
    {"__float2ull_rd", ""},
    {"__float2ull_rn", ""},
    {"__float2ull_ru", ""},
    {"__float2ull_rz", ""},
    {"__float_as_int", "syclct::bit_cast<float, int>"},
    {"__float_as_uint", "syclct::bit_cast<float, unsigned int>"},
    {"__hiloint2double", StringLiteralUnsupported}, // TODO
    {"__int2double_rn", ""},
    {"__int2float_rd", ""},
    {"__int2float_rn", ""},
    {"__int2float_ru", ""},
    {"__int2float_rz", ""},
    {"__int_as_float", "syclct::bit_cast<int, float>"},
    {"__ll2double_rd", ""},
    {"__ll2double_rn", ""},
    {"__ll2double_ru", ""},
    {"__ll2double_rz", ""},
    {"__ll2float_rd", ""},
    {"__ll2float_rn", ""},
    {"__ll2float_ru", ""},
    {"__ll2float_rz", ""},
    {"__longlong_as_double", "syclct::bit_cast<long long, double>"},
    {"__uint2double_rn", ""},
    {"__uint2float_rd", ""},
    {"__uint2float_rn", ""},
    {"__uint2float_ru", ""},
    {"__uint2float_rz", ""},
    {"__uint_as_float", "syclct::bit_cast<unsigned int, float>"},
    {"__ull2double_rd", ""},
    {"__ull2double_rn", ""},
    {"__ull2double_ru", ""},
    {"__ull2double_rz", ""},
    {"__ull2float_rd", ""},
    {"__ull2float_rn", ""},
    {"__ull2float_ru", ""},
    {"__ull2float_rz", ""},
};

// Math function names migration.
const MapNames::MapTy MathFunctionsRule::IntegerFunctionNamesMap{
    // Cuda's max to sycl's max. all have integer types and double types.
    // See "4.13.4 Integer functions" and "4.13.5 Common functions"

    // Integer Intrinsics
    {"__brev", StringLiteralUnsupported},
    {"__brevll", StringLiteralUnsupported},
    {"__byte_perm", StringLiteralUnsupported},
    {"__clz", "cl::sycl::clz"},
    {"__clzll", "cl::sycl::clz"},
    {"__ffs", StringLiteralUnsupported},
    {"__ffsll", StringLiteralUnsupported},
    {"__funnelshift_l", StringLiteralUnsupported},
    {"__funnelshift_lc", StringLiteralUnsupported},
    {"__funnelshift_r", StringLiteralUnsupported},
    {"__funnelshift_rc", StringLiteralUnsupported},
    {"__hadd", "cl::sycl::hadd"},
    {"__mul24", "cl::sycl::mul24"},
    {"__mul64hi", StringLiteralUnsupported},
    {"__mulhi", "cl::sycl::mul_hi"},
    {"__popc", "cl::sycl::popcount"},
    {"__popcll", "cl::sycl::popcount"},
    {"__rhadd", StringLiteralUnsupported},
    {"__sad", StringLiteralUnsupported},
    {"__uhadd", StringLiteralUnsupported},
    {"__umul24", StringLiteralUnsupported},
    {"__umul64hi", StringLiteralUnsupported},
    {"__umulhi", StringLiteralUnsupported},
    {"__urhadd", StringLiteralUnsupported},
    {"__usad", StringLiteralUnsupported},

    // Not in CUDA API but in SYCL API
    // {"acospi", "cl::sycl::acospi"},
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

// Do migration even if they are not part of CUDA math API
const MapNames::MapTy MathFunctionsRule::ExceptionalFunctionNamesMap{
    // <SYCL/sycl_math_builtins_common.h>
    {"abs", "cl::sycl::abs"},
    {"max", "cl::sycl::max"},
    {"min", "cl::sycl::min"},

    {"exp", "cl::sycl::exp"},
    {"fabs", "cl::sycl::fabs"},
    {"sqrt", "cl::sycl::sqrt"},
};


// cudaFuncAttributes names migration
const MapNames::MapTy KernelFunctionInfoRule::AttributesNamesMap{
    {"maxThreadsPerBlock", "max_work_group_size"},
};

std::map<std::string, bool> TranslationStatistics::TranslationTable{
#define ENTRY(APINAME, VALUE, TARGET, COMMENT) {#APINAME, VALUE},
#include "APINames.inc"
#include "APINames_thrust.inc"
#include "APINames_cuBLAS.inc"
#include "APINames_nvJPEG.inc"
#include "APINames_cuFFT.inc"
#include "APINames_cuGRAPH.inc"
#include "APINames_cuRAND.inc"
#include "APINames_cuPARSE.inc"
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
