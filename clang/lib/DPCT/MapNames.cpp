//===--------------- MapNames.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNames.h"
#include "ASTTraversal.h"
#include "CallExprRewriter.h"
#include "DNNAPIMigration.h"
#include "SaveNewFiles.h"
#include <map>

using namespace clang;
using namespace clang::dpct;
// Not use sycl:: namespace explicitly
// KeepNamespace = false/true --> ""/"sycl::"
std::vector<std::string> MapNames::ClNamespace;
// Not use dpct:: namespace explicitly
// KeepNamespace = false/true --> ""/"dpct::"
std::vector<std::string> MapNames::DpctNamespace(2);
std::string MapNames::getClNamespace(bool KeepNamespace, bool IsMathFunc) {
  return ClNamespace[(KeepNamespace << 1) + IsMathFunc];
}
std::string MapNames::getDpctNamespace(bool KeepNamespace) {
  return DpctNamespace[KeepNamespace];
}
std::string MapNames::getExpNamespace(bool KeepNamespace) {
  return getClNamespace(KeepNamespace, false) + "ext::oneapi::experimental::";
}

std::unordered_set<std::string> MapNames::SYCLcompatUnsupportTypes;
std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
    MapNames::TypeNamesMap;
std::unordered_map<std::string, std::shared_ptr<ClassFieldRule>>
    MapNames::ClassFieldMap;
MapNames::MapTy MapNames::RandomEngineTypeMap;
MapNames::MapTy MapNames::RandomOrderingTypeMap;
MapNames::MapTy MapNames::DeviceRandomGeneratorTypeMap;
std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
    MapNames::CuDNNTypeNamesMap;
std::unordered_map<std::string, std::shared_ptr<EnumNameRule>>
    EnumConstantRule::EnumNamesMap;
MapNames::MapTy CuDNNTypeRule::CuDNNEnumNamesMap;
std::map<std::string /*Original API*/, HelperFeatureEnum>
    CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap;
MapNames::ThrustMapTy MapNames::ThrustFuncNamesMap;
std::map<std::string /*Original API*/, HelperFeatureEnum>
    MapNames::ThrustFuncNamesHelperFeaturesMap;
std::unordered_map<std::string, std::string> MapNames::AtomicFuncNamesMap;
MapNames::MapTy MapNames::ITFName;
std::map<std::string, MapNames::BLASGemmExTypeInfo>
    MapNames::BLASTGemmExTypeInfoMap;
std::unordered_map<std::string, std::pair<std::string, std::string>>
    MapNames::MathTypeCastingMap;
MapNames::MapTy MapNames::BLASAPIWithRewriter;
std::unordered_set<std::string> MapNames::SOLVERAPIWithRewriter;
MapNames::MapTy MapNames::BLASEnumsMap;
MapNames::MapTy MapNames::SPBLASEnumsMap;
MapNames::MapTy MapNames::CUBEnumsMap;

namespace {
auto EnumBit = [](auto EnumValue) {
  return 1 << static_cast<unsigned>(EnumValue);
};
void checkExplicitNamespaceBits(unsigned ExplicitNamespaceBits) {
  static constexpr unsigned BitNone = EnumBit(ExplicitNamespace::EN_None);
  static constexpr unsigned BitsExclusive =
      EnumBit(ExplicitNamespace::EN_SYCL) |
      EnumBit(ExplicitNamespace::EN_SYCL_Math);

  while (1) {
    if ((ExplicitNamespaceBits & BitNone) && (ExplicitNamespaceBits ^ BitNone))
      break;

    if ((ExplicitNamespaceBits & BitsExclusive) == BitsExclusive)
      break;

    if (DpctGlobalInfo::useSYCLCompat()) {
      if (ExplicitNamespaceBits & EnumBit(ExplicitNamespace::EN_DPCT))
        break;
    } else if (ExplicitNamespaceBits &
               EnumBit(ExplicitNamespace::EN_SYCLCompat)) {
      break;
    }
    return;
  }
  ShowStatus(MigrationErrorInvalidExplicitNamespace);
  dpctExit(MigrationErrorInvalidExplicitNamespace);
}

const std::string &getDpctNamespaceName() {
  const static std::string Name = [](bool Use) {
    if (Use)
      return "syclcompat";
    else
      return "dpct";
  }(DpctGlobalInfo::useSYCLCompat());
  return Name;
}

std::string LibraryHelperNamespace("dpct::");
bool ExplicitHelperNamespace = true;
bool ExplicitSYCLNamespace = true;

} // namespace

void DpctGlobalInfo::printUsingNamespace(llvm::raw_ostream &OS) {
  auto printUsing = [](llvm::raw_ostream &OS, const std::string &Name) {
    OS << "using namespace " << Name << ";" << getNL();
  };
  if (!ExplicitHelperNamespace)
    printUsing(OS, getDpctNamespaceName());
  if (!ExplicitSYCLNamespace)
    printUsing(OS, "sycl");
}

const std::string &MapNames::getLibraryHelperNamespace() {
  return LibraryHelperNamespace;
}

const std::string &MapNames::getCheckErrorMacroName() {
  static const std::string Name = DpctGlobalInfo::useSYCLCompat()
                                      ? "SYCLCOMPAT_CHECK_ERROR"
                                      : "DPCT_CHECK_ERROR";
  return Name;
}

void MapNames::setExplicitNamespaceMap(
    const std::set<ExplicitNamespace> &ExplicitNamespaces) {

  unsigned ExplicitNamespaceBits = 0;
  for (auto Val : ExplicitNamespaces)
    ExplicitNamespaceBits |= EnumBit(Val);

  checkExplicitNamespaceBits(ExplicitNamespaceBits);
  ExplicitHelperNamespace =
      ExplicitNamespaceBits & (EnumBit(ExplicitNamespace::EN_SYCLCompat) |
                               EnumBit(ExplicitNamespace::EN_DPCT));
  ExplicitSYCLNamespace =
      ExplicitNamespaceBits & EnumBit(ExplicitNamespace::EN_SYCL);

  if (ExplicitHelperNamespace) {
    // always use dpct::/syclcompat:: explicitly
    DpctNamespace[0] = DpctNamespace[1] = getDpctNamespaceName() + "::";
  } else {
    LibraryHelperNamespace.clear();
    DpctNamespace[1] = getDpctNamespaceName() + "::";
  }

  ClNamespace.reserve(4);
  if (ExplicitNamespaceBits & EnumBit(ExplicitNamespace::EN_SYCL_Math)) {
    // Use sycl:: namespce for SYCL math functions
    ClNamespace.push_back("");
  } else if (!ExplicitSYCLNamespace) {
    // Use sycl:: namespace explicitly
    ClNamespace.assign(2, "");
  }
  ClNamespace.resize(4, "sycl::");

  MathTypeCastingMap = {
      {"__half_as_short",
       {"short", MapNames::getClNamespace(false, true) + "half"}},
      {"__half_as_ushort",
       {"unsigned short", MapNames::getClNamespace(false, true) + "half"}},
      {"__short_as_half",
       {MapNames::getClNamespace(false, true) + "half", "short"}},
      {"__ushort_as_half",
       {MapNames::getClNamespace(false, true) + "half", "unsigned short"}},
      {"__double_as_longlong", {"long long", "double"}},
      {"__float_as_int", {"int", "float"}},
      {"__float_as_uint", {"unsigned int", "float"}},
      {"__int_as_float", {"float", "int"}},
      {"__longlong_as_double", {"double", "long long"}},
      {"__uint_as_float", {"float", "unsigned int"}}};
  MacroRuleMap = {
      {"__forceinline__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__forceinline__",
                          DpctGlobalInfo::useSYCLCompat()
                              ? "__syclcompat_inline__"
                              : "__dpct_inline__",
                          HelperFeatureEnum::device_ext)},
      {"__align__", MacroMigrationRule("dpct_build_in_macro_rule",
                                       RulePriority::Fallback, "__align__",
                                       DpctGlobalInfo::useSYCLCompat()
                                           ? "__syclcompat_align__"
                                           : "__dpct_align__",
                                       HelperFeatureEnum::device_ext)},
      {"__CUDA_ALIGN__",
       MacroMigrationRule(
           "dpct_build_in_macro_rule", RulePriority::Fallback, "__CUDA_ALIGN__",
           DpctGlobalInfo::useSYCLCompat() ? "__syclcompat_align__"
                                           : "__dpct_align__",
           HelperFeatureEnum::device_ext)},
      {"__noinline__",
       MacroMigrationRule(
           "dpct_build_in_macro_rule", RulePriority::Fallback, "__noinline__",
           DpctGlobalInfo::useSYCLCompat() ? "__syclcompat_noinline__"
                                           : "__dpct_noinline__",
           HelperFeatureEnum::device_ext)},
      {"cudaMemAttachGlobal",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaMemAttachGlobal", "0")},
      {"cudaStreamDefault",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaStreamDefault", "0")},

      {"CU_LAUNCH_PARAM_BUFFER_SIZE",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CU_LAUNCH_PARAM_BUFFER_SIZE", "((void *) 2)",
                          HelperFeatureEnum::device_ext)},
      {"CU_LAUNCH_PARAM_BUFFER_POINTER",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CU_LAUNCH_PARAM_BUFFER_POINTER", "((void *) 1)",
                          HelperFeatureEnum::device_ext)},
      {"CU_LAUNCH_PARAM_END",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CU_LAUNCH_PARAM_END", "((void *) 0)",
                          HelperFeatureEnum::device_ext)},
      {"CUDART_PI_F",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUDART_PI_F", "3.141592654F")},
      {"CUB_MAX",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUB_MAX", "std::max")},
      {"CUB_MIN",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUB_MIN", "std::min")},
      {"CUB_RUNTIME_FUNCTION",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUB_RUNTIME_FUNCTION", "")},
      {"cudaStreamAttrValue",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaStreamAttrValue", "int")},
      {"NCCL_VERSION_CODE",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "NCCL_VERSION_CODE", "DPCT_COMPAT_CCL_VERSION")},
      {"__CUDA_ARCH__",
       MacroMigrationRule(
           "dpct_build_in_macro_rule", RulePriority::Fallback, "__CUDA_ARCH__",
           DpctGlobalInfo::useSYCLCompat() ? "SYCLCOMPAT_COMPATIBILITY_TEMP"
                                           : "DPCT_COMPATIBILITY_TEMP",
           clang::dpct::HelperFeatureEnum::device_ext)},
      {"__NVCC__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__NVCC__", "SYCL_LANGUAGE_VERSION")},
      {"__CUDACC__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDACC__", "SYCL_LANGUAGE_VERSION")},
      {"__DRIVER_TYPES_H__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__DRIVER_TYPES_H__",
                          DpctGlobalInfo::useSYCLCompat()
                              ? "SYCLCOMPAT_COMPATIBILITY_TEMP"
                              : "__DPCT_HPP__")},
      {"__CUDA_RUNTIME_H__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDA_RUNTIME_H__",
                          DpctGlobalInfo::useSYCLCompat()
                              ? "SYCLCOMPAT_COMPATIBILITY_TEMP"
                              : "__DPCT_HPP__")},
      {"CUDART_VERSION",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUDART_VERSION", "DPCT_COMPAT_RT_VERSION")},
      {"__CUDART_API_VERSION",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDART_API_VERSION", "DPCT_COMPAT_RT_VERSION")},
      {"CUDA_VERSION",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUDA_VERSION", "DPCT_COMPAT_RT_VERSION")},
      {"__CUDACC_VER_MAJOR__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDACC_VER_MAJOR__",
                          "DPCT_COMPAT_RT_MAJOR_VERSION")},
      {"__CUDACC_VER_MINOR__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDACC_VER_MINOR__",
                          "DPCT_COMPAT_RT_MINOR_VERSION")},
      {"CUBLAS_V2_H_",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUBLAS_V2_H_", "MKL_SYCL_HPP")},
      {"__CUDA__",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "__CUDA__", "SYCL_LANGUAGE_VERSION")},
      {"CUFFT_FORWARD",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUFFT_FORWARD", "-1")},
      {"CUFFT_INVERSE",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CUFFT_INVERSE", "1")},
      {"cudaEventDefault",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaEventDefault", "0")},
      //...
  };
  // Type names mapping.
  TypeNamesMap = {
      {"cudaDeviceProp",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "device_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudaError_t",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "err0",
                                      HelperFeatureEnum::device_ext)},
      {"cudaError",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "err0",
                                      HelperFeatureEnum::device_ext)},
      {"CUjit_option", std::make_shared<TypeNameRule>("int")},
      {"CUresult", std::make_shared<TypeNameRule>("int")},
      {"CUcontext", std::make_shared<TypeNameRule>("int")},
      {"CUmodule",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "kernel_library",
                                      HelperFeatureEnum::device_ext)},
      {"CUfunction",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "kernel_function",
                                      HelperFeatureEnum::device_ext)},
      {"CUpointer_attribute",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type")},
      {"cudaPointerAttributes",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "pointer_attributes",
                                      HelperFeatureEnum::device_ext)},
      {"dim3", std::make_shared<TypeNameRule>(getDpctNamespace() + "dim3")},
      {"int2", std::make_shared<TypeNameRule>(getClNamespace() + "int2")},
      {"double2", std::make_shared<TypeNameRule>(getClNamespace() + "double2")},
      {"__half", std::make_shared<TypeNameRule>(getClNamespace() + "half")},
      {"__half2", std::make_shared<TypeNameRule>(getClNamespace() + "half2")},
      {"half", std::make_shared<TypeNameRule>(getClNamespace() + "half")},
      {"half2", std::make_shared<TypeNameRule>(getClNamespace() + "half2")},
      {"cudaEvent_t",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "event_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"CUevent",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "event_ptr")},
      {"CUevent_st",
       std::make_shared<TypeNameRule>(getClNamespace() + "event")},
      {"CUfunc_cache", std::make_shared<TypeNameRule>("int")},
      {"cudaStream_t",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "queue_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"CUstream",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "queue_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"CUstream_st",
       std::make_shared<TypeNameRule>(getClNamespace() + "queue")},
      {"char1", std::make_shared<TypeNameRule>("int8_t")},
      {"char2", std::make_shared<TypeNameRule>(getClNamespace() + "char2")},
      {"char3", std::make_shared<TypeNameRule>(getClNamespace() + "char3")},
      {"char4", std::make_shared<TypeNameRule>(getClNamespace() + "char4")},
      {"double1", std::make_shared<TypeNameRule>("double")},
      {"double2", std::make_shared<TypeNameRule>(getClNamespace() + "double2")},
      {"double3", std::make_shared<TypeNameRule>(getClNamespace() + "double3")},
      {"double4", std::make_shared<TypeNameRule>(getClNamespace() + "double4")},
      {"float1", std::make_shared<TypeNameRule>("float")},
      {"float2", std::make_shared<TypeNameRule>(getClNamespace() + "float2")},
      {"float3", std::make_shared<TypeNameRule>(getClNamespace() + "float3")},
      {"float4", std::make_shared<TypeNameRule>(getClNamespace() + "float4")},
      {"int1", std::make_shared<TypeNameRule>("int32_t")},
      {"int2", std::make_shared<TypeNameRule>(getClNamespace() + "int2")},
      {"int3", std::make_shared<TypeNameRule>(getClNamespace() + "int3")},
      {"int4", std::make_shared<TypeNameRule>(getClNamespace() + "int4")},
      {"long1", std::make_shared<TypeNameRule>("int64_t")},
      {"long2", std::make_shared<TypeNameRule>(getClNamespace() + "long2")},
      {"long3", std::make_shared<TypeNameRule>(getClNamespace() + "long3")},
      {"long4", std::make_shared<TypeNameRule>(getClNamespace() + "long4")},
      {"longlong1", std::make_shared<TypeNameRule>("int64_t")},
      {"longlong2", std::make_shared<TypeNameRule>(getClNamespace() + "long2")},
      {"longlong3", std::make_shared<TypeNameRule>(getClNamespace() + "long3")},
      {"longlong4", std::make_shared<TypeNameRule>(getClNamespace() + "long4")},
      {"short1", std::make_shared<TypeNameRule>("int16_t")},
      {"short2", std::make_shared<TypeNameRule>(getClNamespace() + "short2")},
      {"short3", std::make_shared<TypeNameRule>(getClNamespace() + "short3")},
      {"short4", std::make_shared<TypeNameRule>(getClNamespace() + "short4")},
      {"uchar1", std::make_shared<TypeNameRule>("uint8_t")},
      {"uchar2", std::make_shared<TypeNameRule>(getClNamespace() + "uchar2")},
      {"uchar3", std::make_shared<TypeNameRule>(getClNamespace() + "uchar3")},
      {"uchar4", std::make_shared<TypeNameRule>(getClNamespace() + "uchar4")},
      {"uint1", std::make_shared<TypeNameRule>("uint32_t")},
      {"uint2", std::make_shared<TypeNameRule>(getClNamespace() + "uint2")},
      {"uint3", std::make_shared<TypeNameRule>(getClNamespace() + "uint3")},
      {"uint4", std::make_shared<TypeNameRule>(getClNamespace() + "uint4")},
      {"ulong1", std::make_shared<TypeNameRule>("uint64_t")},
      {"ulong2", std::make_shared<TypeNameRule>(getClNamespace() + "ulong2")},
      {"ulong3", std::make_shared<TypeNameRule>(getClNamespace() + "ulong3")},
      {"ulong4", std::make_shared<TypeNameRule>(getClNamespace() + "ulong4")},
      {"ulonglong1", std::make_shared<TypeNameRule>("uint64_t")},
      {"ulonglong2",
       std::make_shared<TypeNameRule>(getClNamespace() + "ulong2")},
      {"ulonglong3",
       std::make_shared<TypeNameRule>(getClNamespace() + "ulong3")},
      {"ulonglong4",
       std::make_shared<TypeNameRule>(getClNamespace() + "ulong4")},
      {"ushort1", std::make_shared<TypeNameRule>("uint16_t")},
      {"ushort2", std::make_shared<TypeNameRule>(getClNamespace() + "ushort2")},
      {"ushort3", std::make_shared<TypeNameRule>(getClNamespace() + "ushort3")},
      {"ushort4", std::make_shared<TypeNameRule>(getClNamespace() + "ushort4")},
      {"cublasHandle_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "blas::descriptor_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"cublasStatus_t", std::make_shared<TypeNameRule>("int")},
      {"cublasStatus", std::make_shared<TypeNameRule>("int")},
      {"cublasGemmAlgo_t", std::make_shared<TypeNameRule>("int")},
      {"cudaDataType_t", std::make_shared<TypeNameRule>(
                             getLibraryHelperNamespace() + "library_data_t",
                             HelperFeatureEnum::device_ext)},
      {"cudaDataType", std::make_shared<TypeNameRule>(
                           getLibraryHelperNamespace() + "library_data_t",
                           HelperFeatureEnum::device_ext)},
      {"cublasDataType_t", std::make_shared<TypeNameRule>(
                               getLibraryHelperNamespace() + "library_data_t",
                               HelperFeatureEnum::device_ext)},
      {"cublasComputeType_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "compute_type")},
      {"cuComplex",
       std::make_shared<TypeNameRule>(getClNamespace() + "float2")},
      {"cuFloatComplex",
       std::make_shared<TypeNameRule>(getClNamespace() + "float2")},
      {"cuDoubleComplex",
       std::make_shared<TypeNameRule>(getClNamespace() + "double2")},
      {"cublasFillMode_t", std::make_shared<TypeNameRule>("oneapi::mkl::uplo")},
      {"cublasDiagType_t", std::make_shared<TypeNameRule>("oneapi::mkl::diag")},
      {"cublasSideMode_t", std::make_shared<TypeNameRule>("oneapi::mkl::side")},
      {"cublasOperation_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::transpose")},
      {"cublasPointerMode_t", std::make_shared<TypeNameRule>("int")},
      {"cublasAtomicsMode_t", std::make_shared<TypeNameRule>("int")},
      {"cublasMath_t", std::make_shared<TypeNameRule>(
                           getLibraryHelperNamespace() + "blas::math_mode")},
      {"cusparsePointerMode_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseFillMode_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::uplo")},
      {"cusparseDiagType_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::diag")},
      {"cusparseIndexBase_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::index_base")},
      {"cusparseMatrixType_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "sparse::matrix_info::matrix_type",
                                      HelperFeatureEnum::device_ext)},
      {"cusparseOperation_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::transpose")},
      {"cusparseAlgMode_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSolveAnalysisInfo_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                          getLibraryHelperNamespace() +
                                          "sparse::optimize_info>",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::device_ptr",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() + "device_pointer",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::device_reference",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() + "device_reference",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::device_vector",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() + "device_vector",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::device_malloc_allocator",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                          "deprecated::usm_device_allocator",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::maximum",
       std::make_shared<TypeNameRule>("oneapi::dpl::maximum")},
      {"thrust::multiplies", std::make_shared<TypeNameRule>("std::multiplies")},
      {"thrust::plus", std::make_shared<TypeNameRule>("std::plus")},
      {"thrust::seq",
       std::make_shared<TypeNameRule>("oneapi::dpl::execution::seq")},
      {"thrust::device",
       std::make_shared<TypeNameRule>("oneapi::dpl::execution::dpcpp_default")},
      {"thrust::host",
       std::make_shared<TypeNameRule>("oneapi::dpl::execution::seq")},
      {"thrust::minus", std::make_shared<TypeNameRule>("std::minus")},
      {"thrust::nullopt", std::make_shared<TypeNameRule>("std::nullopt")},
      {"thrust::greater", std::make_shared<TypeNameRule>("std::greater")},
      {"thrust::equal_to",
       std::make_shared<TypeNameRule>("oneapi::dpl::equal_to")},
      {"thrust::less", std::make_shared<TypeNameRule>("oneapi::dpl::less")},
      {"thrust::negate", std::make_shared<TypeNameRule>("std::negate")},
      {"thrust::logical_or", std::make_shared<TypeNameRule>("std::logical_or")},
      {"thrust::divides", std::make_shared<TypeNameRule>("std::divides")},
      {"thrust::tuple", std::make_shared<TypeNameRule>("std::tuple")},
      {"thrust::pair", std::make_shared<TypeNameRule>("std::pair")},
      {"thrust::host_vector", std::make_shared<TypeNameRule>("std::vector")},
      {"thrust::complex", std::make_shared<TypeNameRule>("std::complex")},
      {"thrust::counting_iterator",
       std::make_shared<TypeNameRule>("oneapi::dpl::counting_iterator")},
      {"thrust::permutation_iterator",
       std::make_shared<TypeNameRule>("oneapi::dpl::permutation_iterator")},
      {"thrust::transform_iterator",
       std::make_shared<TypeNameRule>("oneapi::dpl::transform_iterator")},
      {"thrust::iterator_difference",
       std::make_shared<TypeNameRule>("std::iterator_traits")},
      {"thrust::tuple_element",
       std::make_shared<TypeNameRule>("std::tuple_element")},
      {"thrust::tuple_size", std::make_shared<TypeNameRule>("std::tuple_size")},
      {"thrust::swap", std::make_shared<TypeNameRule>("std::swap")},
      {"thrust::zip_iterator",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "zip_iterator",
                                      HelperFeatureEnum::device_ext)},
      {"cusolverDnHandle_t",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "queue_ptr")},
      {"cusolverEigType_t", std::make_shared<TypeNameRule>("int64_t")},
      {"cusolverEigMode_t", std::make_shared<TypeNameRule>("oneapi::mkl::job")},
      {"cusolverStatus_t", std::make_shared<TypeNameRule>("int")},
      {"cusolverDnParams_t", std::make_shared<TypeNameRule>("int")},
      {"gesvdjInfo_t", std::make_shared<TypeNameRule>("int")},
      {"syevjInfo_t", std::make_shared<TypeNameRule>("int")},
      {"cudaChannelFormatDesc",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_channel",
                                      HelperFeatureEnum::device_ext)},
      {"cudaChannelFormatKind",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                          "image_channel_data_type",
                                      HelperFeatureEnum::device_ext)},
      {"cudaArray",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper"
               : getDpctNamespace() + "image_matrix",
           HelperFeatureEnum::device_ext)},
      {"cudaArray_t",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper_ptr"
               : getDpctNamespace() + "image_matrix_p",
           HelperFeatureEnum::device_ext)},
      {"cudaMipmappedArray",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper"
               : "cudaMipmappedArray")},
      {"cudaMipmappedArray_t",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper_ptr"
               : "cudaMipmappedArray_t")},
      {"cudaTextureDesc",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "sampling_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudaResourceDesc",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_data",
                                      HelperFeatureEnum::device_ext)},
      {"cudaTextureObject_t",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::sampled_image_handle"
               : getDpctNamespace() + "image_wrapper_base_p",
           HelperFeatureEnum::device_ext)},
      {"textureReference",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_wrapper_base",
                                      HelperFeatureEnum::device_ext)},
      {"cudaTextureAddressMode",
       std::make_shared<TypeNameRule>(getClNamespace() + "addressing_mode")},
      {"cudaTextureFilterMode",
       std::make_shared<TypeNameRule>(getClNamespace() + "filtering_mode")},
      {"curandGenerator_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "rng::host_rng_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"curandRngType_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "rng::random_engine_type",
                                      HelperFeatureEnum::device_ext)},
      {"curandRngType",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "rng::random_engine_type",
                                      HelperFeatureEnum::device_ext)},
      {"curandStatus_t", std::make_shared<TypeNameRule>("int")},
      {"curandStatus", std::make_shared<TypeNameRule>("int")},
      {"curandOrdering_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "rng::random_mode")},
      {"cusparseStatus_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseMatDescr_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                          getLibraryHelperNamespace() +
                                          "sparse::matrix_info>",
                                      HelperFeatureEnum::device_ext)},
      {"cusparseHandle_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "sparse::descriptor_ptr")},
      {"cudaMemoryAdvise", std::make_shared<TypeNameRule>("int")},
      {"cudaStreamCaptureStatus",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtGraph()
               ? getClNamespace() + "ext::oneapi::experimental::queue_state"
               : "cudaStreamCaptureStatus")},
      {"CUmem_advise", std::make_shared<TypeNameRule>("int")},
      {"CUmemorytype",
       std::make_shared<TypeNameRule>(getClNamespace() + "usm::alloc")},
      {"CUmemorytype_enum",
       std::make_shared<TypeNameRule>(getClNamespace() + "usm::alloc")},
      {"cudaPos", std::make_shared<TypeNameRule>(getClNamespace() + "id<3>")},
      {"cudaExtent",
       std::make_shared<TypeNameRule>(getClNamespace() + "range<3>")},
      {"cudaPitchedPtr",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "pitched_data",
                                      HelperFeatureEnum::device_ext)},
      {"cudaMemcpyKind",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_direction")},
      {"CUDA_ARRAY3D_DESCRIPTOR",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::image_descriptor"
               : getDpctNamespace() + "image_matrix_desc")},
      {"CUDA_ARRAY_DESCRIPTOR",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::image_descriptor"
               : getDpctNamespace() + "image_matrix_desc")},
      {"cudaMemcpy3DParms",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_parameter")},
      {"CUDA_MEMCPY3D",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_parameter")},
      {"cudaMemcpy3DPeerParms",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_parameter")},
      {"CUDA_MEMCPY3D_PEER",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_parameter")},
      {"CUDA_MEMCPY2D",
       std::make_shared<TypeNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "memcpy_parameter")},
      {"cudaComputeMode", std::make_shared<TypeNameRule>("int")},
      {"cudaSharedMemConfig", std::make_shared<TypeNameRule>("int")},
      {"cufftReal", std::make_shared<TypeNameRule>("float")},
      {"cufftDoubleReal", std::make_shared<TypeNameRule>("double")},
      {"cufftComplex",
       std::make_shared<TypeNameRule>(getClNamespace() + "float2")},
      {"cufftDoubleComplex",
       std::make_shared<TypeNameRule>(getClNamespace() + "double2")},
      {"cufftResult_t", std::make_shared<TypeNameRule>("int")},
      {"cufftResult", std::make_shared<TypeNameRule>("int")},
      {"cufftType_t", std::make_shared<TypeNameRule>(
                          getLibraryHelperNamespace() + "fft::fft_type",
                          HelperFeatureEnum::device_ext)},
      {"cufftType", std::make_shared<TypeNameRule>(
                        getLibraryHelperNamespace() + "fft::fft_type",
                        HelperFeatureEnum::device_ext)},
      {"cufftHandle", std::make_shared<TypeNameRule>(
                          getLibraryHelperNamespace() + "fft::fft_engine_ptr",
                          HelperFeatureEnum::device_ext)},
      {"CUdevice", std::make_shared<TypeNameRule>("int")},
      {"CUarray_st",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper"
               : getDpctNamespace() + "image_matrix",
           HelperFeatureEnum::device_ext)},
      {"CUarray",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getDpctNamespace() + "experimental::image_mem_wrapper_ptr"
               : getDpctNamespace() + "image_matrix_p",
           HelperFeatureEnum::device_ext)},
      {"CUarray_format",
       std::make_shared<TypeNameRule>(getClNamespace() + "image_channel_type")},
      {"CUarray_format_enum",
       std::make_shared<TypeNameRule>(getClNamespace() + "image_channel_type")},
      {"CUtexObject", std::make_shared<TypeNameRule>(
                          getDpctNamespace() + "image_wrapper_base_p",
                          HelperFeatureEnum::device_ext)},
      {"CUDA_RESOURCE_DESC",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_data",
                                      HelperFeatureEnum::device_ext)},
      {"CUDA_TEXTURE_DESC",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "sampling_info",
                                      HelperFeatureEnum::device_ext)},
      {"CUaddress_mode",
       std::make_shared<TypeNameRule>(getClNamespace() + "addressing_mode")},
      {"CUaddress_mode_enum",
       std::make_shared<TypeNameRule>(getClNamespace() + "addressing_mode")},
      {"CUfilter_mode",
       std::make_shared<TypeNameRule>(getClNamespace() + "filtering_mode")},
      {"CUfilter_mode_enum",
       std::make_shared<TypeNameRule>(getClNamespace() + "filtering_mode")},
      {"CUdeviceptr",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "device_ptr",
                                      HelperFeatureEnum::device_ext)},
      {"CUresourcetype_enum",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_data_type",
                                      HelperFeatureEnum::device_ext)},
      {"CUresourcetype",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_data_type",
                                      HelperFeatureEnum::device_ext)},
      {"cudaResourceType",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "image_data_type",
                                      HelperFeatureEnum::device_ext)},
      {"CUtexref", std::make_shared<TypeNameRule>(
                       DpctGlobalInfo::useExtBindlessImages()
                           ? getDpctNamespace() +
                                 "experimental::bindless_image_wrapper_base_p"
                           : getDpctNamespace() + "image_wrapper_base_p",
                       HelperFeatureEnum::device_ext)},
      {"cudaDeviceAttr", std::make_shared<TypeNameRule>("int")},
      {"__nv_bfloat16", std::make_shared<TypeNameRule>(
                            getClNamespace() + "ext::oneapi::bfloat16")},
      {"__nv_bfloat162", std::make_shared<TypeNameRule>(
                             getClNamespace() + "marray<" + getClNamespace() +
                             "ext::oneapi::bfloat16, 2>")},
      {"nv_bfloat16", std::make_shared<TypeNameRule>(getClNamespace() +
                                                     "ext::oneapi::bfloat16")},
      {"nv_bfloat162", std::make_shared<TypeNameRule>(
                           getClNamespace() + "marray<" + getClNamespace() +
                           "ext::oneapi::bfloat16, 2>")},
      {"libraryPropertyType_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "version_field",
                                      HelperFeatureEnum::device_ext)},
      {"libraryPropertyType", std::make_shared<TypeNameRule>(
                                  getLibraryHelperNamespace() + "version_field",
                                  HelperFeatureEnum::device_ext)},
      {"ncclUniqueId",
       std::make_shared<TypeNameRule>("oneapi::ccl::kvs::address_type",
                                      HelperFeatureEnum::device_ext)},
      {"ncclComm_t", std::make_shared<TypeNameRule>(
                         getLibraryHelperNamespace() + "ccl::comm_ptr",
                         HelperFeatureEnum::device_ext)},
      {"ncclRedOp_t", std::make_shared<TypeNameRule>("oneapi::ccl::reduction")},
      {"ncclDataType_t",
       std::make_shared<TypeNameRule>("oneapi::ccl::datatype")},
      {"cuda::std::tuple", std::make_shared<TypeNameRule>("std::tuple")},
      {"cuda::std::complex", std::make_shared<TypeNameRule>("std::complex")},
      {"cuda::std::array", std::make_shared<TypeNameRule>("std::array")},
      {"cusolverEigRange_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::rangev")},
      {"cudaUUID_t",
       std::make_shared<TypeNameRule>("std::array<unsigned char, 16>")},
      {"CUuuid",
       std::make_shared<TypeNameRule>("std::array<unsigned char, 16>")},
      {"cusparseIndexType_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t")},
      {"cusparseFormat_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "sparse::matrix_format")},
      {"cusparseDnMatDescr_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                      getLibraryHelperNamespace() +
                                      "sparse::dense_matrix_desc>")},
      {"cusparseConstDnMatDescr_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                      getLibraryHelperNamespace() +
                                      "sparse::dense_matrix_desc>")},
      {"cusparseOrder_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::layout")},
      {"cusparseDnVecDescr_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                      getLibraryHelperNamespace() +
                                      "sparse::dense_vector_desc>")},
      {"cusparseConstDnVecDescr_t",
       std::make_shared<TypeNameRule>("std::shared_ptr<" +
                                      getLibraryHelperNamespace() +
                                      "sparse::dense_vector_desc>")},
      {"cusparseSpMatDescr_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "sparse::sparse_matrix_desc_t")},
      {"cusparseConstSpMatDescr_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "sparse::sparse_matrix_desc_t")},
      {"cusparseSpMMAlg_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpMVAlg_t", std::make_shared<TypeNameRule>("int")},
      {"cusolverDnFunction_t", std::make_shared<TypeNameRule>("int")},
      {"cusolverAlgMode_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpGEMMDescr_t",
       std::make_shared<TypeNameRule>("oneapi::mkl::sparse::matmat_descr_t")},
      {"cusparseSpSVDescr_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpGEMMAlg_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpSVAlg_t", std::make_shared<TypeNameRule>("int")},
      {"__half_raw", std::make_shared<TypeNameRule>("uint16_t")},
      {"cudaFuncAttributes",
       std::make_shared<TypeNameRule>(MapNames::getDpctNamespace() +
                                      "kernel_function_info")},
      {"ncclResult_t", std::make_shared<TypeNameRule>("int")},
      {"cudaLaunchAttributeValue", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpSMDescr_t", std::make_shared<TypeNameRule>("int")},
      {"cusparseSpSMAlg_t", std::make_shared<TypeNameRule>("int")},
      {"cublasLtHandle_t", std::make_shared<TypeNameRule>(
                               getLibraryHelperNamespace() +
                               "blas_gemm::experimental::descriptor_ptr")},
      {"cublasLtMatmulDesc_t", std::make_shared<TypeNameRule>(
                                   getLibraryHelperNamespace() +
                                   "blas_gemm::experimental::matmul_desc_ptr")},
      {"cublasLtOrder_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "blas_gemm::experimental::order_t")},
      {"cublasLtPointerMode_t", std::make_shared<TypeNameRule>(
                                    getLibraryHelperNamespace() +
                                    "blas_gemm::experimental::pointer_mode_t")},
      {"cublasLtMatrixLayout_t",
       std::make_shared<TypeNameRule>(
           getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_ptr")},
      {"cublasLtMatrixLayoutAttribute_t",
       std::make_shared<TypeNameRule>(
           getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute")},
      {"cublasLtMatmulDescAttributes_t",
       std::make_shared<TypeNameRule>(
           getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute")},
      {"cublasLtMatmulAlgo_t", std::make_shared<TypeNameRule>("int")},
      {"cublasLtEpilogue_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                      "blas_gemm::experimental::epilogue_t")},
      {"cublasLtMatmulPreference_t", std::make_shared<TypeNameRule>("int")},
      {"cublasLtMatmulHeuristicResult_t",
       std::make_shared<TypeNameRule>("int")},
      {"cublasLtMatrixTransformDesc_t",
       std::make_shared<TypeNameRule>(
           getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_ptr")},
      {"cudaGraphicsMapFlags", std::make_shared<TypeNameRule>("int")},
      {"cudaGraphicsRegisterFlags", std::make_shared<TypeNameRule>("int")},
      // ...
  };
  // SYCLcompat unsupport types
  SYCLcompatUnsupportTypes = {
      "cudaChannelFormatDesc",
      "cudaChannelFormatKind",
      "cudaArray",
      "cudaArray_t",
      "cudaMipmappedArray",
      "cudaMipmappedArray_t",
      "cudaTextureDesc",
      "cudaResourceDesc",
      "cudaTextureObject_t",
      "textureReference",
      "cudaTextureAddressMode",
      "cudaTextureFilterMode",
      "CUDA_ARRAY3D_DESCRIPTOR",
      "CUDA_ARRAY_DESCRIPTOR",
      "CUtexObject",
      "CUarray_format",
      "CUarray",
      "CUarray_st",
      "CUDA_RESOURCE_DESC",
      "CUDA_TEXTURE_DESC",
      "CUaddress_mode",
      "CUaddress_mode_enum",
      "CUfilter_mode",
      "CUfilter_mode_enum",
      "CUresourcetype_enum",
      "CUresourcetype",
      "cudaResourceType",
      "CUtexref",
      "cudaStreamCaptureStatus",
  };

  if (DpctGlobalInfo::useSYCLCompat()) {
    for (const auto &Type : SYCLcompatUnsupportTypes)
      TypeNamesMap.erase(Type);
  }

  // Host Random Engine Type mapping
  RandomEngineTypeMap = {
      {"CURAND_RNG_PSEUDO_DEFAULT",
       getLibraryHelperNamespace() + "rng::random_engine_type::mcg59"},
      {"CURAND_RNG_PSEUDO_XORWOW",
       getLibraryHelperNamespace() + "rng::random_engine_type::mcg59"},
      {"CURAND_RNG_PSEUDO_MRG32K3A",
       getLibraryHelperNamespace() + "rng::random_engine_type::mrg32k3a"},
      {"CURAND_RNG_PSEUDO_MTGP32",
       getLibraryHelperNamespace() + "rng::random_engine_type::mt2203"},
      {"CURAND_RNG_PSEUDO_MT19937",
       getLibraryHelperNamespace() + "rng::random_engine_type::mt19937"},
      {"CURAND_RNG_PSEUDO_PHILOX4_32_10",
       getLibraryHelperNamespace() + "rng::random_engine_type::philox4x32x10"},
      {"CURAND_RNG_QUASI_DEFAULT",
       getLibraryHelperNamespace() + "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SOBOL32",
       getLibraryHelperNamespace() + "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
       getLibraryHelperNamespace() + "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SOBOL64",
       getLibraryHelperNamespace() + "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
       getLibraryHelperNamespace() + "rng::random_engine_type::sobol"},
  };

  // Random Ordering Type mapping
  RandomOrderingTypeMap = {
      {"CURAND_ORDERING_PSEUDO_DEFAULT",
       getLibraryHelperNamespace() + "rng::random_mode::best"},
      {"CURAND_ORDERING_PSEUDO_BEST",
       getLibraryHelperNamespace() + "rng::random_mode::best"},
      // CURAND_ORDERING_PSEUDO_SEEDED not support now.
      {"CURAND_ORDERING_PSEUDO_LEGACY",
       getLibraryHelperNamespace() + "rng::random_mode::legacy"},
      {"CURAND_ORDERING_PSEUDO_DYNAMIC",
       getLibraryHelperNamespace() + "rng::random_mode::optimal"},
      // CURAND_ORDERING_QUASI_DEFAULT not support now.
  };

  // Device Random Generator Type mapping
  DeviceRandomGeneratorTypeMap = {
      {"curandStateXORWOW_t", getLibraryHelperNamespace() +
                                  "rng::device::rng_generator<oneapi::"
                                  "mkl::rng::device::mcg59<1>>"},
      {"curandStateXORWOW", getLibraryHelperNamespace() +
                                "rng::device::rng_generator<oneapi::"
                                "mkl::rng::device::mcg59<1>>"},
      {"curandState_t", getLibraryHelperNamespace() +
                            "rng::device::rng_generator<oneapi::mkl::"
                            "rng::device::mcg59<1>>"},
      {"curandState", getLibraryHelperNamespace() +
                          "rng::device::rng_generator<oneapi::mkl::"
                          "rng::device::mcg59<1>>"},
      {"curandStatePhilox4_32_10_t",
       getLibraryHelperNamespace() +
           "rng::device::rng_generator<oneapi::mkl::rng::device::"
           "philox4x32x10<1>>"},
      {"curandStatePhilox4_32_10",
       getLibraryHelperNamespace() + "rng::device::rng_generator<"
                            "oneapi::mkl::rng::device::philox4x32x10<1>>"},
      {"curandStateMRG32k3a_t", getLibraryHelperNamespace() +
                                    "rng::device::rng_generator<"
                                    "oneapi::mkl::rng::device::mrg32k3a<1>>"},
      {"curandStateMRG32k3a", getLibraryHelperNamespace() +
                                  "rng::device::rng_generator<oneapi::"
                                  "mkl::rng::device::mrg32k3a<1>>"},
  };

  // CuDNN Type names mapping.
  CuDNNTypeNamesMap = {
      {"cudnnHandle_t", std::make_shared<TypeNameRule>(
                            getLibraryHelperNamespace() + "dnnl::engine_ext",
                            HelperFeatureEnum::device_ext)},
      {"cudnnStatus_t",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "err1",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnTensorDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnFilterDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnTensorFormat_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::memory_format_tag",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDataType_t", std::make_shared<TypeNameRule>(
                              getLibraryHelperNamespace() + "library_data_t",
                              HelperFeatureEnum::device_ext)},
      {"cudnnActivationDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::activation_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnActivationMode_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnLRNDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::lrn_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnLRNMode_t", std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnPoolingDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::pooling_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnPoolingMode_t", std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnSoftmaxAlgorithm_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::softmax_algorithm",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnSoftmaxMode_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::softmax_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnReduceTensorDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::reduction_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnReduceTensorOp_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::reduction_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnOpTensorDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::binary_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnOpTensorOp_t", std::make_shared<TypeNameRule>(
                                getLibraryHelperNamespace() + "dnnl::binary_op",
                                HelperFeatureEnum::device_ext)},
      {"cudnnBatchNormOps_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_ops",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnBatchNormMode_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnNormOps_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_ops",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnNormMode_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::convolution_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionFwdAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionBwdDataAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionBwdFilterAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionFwdAlgoPerf_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionBwdFilterAlgoPerf_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionBwdDataAlgoPerf_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNMode_t", std::make_shared<TypeNameRule>(
                             getLibraryHelperNamespace() + "dnnl::rnn_mode",
                             HelperFeatureEnum::device_ext)},
      {"cudnnRNNBiasMode_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::rnn_bias_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDirectionMode_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::rnn_direction",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::rnn_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnForwardMode_t", std::make_shared<TypeNameRule>("dnnl::prop_kind")},
      {"cudnnRNNDataDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNDataLayout_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::rnn_memory_format_tag",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDropoutDescriptor_t",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "dnnl::dropout_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionMode_t", std::make_shared<TypeNameRule>("int")},
      {"cudnnNanPropagation_t", std::make_shared<TypeNameRule>("int")},
  };

  // CuDNN Enum constants name mapping.
  CuDNNTypeRule::CuDNNEnumNamesMap = {
      {"CUDNN_TENSOR_NCHW",
       getLibraryHelperNamespace() + "dnnl::memory_format_tag::nchw"},
      {"CUDNN_TENSOR_NHWC",
       getLibraryHelperNamespace() + "dnnl::memory_format_tag::nhwc"},
      {"CUDNN_TENSOR_NCHW_VECT_C",
       getLibraryHelperNamespace() + "dnnl::memory_format_tag::nchw_blocked"},
      {"CUDNN_DATA_FLOAT", getLibraryHelperNamespace() + "library_data_t::real_float"},
      {"CUDNN_DATA_DOUBLE", getLibraryHelperNamespace() + "library_data_t::real_double"},
      {"CUDNN_DATA_HALF", getLibraryHelperNamespace() + "library_data_t::real_half"},
      {"CUDNN_DATA_INT8", getLibraryHelperNamespace() + "library_data_t::real_int8"},
      {"CUDNN_DATA_UINT8", getLibraryHelperNamespace() + "library_data_t::real_uint8"},
      {"CUDNN_DATA_INT32", getLibraryHelperNamespace() + "library_data_t::real_int32"},
      {"CUDNN_DATA_INT8x4", getLibraryHelperNamespace() + "library_data_t::real_int8_4"},
      {"CUDNN_DATA_INT8x32",
       getLibraryHelperNamespace() + "library_data_t::real_int8_32"},
      {"CUDNN_DATA_UINT8x4",
       getLibraryHelperNamespace() + "library_data_t::real_uint8_4"},
      {"CUDNN_DATA_BFLOAT16",
       getLibraryHelperNamespace() + "library_data_t::real_bfloat16"},
      {"CUDNN_ACTIVATION_SIGMOID",
       "dnnl::algorithm::eltwise_logistic_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_RELU",
       "dnnl::algorithm::eltwise_relu_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_TANH",
       "dnnl::algorithm::eltwise_tanh_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_CLIPPED_RELU", "dnnl::algorithm::eltwise_clip"},
      {"CUDNN_ACTIVATION_ELU", "dnnl::algorithm::eltwise_elu_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_IDENTITY", "dnnl::algorithm::eltwise_linear"},
      {"CUDNN_ACTIVATION_SWISH", "dnnl::algorithm::eltwise_swish"},
      {"CUDNN_LRN_CROSS_CHANNEL_DIM1", "dnnl::algorithm::lrn_across_channels"},
      {"CUDNN_POOLING_MAX", "dnnl::algorithm::pooling_max"},
      {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
       "dnnl::algorithm::pooling_avg_include_padding"},
      {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
       "dnnl::algorithm::pooling_avg_exclude_padding"},
      {"CUDNN_POOLING_MAX_DETERMINISTIC", "dnnl::algorithm::pooling_max"},
      {"CUDNN_SOFTMAX_FAST",
       getLibraryHelperNamespace() + "dnnl::softmax_algorithm::normal"},
      {"CUDNN_SOFTMAX_ACCURATE",
       getLibraryHelperNamespace() + "dnnl::softmax_algorithm::normal"},
      {"CUDNN_SOFTMAX_LOG",
       getLibraryHelperNamespace() + "dnnl::softmax_algorithm::log"},
      {"CUDNN_SOFTMAX_MODE_INSTANCE",
       getLibraryHelperNamespace() + "dnnl::softmax_mode::instance"},
      {"CUDNN_SOFTMAX_MODE_CHANNEL",
       getLibraryHelperNamespace() + "dnnl::softmax_mode::channel"},
      {"CUDNN_REDUCE_TENSOR_ADD",
       getLibraryHelperNamespace() + "dnnl::reduction_op::sum"},
      {"CUDNN_REDUCE_TENSOR_MUL",
       getLibraryHelperNamespace() + "dnnl::reduction_op::mul"},
      {"CUDNN_REDUCE_TENSOR_MIN",
       getLibraryHelperNamespace() + "dnnl::reduction_op::min"},
      {"CUDNN_REDUCE_TENSOR_MAX",
       getLibraryHelperNamespace() + "dnnl::reduction_op::max"},
      {"CUDNN_REDUCE_TENSOR_AMAX",
       getLibraryHelperNamespace() + "dnnl::reduction_op::amax"},
      {"CUDNN_REDUCE_TENSOR_AVG",
       getLibraryHelperNamespace() + "dnnl::reduction_op::mean"},
      {"CUDNN_REDUCE_TENSOR_NORM1",
       getLibraryHelperNamespace() + "dnnl::reduction_op::norm1"},
      {"CUDNN_REDUCE_TENSOR_NORM2",
       getLibraryHelperNamespace() + "dnnl::reduction_op::norm2"},
      {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",
       getLibraryHelperNamespace() + "dnnl::reduction_op::mul_no_zeros"},
      {"CUDNN_OP_TENSOR_ADD", getLibraryHelperNamespace() + "dnnl::binary_op::add"},
      {"CUDNN_OP_TENSOR_MUL", getLibraryHelperNamespace() + "dnnl::binary_op::mul"},
      {"CUDNN_OP_TENSOR_MIN", getLibraryHelperNamespace() + "dnnl::binary_op::min"},
      {"CUDNN_OP_TENSOR_MAX", getLibraryHelperNamespace() + "dnnl::binary_op::max"},
      {"CUDNN_OP_TENSOR_SQRT", getLibraryHelperNamespace() + "dnnl::binary_op::sqrt"},
      {"CUDNN_OP_TENSOR_NOT", getLibraryHelperNamespace() + "dnnl::binary_op::neg"},
      {"CUDNN_BATCHNORM_OPS_BN",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::none"},
      {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::activation"},
      {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::add_activation"},
      {"CUDNN_BATCHNORM_PER_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_mode::per_activation"},
      {"CUDNN_BATCHNORM_SPATIAL",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_NORM_OPS_NORM",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::none"},
      {"CUDNN_NORM_OPS_NORM_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::activation"},
      {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_ops::add_activation"},
      {"CUDNN_NORM_PER_ACTIVATION",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_mode::per_activation"},
      {"CUDNN_NORM_PER_CHANNEL",
       getLibraryHelperNamespace() + "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM", "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_FFT", "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_RNN_RELU", getLibraryHelperNamespace() + "dnnl::rnn_mode::vanilla_relu"},
      {"CUDNN_RNN_TANH", getLibraryHelperNamespace() + "dnnl::rnn_mode::vanilla_tanh"},
      {"CUDNN_LSTM", getLibraryHelperNamespace() + "dnnl::rnn_mode::lstm"},
      {"CUDNN_GRU", getLibraryHelperNamespace() + "dnnl::rnn_mode::gru"},
      {"CUDNN_RNN_NO_BIAS", getLibraryHelperNamespace() + "dnnl::rnn_bias_mode::none"},
      {"CUDNN_RNN_SINGLE_INP_BIAS",
       getLibraryHelperNamespace() + "dnnl::rnn_bias_mode::single"},
      {"CUDNN_UNIDIRECTIONAL",
       getLibraryHelperNamespace() + "dnnl::rnn_direction::unidirectional"},
      {"CUDNN_BIDIRECTIONAL",
       getLibraryHelperNamespace() + "dnnl::rnn_direction::bidirectional"},
      {"CUDNN_FWD_MODE_INFERENCE", "dnnl::prop_kind::forward_inference"},
      {"CUDNN_FWD_MODE_TRAINING", "dnnl::prop_kind::forward_training"},
      {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",
       getLibraryHelperNamespace() + "dnnl::rnn_memory_format_tag::tnc"},
      {"CUDNN_DEFAULT_MATH", "dnnl::fpmath_mode::strict"},
      {"CUDNN_TENSOR_OP_MATH", "dnnl::fpmath_mode::strict"},
      {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION", "dnnl::fpmath_mode::any"},
      {"CUDNN_FMA_MATH", "dnnl::fpmath_mode::strict"},
  };
  // CuDNN Enum constants name to helper feature mapping.
  CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap = {
      {"CUDNN_TENSOR_NCHW", HelperFeatureEnum::device_ext},
      {"CUDNN_TENSOR_NHWC", HelperFeatureEnum::device_ext},
      {"CUDNN_TENSOR_NCHW_VECT_C", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_FLOAT", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_DOUBLE", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_HALF", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_UINT8", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT32", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8x4", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8x32", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_UINT8x4", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_BFLOAT16", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_FAST", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_ACCURATE", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_LOG", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_MODE_INSTANCE", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_MODE_CHANNEL", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_ADD", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MUL", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MIN", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MAX", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_AMAX", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_AVG", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_NORM1", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_NORM2", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_ADD", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MUL", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MIN", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MAX", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_SQRT", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_NOT", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_PER_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_SPATIAL", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_PER_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_PER_CHANNEL", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_RELU", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_TANH", HelperFeatureEnum::device_ext},
      {"CUDNN_LSTM", HelperFeatureEnum::device_ext},
      {"CUDNN_GRU", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_NO_BIAS", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_SINGLE_INP_BIAS", HelperFeatureEnum::device_ext},
      {"CUDNN_UNIDIRECTIONAL", HelperFeatureEnum::device_ext},
      {"CUDNN_BIDIRECTIONAL", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED", HelperFeatureEnum::device_ext},
  };
  // Enum constants name mapping.
  EnumConstantRule::EnumNamesMap = {
      // ...
      // enum Device Attribute
      // ...
      {"cudaDevAttrHostNativeAtomicSupported",
       std::make_shared<EnumNameRule>(DpctGlobalInfo::useSYCLCompat()
                                          ? "is_native_host_atomic_supported"
                                          : "is_native_atomic_supported",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrComputeCapabilityMajor",
       std::make_shared<EnumNameRule>("get_major_version",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrMaxSharedMemoryPerBlockOptin",
       std::make_shared<EnumNameRule>("get_local_mem_size",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrComputeCapabilityMinor",
       std::make_shared<EnumNameRule>("get_minor_version",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrMultiProcessorCount",
       std::make_shared<EnumNameRule>("get_max_compute_units",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrMaxThreadsPerBlock",
       std::make_shared<EnumNameRule>("get_max_work_group_size",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrClockRate",
       std::make_shared<EnumNameRule>("get_max_clock_frequency",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrIntegrated",
       std::make_shared<EnumNameRule>("get_integrated",
                                      HelperFeatureEnum::device_ext)},
      {"cudaDevAttrConcurrentManagedAccess",
       std::make_shared<EnumNameRule>(
           "get_info<sycl::info::device::usm_shared_allocations>")},
      {"cudaDevAttrTextureAlignment",
       std::make_shared<EnumNameRule>("get_mem_base_addr_align_in_bytes",
                                      HelperFeatureEnum::device_ext)},
      // enum Memcpy Kind
      {"cudaMemcpyHostToHost",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "host_to_host")},
      {"cudaMemcpyHostToDevice",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "host_to_device")},
      {"cudaMemcpyDeviceToHost",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "device_to_host")},
      {"cudaMemcpyDeviceToDevice",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "device_to_device")},
      {"cudaMemcpyDefault",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() +
           (DpctGlobalInfo::useSYCLCompat() ? "experimental::" : "") +
           "automatic")},
      // enum cudaMemory Type
      {"cudaMemoryTypeHost",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::host",
                                      HelperFeatureEnum::device_ext)},
      {"cudaMemoryTypeDevice",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::device",
                                      HelperFeatureEnum::device_ext)},
      {"cudaMemoryTypeUnregistered",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::unknown",
                                      HelperFeatureEnum::device_ext)},
      {"cudaMemoryTypeManaged",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::shared",
                                      HelperFeatureEnum::device_ext)},
      // enum Texture Address Mode
      {"cudaAddressModeWrap",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::repeat")},
      {"cudaAddressModeClamp",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::clamp_to_edge")},
      {"cudaAddressModeMirror",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::mirrored_repeat")},
      {"cudaAddressModeBorder",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::clamp")},
      // enum Texture Filter Mode
      {"cudaFilterModePoint",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "filtering_mode::nearest")},
      {"cudaFilterModeLinear",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "filtering_mode::linear")},
      // enum Channel Format Kind
      {"cudaChannelFormatKindSigned",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_channel_data_type::signed_int",
                                      HelperFeatureEnum::device_ext)},
      {"cudaChannelFormatKindUnsigned",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() + "image_channel_data_type::unsigned_int",
           HelperFeatureEnum::device_ext)},
      {"cudaChannelFormatKindFloat",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_channel_data_type::fp",
                                      HelperFeatureEnum::device_ext)},
      // enum Resource Type
      {"cudaResourceTypeArray",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::matrix",
                                      HelperFeatureEnum::device_ext)},
      {"cudaResourceTypeMipmappedArray",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::matrix",
                                      HelperFeatureEnum::device_ext)},
      {"cudaResourceTypeLinear",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::linear",
                                      HelperFeatureEnum::device_ext)},
      {"cudaResourceTypePitch2D",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::pitch",
                                      HelperFeatureEnum::device_ext)},
      // enum cudaMemoryAdvise
      {"cudaMemAdviseSetReadMostly", std::make_shared<EnumNameRule>("0")},
      {"cudaMemAdviseUnsetReadMostly", std::make_shared<EnumNameRule>("0")},
      {"cudaMemAdviseSetPreferredLocation",
       std::make_shared<EnumNameRule>("0")},
      {"cudaMemAdviseUnsetPreferredLocation",
       std::make_shared<EnumNameRule>("0")},
      {"cudaMemAdviseSetAccessedBy", std::make_shared<EnumNameRule>("0")},
      {"cudaMemAdviseUnsetAccessedBy", std::make_shared<EnumNameRule>("0")},
      // enum cudaStreamCaptureStatus
      {"cudaStreamCaptureStatusNone",
       std::make_shared<EnumNameRule>(
           DpctGlobalInfo::useExtGraph()
               ? getClNamespace() +
                     "ext::oneapi::experimental::queue_state::executing"
               : "cudaStreamCaptureStatusNone")},
      {"cudaStreamCaptureStatusActive",
       std::make_shared<EnumNameRule>(
           DpctGlobalInfo::useExtGraph()
               ? getClNamespace() +
                     "ext::oneapi::experimental::queue_state::recording"
               : "cudaStreamCaptureStatusActive")},
      {"cudaStreamCaptureStatusInvalidated",
       std::make_shared<EnumNameRule>("cudaStreamCaptureStatusInvalidated")},
      // enum CUmem_advise_enum
      {"CU_MEM_ADVISE_SET_READ_MOSTLY", std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ADVISE_UNSET_READ_MOSTLY", std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
       std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
       std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ADVISE_SET_ACCESSED_BY", std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ADVISE_UNSET_ACCESSED_BY", std::make_shared<EnumNameRule>("0")},
      // enum Driver Device Attribute
      {"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
       std::make_shared<EnumNameRule>("get_major_version",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
       std::make_shared<EnumNameRule>("get_minor_version",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
       std::make_shared<EnumNameRule>(
           "has(sycl::aspect::usm_host_allocations)")},
      {"CU_DEVICE_ATTRIBUTE_WARP_SIZE",
       std::make_shared<EnumNameRule>("get_max_sub_group_size",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
       std::make_shared<EnumNameRule>("get_max_register_size_per_work_group",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
       std::make_shared<EnumNameRule>("get_max_work_group_size",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
       std::make_shared<EnumNameRule>("get_mem_base_addr_align_in_bytes",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
       std::make_shared<EnumNameRule>("get_global_mem_size",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_INTEGRATED",
       std::make_shared<EnumNameRule>("get_integrated",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
       std::make_shared<EnumNameRule>("get_max_clock_frequency",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
       std::make_shared<EnumNameRule>("get_max_compute_units",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
       std::make_shared<EnumNameRule>(DpctGlobalInfo::useSYCLCompat()
                                          ? "is_native_host_atomic_supported"
                                          : "is_native_atomic_supported",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
       std::make_shared<EnumNameRule>("get_max_work_item_sizes",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
       std::make_shared<EnumNameRule>("get_max_work_item_sizes",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
       std::make_shared<EnumNameRule>("get_max_work_item_sizes",
                                      HelperFeatureEnum::device_ext)},
      {"CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
       std::make_shared<EnumNameRule>(
           "has(sycl::aspect::ext_oneapi_virtual_mem)")},
      {"CU_CTX_MAP_HOST", std::make_shared<EnumNameRule>("0")},
      {"CU_CTX_SCHED_BLOCKING_SYNC", std::make_shared<EnumNameRule>("0")},
      {"CU_CTX_SCHED_SPIN", std::make_shared<EnumNameRule>("0")},
      {"CU_CTX_SCHED_SPIN", std::make_shared<EnumNameRule>("0")},
      {"CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
       std::make_shared<EnumNameRule>("get_device_info().get_local_mem_size",
                                      HelperFeatureEnum::device_ext)},

      // enum CUpointer_attribute
      {"CU_POINTER_ATTRIBUTE_CONTEXT",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::memory_type")},
      {"CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() + "pointer_attributes::type::device_pointer")},
      {"CU_POINTER_ATTRIBUTE_HOST_POINTER",
       std::make_shared<EnumNameRule>(
           getDpctNamespace() + "pointer_attributes::type::host_pointer")},
      {"CU_POINTER_ATTRIBUTE_P2P_TOKENS",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_BUFFER_ID",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_IS_MANAGED",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::is_managed")},
      {"CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::device_id")},
      {"CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_RANGE_SIZE",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_MAPPED",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},
      {"CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                      "pointer_attributes::type::unsupported")},

      // enum CUmemorytype Type
      {"CU_MEMORYTYPE_HOST",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::host",
                                      HelperFeatureEnum::device_ext)},
      {"CU_MEMORYTYPE_DEVICE",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::device",
                                      HelperFeatureEnum::device_ext)},
      {"CU_MEMORYTYPE_UNIFIED",
       std::make_shared<EnumNameRule>(getClNamespace() + "usm::alloc::shared",
                                      HelperFeatureEnum::device_ext)},

      // enum CUlimit
      {"CU_LIMIT_PRINTF_FIFO_SIZE", std::make_shared<EnumNameRule>("INT_MAX")},

      // enum CUarray_format
      {"CU_AD_FORMAT_UNSIGNED_INT8",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::unsigned_int8")},
      {"CU_AD_FORMAT_UNSIGNED_INT16",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::unsigned_int16")},
      {"CU_AD_FORMAT_UNSIGNED_INT32",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::unsigned_int32")},
      {"CU_AD_FORMAT_SIGNED_INT8",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::signed_int8")},
      {"CU_AD_FORMAT_SIGNED_INT16",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::unsigned_int16")},
      {"CU_AD_FORMAT_SIGNED_INT32",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::unsigned_int32")},
      {"CU_AD_FORMAT_HALF", std::make_shared<EnumNameRule>(
                                getClNamespace() + "image_channel_type::fp16")},
      {"CU_AD_FORMAT_FLOAT",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "image_channel_type::fp32")},
      // enum CUaddress_mode_enum
      {"CU_TR_ADDRESS_MODE_WRAP",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::repeat")},
      {"CU_TR_ADDRESS_MODE_CLAMP",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::clamp_to_edge")},
      {"CU_TR_ADDRESS_MODE_MIRROR",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::mirrored_repeat")},
      {"CU_TR_ADDRESS_MODE_BORDER",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "addressing_mode::clamp")},
      // enum CUfilter_mode_enum
      {"CU_TR_FILTER_MODE_POINT",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "filtering_mode::nearest")},
      {"CU_TR_FILTER_MODE_LINEAR",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "filtering_mode::linear")},
      // enum CUresourcetype_enum
      {"CU_RESOURCE_TYPE_ARRAY",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::matrix",
                                      HelperFeatureEnum::device_ext)},
      {"CU_RESOURCE_TYPE_LINEAR",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::linear",
                                      HelperFeatureEnum::device_ext)},
      {"CU_RESOURCE_TYPE_PITCH2D",
       std::make_shared<EnumNameRule>(getDpctNamespace() +
                                          "image_data_type::pitch",
                                      HelperFeatureEnum::device_ext)},
      // enum libraryPropertyType_t
      {"MAJOR_VERSION",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                          "version_field::major",
                                      HelperFeatureEnum::device_ext)},
      {"MINOR_VERSION",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                          "version_field::update",
                                      HelperFeatureEnum::device_ext)},
      {"PATCH_LEVEL", std::make_shared<EnumNameRule>(
                          getLibraryHelperNamespace() + "version_field::patch",
                          HelperFeatureEnum::device_ext)},
      // enum cudaDataType_t
      {"CUDA_R_16F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_half")},
      {"CUDA_C_16F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_half")},
      {"CUDA_R_16BF",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_bfloat16")},
      {"CUDA_C_16BF",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_bfloat16")},
      {"CUDA_R_32F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_float")},
      {"CUDA_C_32F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_float")},
      {"CUDA_R_64F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_double")},
      {"CUDA_C_64F",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_double")},
      {"CUDA_R_4I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_int4")},
      {"CUDA_C_4I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_int4")},
      {"CUDA_R_4U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_uint4")},
      {"CUDA_C_4U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_uint4")},
      {"CUDA_R_8I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_int8")},
      {"CUDA_C_8I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_int8")},
      {"CUDA_R_8U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_uint8")},
      {"CUDA_C_8U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_uint8")},
      {"CUDA_R_16I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_int16")},
      {"CUDA_C_16I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_int16")},
      {"CUDA_R_16U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_uint16")},
      {"CUDA_C_16U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_uint16")},
      {"CUDA_R_32I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_int32")},
      {"CUDA_C_32I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_int32")},
      {"CUDA_R_32U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_uint32")},
      {"CUDA_C_32U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_uint32")},
      {"CUDA_R_64I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_int64")},
      {"CUDA_C_64I",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_int64")},
      {"CUDA_R_64U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_uint64")},
      {"CUDA_C_64U",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::complex_uint64")},
      {"CUDA_R_8F_E4M3",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_f8_e4m3")},
      {"CUDA_R_8F_E5M2",
       std::make_shared<EnumNameRule>(getLibraryHelperNamespace() +
                                      "library_data_t::real_f8_e5m2")},
      {"cuda::thread_scope_system",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_scope::system")},
      {"cuda::thread_scope_device",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_scope::device")},
      {"cuda::thread_scope_block",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_scope::work_group")},
      {"cuda::memory_order_relaxed",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_order::relaxed")},
      {"cuda::memory_order_acq_rel",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_order::acq_rel")},
      {"cuda::memory_order_release",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_order::release")},
      {"cuda:::memory_order_acquire",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_order::acquire")},
      {"cuda::memory_order_seq_cst",
       std::make_shared<EnumNameRule>(getClNamespace() +
                                      "memory_order::seq_cst")},
      {"CUFFT_R2C", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::real_float_to_complex_float",
                        HelperFeatureEnum::device_ext)},
      {"CUFFT_C2R", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::complex_float_to_real_float",
                        HelperFeatureEnum::device_ext)},
      {"CUFFT_D2Z", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::real_double_to_complex_double",
                        HelperFeatureEnum::device_ext)},
      {"CUFFT_Z2D", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::complex_double_to_real_double",
                        HelperFeatureEnum::device_ext)},
      {"CUFFT_C2C", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::complex_float_to_complex_float",
                        HelperFeatureEnum::device_ext)},
      {"CUFFT_Z2Z", std::make_shared<EnumNameRule>(
                        getLibraryHelperNamespace() +
                            "fft::fft_type::complex_double_to_complex_double",
                        HelperFeatureEnum::device_ext)},
      {"ncclSum",
       std::make_shared<EnumNameRule>("oneapi::ccl::reduction::sum")},
      {"ncclProd",
       std::make_shared<EnumNameRule>("oneapi::ccl::reduction::prod")},
      {"ncclMin",
       std::make_shared<EnumNameRule>("oneapi::ccl::reduction::min")},
      {"ncclMax",
       std::make_shared<EnumNameRule>("oneapi::ccl::reduction::max")},
      {"ncclInt8",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::int8")},
      {"ncclChar",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::int8")},
      {"ncclUint8",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::uint8")},
      {"ncclInt32",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::int32")},
      {"ncclInt",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::int32")},
      {"ncclUint32",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::uint32")},
      {"ncclInt64",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::int64")},
      {"ncclUint64",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::uint64")},
      {"ncclFloat16",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float16")},
      {"ncclHalf",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float16")},
      {"ncclFloat32",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float32")},
      {"ncclFloat",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float32")},
      {"ncclFloat64",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float64")},
      {"ncclDouble",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::float64")},
      {"ncclBfloat16",
       std::make_shared<EnumNameRule>("oneapi::ccl::datatype::bfloat16")},
      {"CUSOLVER_EIG_RANGE_ALL",
       std::make_shared<EnumNameRule>("oneapi::mkl::rangev::all")},
      {"CUSOLVER_EIG_RANGE_V",
       std::make_shared<EnumNameRule>("oneapi::mkl::rangev::values")},
      {"CUSOLVER_EIG_RANGE_I",
       std::make_shared<EnumNameRule>("oneapi::mkl::rangev::indices")},
      {"ncclSuccess", std::make_shared<EnumNameRule>("0")},
      // enum cudaGraphicsMapFlags
      {"cudaGraphicsMapFlagsNone", std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsMapFlagsReadOnly", std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsMapFlagsWriteDiscard", std::make_shared<EnumNameRule>("0")},
      // enum cudaGraphicsRegisterFlags
      {"cudaGraphicsRegisterFlagsNone", std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsRegisterFlagsReadOnly",
       std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsRegisterFlagsWriteDiscard",
       std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsRegisterFlagsSurfaceLoadStore",
       std::make_shared<EnumNameRule>("0")},
      {"cudaGraphicsRegisterFlagsTextureGather",
       std::make_shared<EnumNameRule>("0")},
      // ...
  };

  // spBLAS enums mapping
  SPBLASEnumsMap = {
      {"CUSPARSE_OPERATION_NON_TRANSPOSE", "oneapi::mkl::transpose::nontrans"},
      {"CUSPARSE_OPERATION_TRANSPOSE", "oneapi::mkl::transpose::trans"},
      {"CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
       "oneapi::mkl::transpose::conjtrans"},
      {"CUSPARSE_FILL_MODE_LOWER", "oneapi::mkl::uplo::lower"},
      {"CUSPARSE_FILL_MODE_UPPER", "oneapi::mkl::uplo::upper"},
      {"CUSPARSE_DIAG_TYPE_NON_UNIT", "oneapi::mkl::diag::nonunit"},
      {"CUSPARSE_DIAG_TYPE_UNIT", "oneapi::mkl::diag::unit"},
      {"CUSPARSE_INDEX_BASE_ZERO", "oneapi::mkl::index_base::zero"},
      {"CUSPARSE_INDEX_BASE_ONE", "oneapi::mkl::index_base::one"},
      {"CUSPARSE_MATRIX_TYPE_GENERAL",
       getLibraryHelperNamespace() + "sparse::matrix_info::matrix_type::ge"},
      {"CUSPARSE_MATRIX_TYPE_SYMMETRIC",
       getLibraryHelperNamespace() + "sparse::matrix_info::matrix_type::sy"},
      {"CUSPARSE_MATRIX_TYPE_HERMITIAN",
       getLibraryHelperNamespace() + "sparse::matrix_info::matrix_type::he"},
      {"CUSPARSE_MATRIX_TYPE_TRIANGULAR",
       getLibraryHelperNamespace() + "sparse::matrix_info::matrix_type::tr"},
      {"CUSPARSE_SPMAT_FILL_MODE",
       getLibraryHelperNamespace() + "sparse::matrix_attribute::uplo"},
      {"CUSPARSE_SPMAT_DIAG_TYPE",
       getLibraryHelperNamespace() + "sparse::matrix_attribute::diag"},
      {"CUSPARSE_INDEX_16U",
       getLibraryHelperNamespace() + "library_data_t::real_uint16"},
      {"CUSPARSE_INDEX_32I", getLibraryHelperNamespace() + "library_data_t::real_int32"},
      {"CUSPARSE_INDEX_64I", getLibraryHelperNamespace() + "library_data_t::real_int64"},
      {"CUSPARSE_ORDER_COL", "oneapi::mkl::layout::col_major"},
      {"CUSPARSE_ORDER_ROW", "oneapi::mkl::layout::row_major"},
      {"CUSPARSE_ACTION_SYMBOLIC",
       getLibraryHelperNamespace() + "sparse::conversion_scope::index"},
      {"CUSPARSE_ACTION_NUMERIC",
       getLibraryHelperNamespace() + "sparse::conversion_scope::index_and_value"},
  };

  // CUB enums mapping
  // clang-format off
  CUBEnumsMap = {
    {"BLOCK_STORE_DIRECT", getDpctNamespace() + "group::group_store_algorithm::blocked"},
    {"BLOCK_STORE_STRIPED", getDpctNamespace() + "group::group_store_algorithm::striped"},
    {"BLOCK_LOAD_DIRECT", getDpctNamespace() + "group::group_load_algorithm::blocked"},
    {"BLOCK_LOAD_STRIPED", getDpctNamespace() + "group::group_load_algorithm::striped"}
  };
  // clang-format on

  // BLAS enums mapping
  BLASEnumsMap = {
      {"CUBLAS_OP_N", "oneapi::mkl::transpose::nontrans"},
      {"CUBLAS_OP_T", "oneapi::mkl::transpose::trans"},
      {"CUBLAS_OP_C", "oneapi::mkl::transpose::conjtrans"},
      {"CUBLAS_SIDE_LEFT", "oneapi::mkl::side::left"},
      {"CUBLAS_SIDE_RIGHT", "oneapi::mkl::side::right"},
      {"CUBLAS_FILL_MODE_LOWER", "oneapi::mkl::uplo::lower"},
      {"CUBLAS_FILL_MODE_UPPER", "oneapi::mkl::uplo::upper"},
      {"CUBLAS_DIAG_NON_UNIT", "oneapi::mkl::diag::nonunit"},
      {"CUBLAS_DIAG_UNIT", "oneapi::mkl::diag::unit"},
      {"CUBLAS_DEFAULT_MATH",
       getLibraryHelperNamespace() + "blas::math_mode::mm_default"},
      {"CUBLAS_TENSOR_OP_MATH",
       getLibraryHelperNamespace() + "blas::math_mode::mm_tf32"},
      {"CUBLAS_PEDANTIC_MATH",
       getLibraryHelperNamespace() + "blas::math_mode::mm_default"},
      {"CUBLAS_TF32_TENSOR_OP_MATH",
       getLibraryHelperNamespace() + "blas::math_mode::mm_tf32"},
      {"CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",
       getLibraryHelperNamespace() + "blas::math_mode::mm_default"},
      {"CUBLASLT_ORDER_COL",
       getLibraryHelperNamespace() + "blas_gemm::experimental::order_t::col"},
      {"CUBLASLT_ORDER_ROW",
       getLibraryHelperNamespace() + "blas_gemm::experimental::order_t::row"},
      {"CUBLASLT_ORDER_COL32",
       getLibraryHelperNamespace() + "blas_gemm::experimental::order_t::col32"},
      {"CUBLASLT_ORDER_COL4_4R2_8C",
       getLibraryHelperNamespace() + "blas_gemm::experimental::order_t::col4_4r2_8c"},
      {"CUBLASLT_ORDER_COL32_2R_4R4",
       getLibraryHelperNamespace() + "blas_gemm::experimental::order_t::col32_2r_4r4"},
      {"CUBLASLT_POINTER_MODE_HOST",
       getLibraryHelperNamespace() + "blas_gemm::experimental::pointer_mode_t::host"},
      {"CUBLASLT_POINTER_MODE_DEVICE",
       getLibraryHelperNamespace() + "blas_gemm::experimental::pointer_mode_t::device"},
      {"CUBLASLT_POINTER_MODE_DEVICE_VECTOR",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::pointer_mode_t::device_vector"},
      {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",
       getLibraryHelperNamespace() + "blas_gemm::experimental::pointer_mode_t::alpha_"
                            "device_vector_beta_zero"},
      {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",
       getLibraryHelperNamespace() + "blas_gemm::experimental::pointer_mode_t::alpha_"
                            "device_vector_beta_host"},
      {"CUBLASLT_MATRIX_LAYOUT_TYPE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute::type"},
      {"CUBLASLT_MATRIX_LAYOUT_BATCH_COUT",
       getLibraryHelperNamespace() + 
           "blas_gemm::experimental::matrix_layout_t::attribute::batch_count"},
      {"CUBLASLT_MATRIX_LAYOUT_ORDER",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute::order"},
      {"CUBLASLT_MATRIX_LAYOUT_ROWS",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute::rows"},
      {"CUBLASLT_MATRIX_LAYOUT_COLS",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute::cols"},
      {"CUBLASLT_MATRIX_LAYOUT_LD",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute::ld"},
      {"CUBLASLT_MATMUL_DESC_COMPUTE_TYPE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::compute_type"},
      {"CUBLASLT_MATMUL_DESC_SCALE_TYPE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::scale_type"},
      {"CUBLASLT_MATMUL_DESC_POINTER_MODE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::pointer_mode"},
      {"CUBLASLT_MATMUL_DESC_TRANSA",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::trans_a"},
      {"CUBLASLT_MATMUL_DESC_TRANSB",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::trans_b"},
      {"CUBLASLT_MATMUL_DESC_TRANSC",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::trans_c"},
      {"CUBLASLT_MATMUL_DESC_EPILOGUE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::epilogue"},
      {"CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_MATMUL_DESC_FAST_ACCUM",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_MATMUL_DESC_A_SCALE_POINTER",
       getLibraryHelperNamespace() + "blas_gemm::experimental::matmul_desc_t::attribute:"
                            ":a_scale_pointer"},
      {"CUBLASLT_MATMUL_DESC_B_SCALE_POINTER",
       getLibraryHelperNamespace() + "blas_gemm::experimental::matmul_desc_t::attribute:"
                            ":b_scale_pointer"},
      {"CUBLASLT_MATMUL_DESC_D_SCALE_POINTER",
       getLibraryHelperNamespace() + "blas_gemm::experimental::matmul_desc_t::attribute:"
                            ":d_scale_pointer"},
      {"CUBLASLT_MATMUL_DESC_AMAX_D_POINTER",
       getLibraryHelperNamespace() + "blas_gemm::experimental::matmul_desc_t::attribute:"
                            ":absmax_d_pointer"},
      {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::attribute::unsupport"},
      {"CUBLASLT_EPILOGUE_DEFAULT",
       getLibraryHelperNamespace() + "blas_gemm::experimental::epilogue_t::nop"},
      {"CUBLASLT_EPILOGUE_RELU",
       getLibraryHelperNamespace() + "blas_gemm::experimental::epilogue_t::relu"},
      {"CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t::attribute::scale_type"},
      {"CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",
       getLibraryHelperNamespace() + "blas_gemm::experimental::transform_desc_t::"
                            "attribute::pointer_mode"},
      {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t::attribute::trans_a"},
      {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t::attribute::trans_b"},
      // cublasComputeType_t
      {"CUBLAS_COMPUTE_16F", getLibraryHelperNamespace() + "compute_type::f16"},
      {"CUBLAS_COMPUTE_16F_PEDANTIC",
       getLibraryHelperNamespace() + "compute_type::f16_standard"},
      {"CUBLAS_COMPUTE_32F", getLibraryHelperNamespace() + "compute_type::f32"},
      {"CUBLAS_COMPUTE_32F_PEDANTIC",
       getLibraryHelperNamespace() + "compute_type::f32_standard"},
      {"CUBLAS_COMPUTE_32F_FAST_16F", getLibraryHelperNamespace() + "compute_type::f32"},
      {"CUBLAS_COMPUTE_32F_FAST_16BF",
       getLibraryHelperNamespace() + "compute_type::f32_fast_bf16"},
      {"CUBLAS_COMPUTE_32F_FAST_TF32",
       getLibraryHelperNamespace() + "compute_type::f32_fast_tf32"},
      {"CUBLAS_COMPUTE_64F", getLibraryHelperNamespace() + "compute_type::f64"},
      {"CUBLAS_COMPUTE_64F_PEDANTIC",
       getLibraryHelperNamespace() + "compute_type::f64_standard"},
      {"CUBLAS_COMPUTE_32I", getLibraryHelperNamespace() + "compute_type::i32"},
      {"CUBLAS_COMPUTE_32I_PEDANTIC",
       getLibraryHelperNamespace() + "compute_type::i32_standard"},
  };

  ClassFieldMap = {};

  // Thrust function name mapping
  ThrustFuncNamesMap = {
#define ENTRY(from, to, policy) {from, {to, policy}},
#define ENTRY_HOST(from, to, policy) ENTRY(from, to, policy)
#define ENTRY_DEVICE(from, to, policy) ENTRY(from, to, policy)
#define ENTRY_BOTH(from, to, policy) ENTRY(from, to, policy)
#include "APINamesMapThrust.inc"
#undef ENTRY
#undef ENTRY_HOST
#undef ENTRY_DEVICE
#undef ENTRY_BOTH
  };

  ThrustFuncNamesHelperFeaturesMap = {
      {"thrust::sequence", HelperFeatureEnum::device_ext},
      {"thrust::stable_sort_by_key", HelperFeatureEnum::device_ext},
      {"thrust::transform_if", HelperFeatureEnum::device_ext},
      {"thrust::device_free", HelperFeatureEnum::device_ext},
      {"thrust::device_malloc", HelperFeatureEnum::device_ext},
      {"thrust::raw_pointer_cast", HelperFeatureEnum::device_ext},
      {"thrust::make_counting_iterator", HelperFeatureEnum::device_ext},
      {"thrust::device_pointer_cast", HelperFeatureEnum::device_ext},
      {"thrust::make_constant_iterator", HelperFeatureEnum::device_ext},
      {"thrust::partition_point", HelperFeatureEnum::device_ext}};

  ITFName = {
#define ENTRY(INTERFACENAME, APINAME, VALUE, FLAG, TARGET, COMMENT)            \
  {#APINAME, #INTERFACENAME},
#define ENTRY_MEMBER_FUNCTION(INTERFACEOBJNAME, OBJNAME, INTERFACENAME,        \
                              APINAME, VALUE, FLAG, TARGET, COMMENT)           \
  {#OBJNAME "::" #APINAME, #INTERFACEOBJNAME "::" #INTERFACENAME},
#include "APINames.inc"
#include "APINames_CUB.inc"
#include "APINames_NCCL.inc"
#include "APINames_cuBLAS.inc"
#include "APINames_cuFFT.inc"
#include "APINames_cuRAND.inc"
#include "APINames_cuSOLVER.inc"
#include "APINames_cuSPARSE.inc"
#include "APINames_nvGRAPH.inc"
#include "APINames_nvJPEG.inc"
#include "APINames_thrust.inc"
#include "APINames_wmma.inc"
#undef ENTRY_MEMBER_FUNCTION
#undef ENTRY
  };

  BLASTGemmExTypeInfoMap = {
      {"2:2",
       {"float", getClNamespace() + "half", getClNamespace() + "half",
        getClNamespace() + "half", getClNamespace() + "half",
        getClNamespace() + "half"}},
      {"2:0",
       {"float", "float", getClNamespace() + "half", getClNamespace() + "half",
        "float", "float"}},
      {"0:0", {"float", "float", "float", "float", "float", "float"}},
      {"4:4",
       {"std::complex<float>", "std::complex<float>",
        getClNamespace() + "float2", "std::complex<float>",
        getClNamespace() + "float2", "std::complex<float>"}}};
  // Atomic function names mapping
  AtomicFuncNamesMap = {
      {"atomicAdd", getDpctNamespace() + "atomic_fetch_add"},
      {"atomicAdd_system", getDpctNamespace() + "atomic_fetch_add"},
      {"atomicSub", getDpctNamespace() + "atomic_fetch_sub"},
      {"atomicSub_system", getDpctNamespace() + "atomic_fetch_sub"},
      {"atomicAnd", getDpctNamespace() + "atomic_fetch_and"},
      {"atomicAnd_system", getDpctNamespace() + "atomic_fetch_and"},
      {"atomicOr", getDpctNamespace() + "atomic_fetch_or"},
      {"atomicOr_system", getDpctNamespace() + "atomic_fetch_or"},
      {"atomicXor", getDpctNamespace() + "atomic_fetch_xor"},
      {"atomicXor_system", getDpctNamespace() + "atomic_fetch_xor"},
      {"atomicMin", getDpctNamespace() + "atomic_fetch_min"},
      {"atomicMin_system", getDpctNamespace() + "atomic_fetch_min"},
      {"atomicMax", getDpctNamespace() + "atomic_fetch_max"},
      {"atomicMax_system", getDpctNamespace() + "atomic_fetch_max"},
      {"atomicExch", getDpctNamespace() + "atomic_exchange"},
      {"atomicExch_system", getDpctNamespace() + "atomic_exchange"},
      {"atomicCAS", getDpctNamespace() + "atomic_compare_exchange_strong"},
      {"atomicCAS_system",
       getDpctNamespace() + "atomic_compare_exchange_strong"},
      {"atomicInc", getDpctNamespace() + "atomic_fetch_compare_inc"},
      {"atomicInc_system", getDpctNamespace() + "atomic_fetch_compare_inc"},
      {"atomicDec", getDpctNamespace() + "atomic_fetch_compare_dec"},
      {"atomicDec_system", getDpctNamespace() + "atomic_fetch_compare_dec"},
  };

  BLASAPIWithRewriter = {
      {"cublasCreate_v2", "handle = queue_p"},
      {"cublasDestroy_v2", "handle = nullptr"},
      {"cublasSetStream_v2", "handle = s"},
      {"cublasGetStream_v2", "s = handle"},
      {"cublasSetKernelStream",
       getLibraryHelperNamespace() + "blas::descriptor::set_saved_queue"},
      {"cublasSetMathMode",
       getLibraryHelperNamespace() + "blas::descriptor::set_math_mode"},
      {"cublasGetMathMode",
       getLibraryHelperNamespace() + "blas::descriptor::get_math_mode"},
      {"cublasHgemm", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasSgemm_v2", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasDgemm_v2", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasCgemm_v2", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasZgemm_v2", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasSsyrk_v2", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasDsyrk_v2", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasCsyrk_v2", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasZsyrk_v2", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasSsymm_v2", "oneapi::mkl::blas::column_major::symm"},
      {"cublasDsymm_v2", "oneapi::mkl::blas::column_major::symm"},
      {"cublasCsymm_v2", "oneapi::mkl::blas::column_major::symm"},
      {"cublasZsymm_v2", "oneapi::mkl::blas::column_major::symm"},
      {"cublasStrsm_v2", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasDtrsm_v2", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasCtrsm_v2", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasZtrsm_v2", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasChemm_v2", "oneapi::mkl::blas::column_major::hemm"},
      {"cublasZhemm_v2", "oneapi::mkl::blas::column_major::hemm"},
      {"cublasCherk_v2", "oneapi::mkl::blas::column_major::herk"},
      {"cublasZherk_v2", "oneapi::mkl::blas::column_major::herk"},
      {"cublasSsyr2k_v2", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasDsyr2k_v2", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasCsyr2k_v2", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasZsyr2k_v2", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasCher2k_v2", "oneapi::mkl::blas::column_major::her2k"},
      {"cublasZher2k_v2", "oneapi::mkl::blas::column_major::her2k"},
      {"cublasHgemmStridedBatched",
       "oneapi::mkl::blas::column_major::gemm_batch"},
      {"cublasSgemmStridedBatched",
       "oneapi::mkl::blas::column_major::gemm_batch"},
      {"cublasDgemmStridedBatched",
       "oneapi::mkl::blas::column_major::gemm_batch"},
      {"cublasCgemmStridedBatched",
       "oneapi::mkl::blas::column_major::gemm_batch"},
      {"cublasZgemmStridedBatched",
       "oneapi::mkl::blas::column_major::gemm_batch"},
      {"cublasNrm2Ex", getLibraryHelperNamespace() + "blas::nrm2"},
      {"cublasNrm2Ex_64", getLibraryHelperNamespace() + "blas::nrm2"},
      {"cublasDotEx", getLibraryHelperNamespace() + "blas::dot"},
      {"cublasDotEx_64", getLibraryHelperNamespace() + "blas::dot"},
      {"cublasDotcEx", getLibraryHelperNamespace() + "blas::dotc"},
      {"cublasDotcEx_64", getLibraryHelperNamespace() + "blas::dotc"},
      {"cublasScalEx", getLibraryHelperNamespace() + "blas::scal"},
      {"cublasScalEx_64", getLibraryHelperNamespace() + "blas::scal"},
      {"cublasAxpyEx", getLibraryHelperNamespace() + "blas::axpy"},
      {"cublasAxpyEx_64", getLibraryHelperNamespace() + "blas::axpy"},
      {"cublasRotEx", getLibraryHelperNamespace() + "blas::rot"},
      {"cublasRotEx_64", getLibraryHelperNamespace() + "blas::rot"},
      {"cublasGemmEx", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasSgemmEx", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasCgemmEx", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasCgemm3mEx", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasGemmBatchedEx", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasGemmBatchedEx_64",
       getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasGemmStridedBatchedEx",
       getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasGemmStridedBatchedEx_64",
       getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasSsyrkx", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasDsyrkx", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasCsyrkx", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasZsyrkx", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasCherkx", getLibraryHelperNamespace() + "blas::herk"},
      {"cublasZherkx", getLibraryHelperNamespace() + "blas::herk"},
      {"cublasSgeam", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasDgeam", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasCgeam", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasZgeam", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasSdgmm", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasDdgmm", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasCdgmm", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasZdgmm", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasHgemmBatched", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasSgemmBatched", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasDgemmBatched", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasCgemmBatched", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasZgemmBatched", getLibraryHelperNamespace() + "blas::gemm_batch"},
      {"cublasStrsmBatched", getLibraryHelperNamespace() + "blas::trsm_batch"},
      {"cublasDtrsmBatched", getLibraryHelperNamespace() + "blas::trsm_batch"},
      {"cublasCtrsmBatched", getLibraryHelperNamespace() + "blas::trsm_batch"},
      {"cublasZtrsmBatched", getLibraryHelperNamespace() + "blas::trsm_batch"},
      {"cublasStrmm_v2", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasDtrmm_v2", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasCtrmm_v2", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasZtrmm_v2", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasSgetrfBatched",
       getLibraryHelperNamespace() + "getrf_batch_wrapper"},
      {"cublasDgetrfBatched",
       getLibraryHelperNamespace() + "getrf_batch_wrapper"},
      {"cublasCgetrfBatched",
       getLibraryHelperNamespace() + "getrf_batch_wrapper"},
      {"cublasZgetrfBatched",
       getLibraryHelperNamespace() + "getrf_batch_wrapper"},
      {"cublasSgetrsBatched",
       getLibraryHelperNamespace() + "getrs_batch_wrapper"},
      {"cublasDgetrsBatched",
       getLibraryHelperNamespace() + "getrs_batch_wrapper"},
      {"cublasCgetrsBatched",
       getLibraryHelperNamespace() + "getrs_batch_wrapper"},
      {"cublasZgetrsBatched",
       getLibraryHelperNamespace() + "getrs_batch_wrapper"},
      {"cublasSgetriBatched",
       getLibraryHelperNamespace() + "getri_batch_wrapper"},
      {"cublasDgetriBatched",
       getLibraryHelperNamespace() + "getri_batch_wrapper"},
      {"cublasCgetriBatched",
       getLibraryHelperNamespace() + "getri_batch_wrapper"},
      {"cublasZgetriBatched",
       getLibraryHelperNamespace() + "getri_batch_wrapper"},
      {"cublasSgeqrfBatched",
       getLibraryHelperNamespace() + "geqrf_batch_wrapper"},
      {"cublasDgeqrfBatched",
       getLibraryHelperNamespace() + "geqrf_batch_wrapper"},
      {"cublasCgeqrfBatched",
       getLibraryHelperNamespace() + "geqrf_batch_wrapper"},
      {"cublasZgeqrfBatched",
       getLibraryHelperNamespace() + "geqrf_batch_wrapper"},
      {"cublasSgelsBatched",
       getLibraryHelperNamespace() + "gels_batch_wrapper"},
      {"cublasDgelsBatched",
       getLibraryHelperNamespace() + "gels_batch_wrapper"},
      {"cublasCgelsBatched",
       getLibraryHelperNamespace() + "gels_batch_wrapper"},
      {"cublasZgelsBatched",
       getLibraryHelperNamespace() + "gels_batch_wrapper"},
      {"cublasGetStatusString", ""},
      {"cublasCgemm3m", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasZgemm3m", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasHgemm_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasSgemm_v2_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasDgemm_v2_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasCgemm_v2_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasZgemm_v2_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasSsyrk_v2_64", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasDsyrk_v2_64", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasCsyrk_v2_64", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasZsyrk_v2_64", "oneapi::mkl::blas::column_major::syrk"},
      {"cublasSsymm_v2_64", "oneapi::mkl::blas::column_major::symm"},
      {"cublasDsymm_v2_64", "oneapi::mkl::blas::column_major::symm"},
      {"cublasCsymm_v2_64", "oneapi::mkl::blas::column_major::symm"},
      {"cublasZsymm_v2_64", "oneapi::mkl::blas::column_major::symm"},
      {"cublasStrsm_v2_64", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasDtrsm_v2_64", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasCtrsm_v2_64", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasZtrsm_v2_64", "oneapi::mkl::blas::column_major::trsm"},
      {"cublasChemm_v2_64", "oneapi::mkl::blas::column_major::hemm"},
      {"cublasZhemm_v2_64", "oneapi::mkl::blas::column_major::hemm"},
      {"cublasCherk_v2_64", "oneapi::mkl::blas::column_major::herk"},
      {"cublasZherk_v2_64", "oneapi::mkl::blas::column_major::herk"},
      {"cublasSsyr2k_v2_64", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasDsyr2k_v2_64", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasCsyr2k_v2_64", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasZsyr2k_v2_64", "oneapi::mkl::blas::column_major::syr2k"},
      {"cublasCher2k_v2_64", "oneapi::mkl::blas::column_major::her2k"},
      {"cublasZher2k_v2_64", "oneapi::mkl::blas::column_major::her2k"},
      {"cublasSgeam_64", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasDgeam_64", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasCgeam_64", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasZgeam_64", "oneapi::mkl::blas::column_major::omatadd"},
      {"cublasSdgmm_64", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasDdgmm_64", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasCdgmm_64", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasZdgmm_64", "oneapi::mkl::blas::column_major::dgmm"},
      {"cublasIsamax_v2_64", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIdamax_v2_64", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIcamax_v2_64", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIzamax_v2_64", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIsamin_v2_64", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIdamin_v2_64", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIcamin_v2_64", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIzamin_v2_64", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIsamax_v2", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIdamax_v2", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIcamax_v2", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIzamax_v2", "oneapi::mkl::blas::column_major::iamax"},
      {"cublasIsamin_v2", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIdamin_v2", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIcamin_v2", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasIzamin_v2", "oneapi::mkl::blas::column_major::iamin"},
      {"cublasSnrm2_v2_64", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasDnrm2_v2_64", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasScnrm2_v2_64", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasDznrm2_v2_64", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasSdot_v2_64", "oneapi::mkl::blas::column_major::dot"},
      {"cublasDdot_v2_64", "oneapi::mkl::blas::column_major::dot"},
      {"cublasCdotu_v2_64", "oneapi::mkl::blas::column_major::dotu"},
      {"cublasCdotc_v2_64", "oneapi::mkl::blas::column_major::dotc"},
      {"cublasZdotu_v2_64", "oneapi::mkl::blas::column_major::dotu"},
      {"cublasZdotc_v2_64", "oneapi::mkl::blas::column_major::dotc"},
      {"cublasSscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasDscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasCscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasCsscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasZscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasZdscal_v2_64", "oneapi::mkl::blas::column_major::scal"},
      {"cublasSaxpy_v2_64", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasDaxpy_v2_64", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasCaxpy_v2_64", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasZaxpy_v2_64", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasScopy_v2_64", "oneapi::mkl::blas::column_major::copy"},
      {"cublasDcopy_v2_64", "oneapi::mkl::blas::column_major::copy"},
      {"cublasCcopy_v2_64", "oneapi::mkl::blas::column_major::copy"},
      {"cublasZcopy_v2_64", "oneapi::mkl::blas::column_major::copy"},
      {"cublasSswap_v2_64", "oneapi::mkl::blas::column_major::swap"},
      {"cublasDswap_v2_64", "oneapi::mkl::blas::column_major::swap"},
      {"cublasCswap_v2_64", "oneapi::mkl::blas::column_major::swap"},
      {"cublasZswap_v2_64", "oneapi::mkl::blas::column_major::swap"},
      {"cublasSasum_v2_64", "oneapi::mkl::blas::column_major::asum"},
      {"cublasDasum_v2_64", "oneapi::mkl::blas::column_major::asum"},
      {"cublasScasum_v2_64", "oneapi::mkl::blas::column_major::asum"},
      {"cublasDzasum_v2_64", "oneapi::mkl::blas::column_major::asum"},
      {"cublasSnrm2_v2", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasDnrm2_v2", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasScnrm2_v2", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasDznrm2_v2", "oneapi::mkl::blas::column_major::nrm2"},
      {"cublasSdot_v2", "oneapi::mkl::blas::column_major::dot"},
      {"cublasDdot_v2", "oneapi::mkl::blas::column_major::dot"},
      {"cublasCdotu_v2", "oneapi::mkl::blas::column_major::dotu"},
      {"cublasCdotc_v2", "oneapi::mkl::blas::column_major::dotc"},
      {"cublasZdotu_v2", "oneapi::mkl::blas::column_major::dotu"},
      {"cublasZdotc_v2", "oneapi::mkl::blas::column_major::dotc"},
      {"cublasSscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasDscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasCscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasCsscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasZscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasZdscal_v2", "oneapi::mkl::blas::column_major::scal"},
      {"cublasSaxpy_v2", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasDaxpy_v2", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasCaxpy_v2", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasZaxpy_v2", "oneapi::mkl::blas::column_major::axpy"},
      {"cublasScopy_v2", "oneapi::mkl::blas::column_major::copy"},
      {"cublasDcopy_v2", "oneapi::mkl::blas::column_major::copy"},
      {"cublasCcopy_v2", "oneapi::mkl::blas::column_major::copy"},
      {"cublasZcopy_v2", "oneapi::mkl::blas::column_major::copy"},
      {"cublasSswap_v2", "oneapi::mkl::blas::column_major::swap"},
      {"cublasDswap_v2", "oneapi::mkl::blas::column_major::swap"},
      {"cublasCswap_v2", "oneapi::mkl::blas::column_major::swap"},
      {"cublasZswap_v2", "oneapi::mkl::blas::column_major::swap"},
      {"cublasSasum_v2", "oneapi::mkl::blas::column_major::asum"},
      {"cublasDasum_v2", "oneapi::mkl::blas::column_major::asum"},
      {"cublasScasum_v2", "oneapi::mkl::blas::column_major::asum"},
      {"cublasDzasum_v2", "oneapi::mkl::blas::column_major::asum"},
      {"cublasSrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasDrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasCrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasCsrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasZrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasZdrot_v2", "oneapi::mkl::blas::column_major::rot"},
      {"cublasSrotg_v2", "oneapi::mkl::blas::column_major::rotg"},
      {"cublasDrotg_v2", "oneapi::mkl::blas::column_major::rotg"},
      {"cublasCrotg_v2", "oneapi::mkl::blas::column_major::rotg"},
      {"cublasZrotg_v2", "oneapi::mkl::blas::column_major::rotg"},
      {"cublasSrotm_v2", "oneapi::mkl::blas::column_major::rotm"},
      {"cublasDrotm_v2", "oneapi::mkl::blas::column_major::rotm"},
      {"cublasSrotmg_v2", "oneapi::mkl::blas::column_major::rotmg"},
      {"cublasDrotmg_v2", "oneapi::mkl::blas::column_major::rotmg"},
      {"cublasSrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasDrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasCrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasCsrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasZrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasZdrot_v2_64", "oneapi::mkl::blas::column_major::rot"},
      {"cublasSrotm_v2_64", "oneapi::mkl::blas::column_major::rotm"},
      {"cublasDrotm_v2_64", "oneapi::mkl::blas::column_major::rotm"},
      {"cublasSgemv_v2", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasDgemv_v2", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasCgemv_v2", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasZgemv_v2", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasSgbmv_v2", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasDgbmv_v2", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasCgbmv_v2", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasZgbmv_v2", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasStrmv_v2", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasDtrmv_v2", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasCtrmv_v2", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasZtrmv_v2", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasStbmv_v2", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasDtbmv_v2", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasCtbmv_v2", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasZtbmv_v2", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasStpmv_v2", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasDtpmv_v2", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasCtpmv_v2", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasZtpmv_v2", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasStrsv_v2", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasDtrsv_v2", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasCtrsv_v2", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasZtrsv_v2", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasStpsv_v2", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasDtpsv_v2", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasCtpsv_v2", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasZtpsv_v2", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasSgemv_v2_64", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasDgemv_v2_64", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasCgemv_v2_64", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasZgemv_v2_64", "oneapi::mkl::blas::column_major::gemv"},
      {"cublasSgbmv_v2_64", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasDgbmv_v2_64", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasCgbmv_v2_64", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasZgbmv_v2_64", "oneapi::mkl::blas::column_major::gbmv"},
      {"cublasStrmv_v2_64", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasDtrmv_v2_64", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasCtrmv_v2_64", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasZtrmv_v2_64", "oneapi::mkl::blas::column_major::trmv"},
      {"cublasStbmv_v2_64", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasDtbmv_v2_64", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasCtbmv_v2_64", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasZtbmv_v2_64", "oneapi::mkl::blas::column_major::tbmv"},
      {"cublasStpmv_v2_64", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasDtpmv_v2_64", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasCtpmv_v2_64", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasZtpmv_v2_64", "oneapi::mkl::blas::column_major::tpmv"},
      {"cublasStrsv_v2_64", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasDtrsv_v2_64", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasCtrsv_v2_64", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasZtrsv_v2_64", "oneapi::mkl::blas::column_major::trsv"},
      {"cublasStpsv_v2_64", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasDtpsv_v2_64", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasCtpsv_v2_64", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasZtpsv_v2_64", "oneapi::mkl::blas::column_major::tpsv"},
      {"cublasStbsv_v2", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasDtbsv_v2", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasCtbsv_v2", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasZtbsv_v2", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasSsymv_v2", "oneapi::mkl::blas::column_major::symv"},
      {"cublasDsymv_v2", "oneapi::mkl::blas::column_major::symv"},
      {"cublasCsymv_v2", "oneapi::mkl::blas::column_major::symv"},
      {"cublasZsymv_v2", "oneapi::mkl::blas::column_major::symv"},
      {"cublasChemv_v2", "oneapi::mkl::blas::column_major::hemv"},
      {"cublasZhemv_v2", "oneapi::mkl::blas::column_major::hemv"},
      {"cublasSsbmv_v2", "oneapi::mkl::blas::column_major::sbmv"},
      {"cublasDsbmv_v2", "oneapi::mkl::blas::column_major::sbmv"},
      {"cublasChbmv_v2", "oneapi::mkl::blas::column_major::hbmv"},
      {"cublasZhbmv_v2", "oneapi::mkl::blas::column_major::hbmv"},
      {"cublasSspmv_v2", "oneapi::mkl::blas::column_major::spmv"},
      {"cublasDspmv_v2", "oneapi::mkl::blas::column_major::spmv"},
      {"cublasChpmv_v2", "oneapi::mkl::blas::column_major::hpmv"},
      {"cublasZhpmv_v2", "oneapi::mkl::blas::column_major::hpmv"},
      {"cublasStbsv_v2_64", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasDtbsv_v2_64", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasCtbsv_v2_64", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasZtbsv_v2_64", "oneapi::mkl::blas::column_major::tbsv"},
      {"cublasSsymv_v2_64", "oneapi::mkl::blas::column_major::symv"},
      {"cublasDsymv_v2_64", "oneapi::mkl::blas::column_major::symv"},
      {"cublasCsymv_v2_64", "oneapi::mkl::blas::column_major::symv"},
      {"cublasZsymv_v2_64", "oneapi::mkl::blas::column_major::symv"},
      {"cublasChemv_v2_64", "oneapi::mkl::blas::column_major::hemv"},
      {"cublasZhemv_v2_64", "oneapi::mkl::blas::column_major::hemv"},
      {"cublasSsbmv_v2_64", "oneapi::mkl::blas::column_major::sbmv"},
      {"cublasDsbmv_v2_64", "oneapi::mkl::blas::column_major::sbmv"},
      {"cublasChbmv_v2_64", "oneapi::mkl::blas::column_major::hbmv"},
      {"cublasZhbmv_v2_64", "oneapi::mkl::blas::column_major::hbmv"},
      {"cublasSspmv_v2_64", "oneapi::mkl::blas::column_major::spmv"},
      {"cublasDspmv_v2_64", "oneapi::mkl::blas::column_major::spmv"},
      {"cublasChpmv_v2_64", "oneapi::mkl::blas::column_major::hpmv"},
      {"cublasZhpmv_v2_64", "oneapi::mkl::blas::column_major::hpmv"},
      {"cublasSger_v2", "oneapi::mkl::blas::column_major::ger"},
      {"cublasDger_v2", "oneapi::mkl::blas::column_major::ger"},
      {"cublasCgeru_v2", "oneapi::mkl::blas::column_major::geru"},
      {"cublasCgerc_v2", "oneapi::mkl::blas::column_major::gerc"},
      {"cublasZgeru_v2", "oneapi::mkl::blas::column_major::geru"},
      {"cublasZgerc_v2", "oneapi::mkl::blas::column_major::gerc"},
      {"cublasSsyr_v2", "oneapi::mkl::blas::column_major::syr"},
      {"cublasDsyr_v2", "oneapi::mkl::blas::column_major::syr"},
      {"cublasCsyr_v2", "oneapi::mkl::blas::column_major::syr"},
      {"cublasZsyr_v2", "oneapi::mkl::blas::column_major::syr"},
      {"cublasCher_v2", "oneapi::mkl::blas::column_major::her"},
      {"cublasZher_v2", "oneapi::mkl::blas::column_major::her"},
      {"cublasSspr_v2", "oneapi::mkl::blas::column_major::spr"},
      {"cublasDspr_v2", "oneapi::mkl::blas::column_major::spr"},
      {"cublasChpr_v2", "oneapi::mkl::blas::column_major::hpr"},
      {"cublasZhpr_v2", "oneapi::mkl::blas::column_major::hpr"},
      {"cublasSsyr2_v2", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasDsyr2_v2", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasCsyr2_v2", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasZsyr2_v2", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasCher2_v2", "oneapi::mkl::blas::column_major::her2"},
      {"cublasZher2_v2", "oneapi::mkl::blas::column_major::her2"},
      {"cublasSspr2_v2", "oneapi::mkl::blas::column_major::spr2"},
      {"cublasDspr2_v2", "oneapi::mkl::blas::column_major::spr2"},
      {"cublasChpr2_v2", "oneapi::mkl::blas::column_major::hpr2"},
      {"cublasZhpr2_v2", "oneapi::mkl::blas::column_major::hpr2"},
      {"cublasSger_v2_64", "oneapi::mkl::blas::column_major::ger"},
      {"cublasDger_v2_64", "oneapi::mkl::blas::column_major::ger"},
      {"cublasCgeru_v2_64", "oneapi::mkl::blas::column_major::geru"},
      {"cublasCgerc_v2_64", "oneapi::mkl::blas::column_major::gerc"},
      {"cublasZgeru_v2_64", "oneapi::mkl::blas::column_major::geru"},
      {"cublasZgerc_v2_64", "oneapi::mkl::blas::column_major::gerc"},
      {"cublasSsyr_v2_64", "oneapi::mkl::blas::column_major::syr"},
      {"cublasDsyr_v2_64", "oneapi::mkl::blas::column_major::syr"},
      {"cublasCsyr_v2_64", "oneapi::mkl::blas::column_major::syr"},
      {"cublasZsyr_v2_64", "oneapi::mkl::blas::column_major::syr"},
      {"cublasCher_v2_64", "oneapi::mkl::blas::column_major::her"},
      {"cublasZher_v2_64", "oneapi::mkl::blas::column_major::her"},
      {"cublasSspr_v2_64", "oneapi::mkl::blas::column_major::spr"},
      {"cublasDspr_v2_64", "oneapi::mkl::blas::column_major::spr"},
      {"cublasChpr_v2_64", "oneapi::mkl::blas::column_major::hpr"},
      {"cublasZhpr_v2_64", "oneapi::mkl::blas::column_major::hpr"},
      {"cublasSsyr2_v2_64", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasDsyr2_v2_64", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasCsyr2_v2_64", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasZsyr2_v2_64", "oneapi::mkl::blas::column_major::syr2"},
      {"cublasCher2_v2_64", "oneapi::mkl::blas::column_major::her2"},
      {"cublasZher2_v2_64", "oneapi::mkl::blas::column_major::her2"},
      {"cublasSspr2_v2_64", "oneapi::mkl::blas::column_major::spr2"},
      {"cublasDspr2_v2_64", "oneapi::mkl::blas::column_major::spr2"},
      {"cublasChpr2_v2_64", "oneapi::mkl::blas::column_major::hpr2"},
      {"cublasZhpr2_v2_64", "oneapi::mkl::blas::column_major::hpr2"},
      {"cublasCgemm3m_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasZgemm3m_64", "oneapi::mkl::blas::column_major::gemm"},
      {"cublasSsyrkx_64", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasDsyrkx_64", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasCsyrkx_64", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasZsyrkx_64", getLibraryHelperNamespace() + "blas::syrk"},
      {"cublasCherkx_64", getLibraryHelperNamespace() + "blas::herk"},
      {"cublasZherkx_64", getLibraryHelperNamespace() + "blas::herk"},
      {"cublasStrmm_v2_64", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasDtrmm_v2_64", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasCtrmm_v2_64", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasZtrmm_v2_64", getLibraryHelperNamespace() + "blas::trmm"},
      {"cublasCopyEx", getLibraryHelperNamespace() + "blas::copy"},
      {"cublasSwapEx", getLibraryHelperNamespace() + "blas::swap"},
      {"cublasIamaxEx", getLibraryHelperNamespace() + "blas::iamax"},
      {"cublasIaminEx", getLibraryHelperNamespace() + "blas::iamin"},
      {"cublasAsumEx", getLibraryHelperNamespace() + "blas::asum"},
      {"cublasRotmEx", getLibraryHelperNamespace() + "blas::rotm"},
      {"cublasCopyEx_64", getLibraryHelperNamespace() + "blas::copy"},
      {"cublasSwapEx_64", getLibraryHelperNamespace() + "blas::swap"},
      {"cublasIamaxEx_64", getLibraryHelperNamespace() + "blas::iamax"},
      {"cublasIaminEx_64", getLibraryHelperNamespace() + "blas::iamin"},
      {"cublasAsumEx_64", getLibraryHelperNamespace() + "blas::asum"},
      {"cublasRotmEx_64", getLibraryHelperNamespace() + "blas::rotm"},
      {"cublasSgemmEx_64", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasCgemmEx_64", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasCgemm3mEx_64", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasGemmEx_64", getLibraryHelperNamespace() + "blas::gemm"},
      {"cublasCsyrkEx", getLibraryHelperNamespace() + "blas::syherk<false>"},
      {"cublasCsyrk3mEx", getLibraryHelperNamespace() + "blas::syherk<false>"},
      {"cublasCherkEx", getLibraryHelperNamespace() + "blas::syherk<true>"},
      {"cublasCherk3mEx", getLibraryHelperNamespace() + "blas::syherk<true>"},
      {"cublasCsyrkEx_64", getLibraryHelperNamespace() + "blas::syherk<false>"},
      {"cublasCsyrk3mEx_64",
       getLibraryHelperNamespace() + "blas::syherk<false>"},
      {"cublasCherkEx_64", getLibraryHelperNamespace() + "blas::syherk<true>"},
      {"cublasCherk3mEx_64",
       getLibraryHelperNamespace() + "blas::syherk<true>"},
      // cublasLt
      {"cublasLtCreate", "new " + getLibraryHelperNamespace() +
                             "blas_gemm::experimental::descriptor"},
      {"cublasLtDestroy", "delete " + getLibraryHelperNamespace() +
                              "blas_gemm::experimental::descriptor"},
      {"cublasLtMatmulDescCreate",
       "new " + getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t"},
      {"cublasLtMatmulDescDestroy",
       "delete " + getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t"},
      {"cublasLtMatmulDescSetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::set_attribute"},
      {"cublasLtMatmulDescGetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matmul_desc_t::get_attribute"},
      {"cublasLtMatrixLayoutCreate",
       "new " + getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t"},
      {"cublasLtMatrixLayoutDestroy",
       "delete " + getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t"},
      {"cublasLtMatrixLayoutSetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::set_attribute"},
      {"cublasLtMatrixLayoutGetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_layout_t::get_attribute"},
      {"cublasLtMatmul",
       getLibraryHelperNamespace() + "blas_gemm::experimental::matmul"},
      {"cublasLtMatmulPreferenceCreate", ""},
      {"cublasLtMatmulPreferenceDestroy", ""},
      {"cublasLtMatmulPreferenceSetAttribute", ""},
      {"cublasLtMatmulPreferenceGetAttribute", ""},
      {"cublasLtMatmulAlgoGetHeuristic", ""},
      {"cublasLtMatrixTransformDescCreate",
       "new " + getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t"},
      {"cublasLtMatrixTransformDescDestroy",
       "delete" + getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t"},
      {"cublasLtMatrixTransformDescSetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t::set_attribute"},
      {"cublasLtMatrixTransformDescGetAttribute",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::transform_desc_t::get_attribute"},
      {"cublasLtMatrixTransform",
       getLibraryHelperNamespace() +
           "blas_gemm::experimental::matrix_transform"},
      {"cublasLtGetVersion", getLibraryHelperNamespace() + "dnnl::get_version"},
  };

  SOLVERAPIWithRewriter = {"cusolverDnSetAdvOptions",
                           "cusolverDnSetStream",
                           "cusolverDnGetStream",
                           "cusolverDnCreateParams",
                           "cusolverDnDestroyParams",
                           "cusolverDnSpotrfBatched",
                           "cusolverDnDpotrfBatched",
                           "cusolverDnCpotrfBatched",
                           "cusolverDnZpotrfBatched",
                           "cusolverDnSpotrsBatched",
                           "cusolverDnDpotrsBatched",
                           "cusolverDnCpotrsBatched",
                           "cusolverDnZpotrsBatched",
                           "cusolverDnSsygvd",
                           "cusolverDnDsygvd",
                           "cusolverDnSsygvd_bufferSize",
                           "cusolverDnDsygvd_bufferSize",
                           "cusolverDnChegvd",
                           "cusolverDnZhegvd",
                           "cusolverDnChegvd_bufferSize",
                           "cusolverDnZhegvd_bufferSize",
                           "cusolverDnXgetrf",
                           "cusolverDnXgetrf_bufferSize",
                           "cusolverDnXgetrs",
                           "cusolverDnXgeqrf",
                           "cusolverDnXgeqrf_bufferSize",
                           "cusolverDnGetrf",
                           "cusolverDnGetrf_bufferSize",
                           "cusolverDnGetrs",
                           "cusolverDnGeqrf",
                           "cusolverDnGeqrf_bufferSize",
                           "cusolverDnCreateGesvdjInfo",
                           "cusolverDnDestroyGesvdjInfo",
                           "cusolverDnSgesvdj_bufferSize",
                           "cusolverDnDgesvdj_bufferSize",
                           "cusolverDnCgesvdj_bufferSize",
                           "cusolverDnZgesvdj_bufferSize",
                           "cusolverDnXgesvd_bufferSize",
                           "cusolverDnGesvd_bufferSize",
                           "cusolverDnSgesvdj",
                           "cusolverDnDgesvdj",
                           "cusolverDnCgesvdj",
                           "cusolverDnZgesvdj",
                           "cusolverDnXgesvd",
                           "cusolverDnGesvd",
                           "cusolverDnXpotrf_bufferSize",
                           "cusolverDnPotrf_bufferSize",
                           "cusolverDnXpotrf",
                           "cusolverDnPotrf",
                           "cusolverDnXpotrs",
                           "cusolverDnPotrs",
                           "cusolverDnSgeqrf_bufferSize",
                           "cusolverDnDgeqrf_bufferSize",
                           "cusolverDnCgeqrf_bufferSize",
                           "cusolverDnZgeqrf_bufferSize",
                           "cusolverDnSsyevdx",
                           "cusolverDnDsyevdx",
                           "cusolverDnSsyevdx_bufferSize",
                           "cusolverDnDsyevdx_bufferSize",
                           "cusolverDnCheevdx",
                           "cusolverDnZheevdx",
                           "cusolverDnCheevdx_bufferSize",
                           "cusolverDnZheevdx_bufferSize",
                           "cusolverDnSsygvdx",
                           "cusolverDnDsygvdx",
                           "cusolverDnSsygvdx_bufferSize",
                           "cusolverDnDsygvdx_bufferSize",
                           "cusolverDnChegvdx",
                           "cusolverDnZhegvdx",
                           "cusolverDnChegvdx_bufferSize",
                           "cusolverDnZhegvdx_bufferSize",
                           "cusolverDnSsygvj",
                           "cusolverDnDsygvj",
                           "cusolverDnSsygvj_bufferSize",
                           "cusolverDnDsygvj_bufferSize",
                           "cusolverDnChegvj",
                           "cusolverDnZhegvj",
                           "cusolverDnChegvj_bufferSize",
                           "cusolverDnZhegvj_bufferSize",
                           "cusolverDnXsyevdx",
                           "cusolverDnXsyevdx_bufferSize",
                           "cusolverDnSyevdx",
                           "cusolverDnSyevdx_bufferSize",
                           "cusolverDnCreateSyevjInfo",
                           "cusolverDnDestroySyevjInfo",
                           "cusolverDnSsyevj",
                           "cusolverDnDsyevj",
                           "cusolverDnSsyevj_bufferSize",
                           "cusolverDnDsyevj_bufferSize",
                           "cusolverDnCheevj",
                           "cusolverDnZheevj",
                           "cusolverDnCheevj_bufferSize",
                           "cusolverDnZheevj_bufferSize",
                           "cusolverDnXsyevd",
                           "cusolverDnXsyevd_bufferSize",
                           "cusolverDnSyevd",
                           "cusolverDnSyevd_bufferSize",
                           "cusolverDnXtrtri",
                           "cusolverDnXtrtri_bufferSize",
                           "cusolverDnSsyevd_bufferSize",
                           "cusolverDnDsyevd_bufferSize",
                           "cusolverDnCheevd_bufferSize",
                           "cusolverDnZheevd_bufferSize",
                           "cusolverDnSsyevd",
                           "cusolverDnDsyevd",
                           "cusolverDnCheevd",
                           "cusolverDnZheevd"};
}

// Supported vector types
const MapNames::SetTy MapNames::SupportedVectorTypes{SUPPORTEDVECTORTYPENAMES};
const MapNames::SetTy MapNames::VectorTypes2MArray{VECTORTYPE2MARRAYNAMES};

const std::map<std::string, int> MapNames::VectorTypeMigratedTypeSizeMap{
    {"char1", 1},       {"char2", 2},       {"char3", 4},
    {"char4", 4},       {"uchar1", 1},      {"uchar2", 2},
    {"uchar3", 4},      {"uchar4", 4},      {"short1", 2},
    {"short2", 4},      {"short3", 8},      {"short4", 8},
    {"ushort1", 2},     {"ushort2", 4},     {"ushort3", 8},
    {"ushort4", 8},     {"int1", 4},        {"int2", 8},
    {"int3", 16},       {"int4", 16},       {"uint1", 4},
    {"uint2", 8},       {"uint3", 16},      {"uint4", 16},
    {"long1", 8},       {"long2", 16},      {"long3", 32},
    {"long4", 32},      {"ulong1", 8},      {"ulong2", 16},
    {"ulong3", 32},     {"ulong4", 32},     {"longlong1", 8},
    {"longlong2", 16},  {"longlong3", 32},  {"longlong4", 32},
    {"ulonglong1", 8},  {"ulonglong2", 16}, {"ulonglong3", 32},
    {"ulonglong4", 32}, {"float1", 4},      {"float2", 8},
    {"float3", 16},     {"float4", 16},     {"double1", 8},
    {"double2", 16},    {"double3", 32},    {"double4", 32},
    {"__half", 2},      {"__half2", 4},     {"__half_raw", 2}};

const std::map<clang::dpct::KernelArgType, int> MapNames::KernelArgTypeSizeMap{
    {clang::dpct::KernelArgType::KAT_Stream, 208},
    {clang::dpct::KernelArgType::KAT_Texture,
     48 /*32(image accessor) + 16(sampler)*/},
    {clang::dpct::KernelArgType::KAT_Accessor1D, 32},
    {clang::dpct::KernelArgType::KAT_Accessor2D, 56},
    {clang::dpct::KernelArgType::KAT_Accessor3D, 80},
    {clang::dpct::KernelArgType::KAT_Array1D, 8},
    {clang::dpct::KernelArgType::KAT_Array2D, 24},
    {clang::dpct::KernelArgType::KAT_Array3D, 32},
    {clang::dpct::KernelArgType::KAT_Default, 8},
    {clang::dpct::KernelArgType::KAT_MaxParameterSize, 1024}};

int MapNames::getArrayTypeSize(const int Dim) {
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    if (Dim == 2) {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor2D);
    } else if (Dim == 3) {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor3D);
    } else {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor1D);
    }
  } else {
    if (Dim == 2) {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array2D);
    } else if (Dim == 3) {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array3D);
    } else {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array1D);
    }
  }
}

const MapNames::MapTy MapNames::RemovedAPIWarningMessage{
#define ENTRY(APINAME, MSG) {#APINAME, MSG},
#include "APINames_removed.inc"
#undef ENTRY
};

// SOLVER enums mapping
const MapNames::MapTy MapNames::SOLVEREnumsMap{
    {"CUSOLVER_EIG_TYPE_1", "1"},
    {"CUSOLVER_EIG_TYPE_2", "2"},
    {"CUSOLVER_EIG_TYPE_3", "3"},
    {"CUSOLVER_EIG_MODE_NOVECTOR", "oneapi::mkl::job::novec"},
    {"CUSOLVER_EIG_MODE_VECTOR", "oneapi::mkl::job::vec"},
};

const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
    MapNames::LegacyBLASFuncReplInfoMap{
        /*BLAS level 1 */
        {"cublasIsamax",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"float", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamax"}},
        {"cublasIdamax",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"double", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamax"}},
        {"cublasIsamin",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"float", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamin"}},
        {"cublasIdamin",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"double", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamin"}},
        {"cublasSasum",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::asum"}},
        {"cublasDasum",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::asum"}},
        {"cublasSaxpy",
         {std::vector<int>{2, 4}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::axpy"}},
        {"cublasDaxpy",
         {std::vector<int>{2, 4}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::axpy"}},
        {"cublasScopy",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::copy"}},
        {"cublasDcopy",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::copy"}},
        {"cublasSdot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dot"}},
        {"cublasDdot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dot"}},
        {"cublasSnrm2",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::nrm2"}},
        {"cublasDnrm2",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::nrm2"}},
        {"cublasSrot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rot"}},
        {"cublasDrot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rot"}},
        {"cublasCrot",
         {std::vector<int>{1, 3}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::rot"}},
        {"cublasZrot",
         {std::vector<int>{1, 3}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::rot"}},
        {"cublasSrotg",
         {std::vector<int>{0, 1, 2, 3}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotg"}},
        {"cublasDrotg",
         {std::vector<int>{0, 1, 2, 3}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotg"}},
        {"cublasSrotm",
         {std::vector<int>{1, 3, 5}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotm"}},
        {"cublasDrotm",
         {std::vector<int>{1, 3, 5}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotm"}},
        {"cublasSrotmg",
         {std::vector<int>{0, 1, 2, 4}, std::vector<int>{3},
          std::vector<std::string>{"float", "float", "float", "float"},
          std::vector<std::string>{"float"}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotmg"}},
        {"cublasDrotmg",
         {std::vector<int>{0, 1, 2, 4}, std::vector<int>{3},
          std::vector<std::string>{"double", "double", "double", "double"},
          std::vector<std::string>{"double"}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rotmg"}},
        {"cublasSscal",
         {std::vector<int>{2}, std::vector<int>{},
          std::vector<std::string>{"float"}, std::vector<std::string>{},
          std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::scal"}},
        {"cublasDscal",
         {std::vector<int>{2}, std::vector<int>{},
          std::vector<std::string>{"double"}, std::vector<std::string>{},
          std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::scal"}},
        {"cublasSswap",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::swap"}},
        {"cublasDswap",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::swap"}},
        /*BLAS level 2 */
        {"cublasSgbmv",
         {std::vector<int>{6, 8, 11}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gbmv"}},
        {"cublasDgbmv",
         {std::vector<int>{6, 8, 11}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gbmv"}},
        {"cublasSgemv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemv"}},
        {"cublasDgemv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemv"}},
        {"cublasSger",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::ger"}},
        {"cublasDger",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::ger"}},
        {"cublasSsbmv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::sbmv"}},
        {"cublasDsbmv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::sbmv"}},
        {"cublasSspmv",
         {std::vector<int>{3, 4, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spmv"}},
        {"cublasDspmv",
         {std::vector<int>{3, 4, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spmv"}},
        {"cublasSspr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spr"}},
        {"cublasDspr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spr"}},
        {"cublasSspr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spr2"}},
        {"cublasDspr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::spr2"}},
        {"cublasSsymv",
         {std::vector<int>{3, 5, 8}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::symv"}},
        {"cublasDsymv",
         {std::vector<int>{3, 5, 8}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::symv"}},
        {"cublasSsyr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr"}},
        {"cublasDsyr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr"}},
        {"cublasSsyr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2"}},
        {"cublasDsyr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2"}},
        {"cublasStbmv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbmv"}},
        {"cublasDtbmv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbmv"}},
        {"cublasStbsv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbsv"}},
        {"cublasDtbsv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbsv"}},
        {"cublasStpmv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpmv"}},
        {"cublasDtpmv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpmv"}},
        {"cublasStpsv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpsv"}},
        {"cublasDtpsv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpsv"}},
        {"cublasStrmv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trmv"}},
        {"cublasDtrmv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trmv"}},
        {"cublasStrsv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trsv"}},
        {"cublasDtrsv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trsv"}},
        /*BLAS level 3 */
        {"cublasSgemm",
         {std::vector<int>{6, 8, 11}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{0, 1}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemm"}},
        {"cublasDgemm",
         {std::vector<int>{6, 8, 11}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{0, 1}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemm"}},
        {"cublasSsymm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::symm"}},
        {"cublasDsymm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::symm"}},
        {"cublasSsyrk",
         {std::vector<int>{5, 8}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syrk"}},
        {"cublasDsyrk",
         {std::vector<int>{5, 8}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syrk"}},
        {"cublasSsyr2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{},
          std::vector<std::string>{"float", "float", "float"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2k"}},
        {"cublasDsyr2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{},
          std::vector<std::string>{"double", "double", "double"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2k"}},
        {"cublasStrsm",
         {std::vector<int>{7, 9}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{2}, 1, 0, 3,
          "oneapi::mkl::blas::column_major::trsm"}},
        {"cublasDtrsm",
         {std::vector<int>{7, 9}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{2}, 1, 0, 3,
          "oneapi::mkl::blas::column_major::trsm"}},
        {"cublasStrmm",
         {std::vector<int>{7, 9}, std::vector<int>{},
          std::vector<std::string>{"float", "float"},
          std::vector<std::string>{}, std::vector<int>{2}, 1, 0, 3,
          "oneapi::mkl::blas::column_major::trmm"}},
        {"cublasDtrmm",
         {std::vector<int>{7, 9}, std::vector<int>{},
          std::vector<std::string>{"double", "double"},
          std::vector<std::string>{}, std::vector<int>{2}, 1, 0, 3,
          "oneapi::mkl::blas::column_major::trmm"}},
        // level 1
        {"cublasIcamax",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamax"}},
        {"cublasIzamax",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamax"}},
        {"cublasIcamin",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamin"}},
        {"cublasIzamin",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>", "int64_t"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::iamin"}},
        {"cublasScasum",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::asum"}},
        {"cublasDzasum",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::asum"}},
        {"cublasCaxpy",
         {std::vector<int>{2, 4}, std::vector<int>{1},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::axpy"}},
        {"cublasZaxpy",
         {std::vector<int>{2, 4}, std::vector<int>{1},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::axpy"}},
        {"cublasCcopy",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::copy"}},
        {"cublasZcopy",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::copy"}},
        {"cublasCdotu",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dotu"}},
        {"cublasCdotc",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dotc"}},
        {"cublasZdotu",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dotu"}},
        {"cublasZdotc",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::dotc"}},
        {"cublasScnrm2",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>", "float"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::nrm2"}},
        {"cublasDznrm2",
         {std::vector<int>{1}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>", "double"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::nrm2"}},
        {"cublasCsrot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rot"}},
        {"cublasZdrot",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::rot"}},
        {"cublasCscal",
         {std::vector<int>{2}, std::vector<int>{1},
          std::vector<std::string>{"std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::scal"}},
        {"cublasZscal",
         {std::vector<int>{2}, std::vector<int>{1},
          std::vector<std::string>{"std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::scal"}},
        {"cublasCsscal",
         {std::vector<int>{2}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::scal"}},
        {"cublasZdscal",
         {std::vector<int>{2}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::scal"}},
        {"cublasCswap",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::swap"}},
        {"cublasZswap",
         {std::vector<int>{1, 3}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::swap"}},
        /*BLAS level 2 */
        {"cublasCgbmv",
         {std::vector<int>{6, 8, 11}, std::vector<int>{5, 10},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gbmv"}},
        {"cublasZgbmv",
         {std::vector<int>{6, 8, 11}, std::vector<int>{5, 10},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gbmv"}},
        {"cublasCgemv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{3, 8},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemv"}},
        {"cublasZgemv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{3, 8},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{0}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemv"}},
        {"cublasCgeru",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::geru"}},
        {"cublasCgerc",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::gerc"}},
        {"cublasZgeru",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::geru"}},
        {"cublasZgerc",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          -1, -1, -1, "oneapi::mkl::blas::column_major::gerc"}},
        {"cublasCtbmv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbmv"}},
        {"cublasZtbmv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbmv"}},
        {"cublasCtbsv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbsv"}},
        {"cublasZtbsv",
         {std::vector<int>{5, 7}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tbsv"}},
        {"cublasCtpmv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpmv"}},
        {"cublasZtpmv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpmv"}},
        {"cublasCtpsv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpsv"}},
        {"cublasZtpsv",
         {std::vector<int>{4, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::tpsv"}},
        {"cublasCtrmv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trmv"}},
        {"cublasZtrmv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trmv"}},
        {"cublasCtrsv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trsv"}},
        {"cublasZtrsv",
         {std::vector<int>{4, 6}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, 2,
          "oneapi::mkl::blas::column_major::trsv"}},
        {"cublasChemv",
         {std::vector<int>{3, 5, 8}, std::vector<int>{2, 7},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hemv"}},
        {"cublasZhemv",
         {std::vector<int>{3, 5, 8}, std::vector<int>{2, 7},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hemv"}},
        {"cublasChbmv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{3, 8},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hbmv"}},
        {"cublasZhbmv",
         {std::vector<int>{4, 6, 9}, std::vector<int>{3, 8},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hbmv"}},
        {"cublasChpmv",
         {std::vector<int>{3, 4, 7}, std::vector<int>{2, 6},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hpmv"}},
        {"cublasZhpmv",
         {std::vector<int>{3, 4, 7}, std::vector<int>{2, 6},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hpmv"}},
        {"cublasCher",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::her"}},
        {"cublasZher",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::her"}},
        {"cublasCher2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          0, -1, -1, "oneapi::mkl::blas::column_major::her2"}},
        {"cublasZher2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          0, -1, -1, "oneapi::mkl::blas::column_major::her2"}},
        {"cublasChpr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hpr"}},
        {"cublasZhpr",
         {std::vector<int>{3, 5}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::hpr"}},
        {"cublasChpr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{},
          0, -1, -1, "oneapi::mkl::blas::column_major::hpr2"}},
        {"cublasZhpr2",
         {std::vector<int>{3, 5, 7}, std::vector<int>{2},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{},
          0, -1, -1, "oneapi::mkl::blas::column_major::hpr2"}},
        /*BLAS level 3 */
        {"cublasCgemm",
         {std::vector<int>{6, 8, 11}, std::vector<int>{5, 10},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{0, 1}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemm"}},
        {"cublasZgemm",
         {std::vector<int>{6, 8, 11}, std::vector<int>{5, 10},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{0, 1}, -1, -1, -1,
          "oneapi::mkl::blas::column_major::gemm"}},
        {"cublasCsyrk",
         {std::vector<int>{5, 8}, std::vector<int>{4, 7},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syrk"}},
        {"cublasZsyrk",
         {std::vector<int>{5, 8}, std::vector<int>{4, 7},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syrk"}},
        {"cublasCherk",
         {std::vector<int>{5, 8}, std::vector<int>{},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::herk"}},
        {"cublasZherk",
         {std::vector<int>{5, 8}, std::vector<int>{},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{}, std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::herk"}},
        {"cublasCsyr2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2k"}},
        {"cublasZsyr2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{1}, 0, -1, -1,
          "oneapi::mkl::blas::column_major::syr2k"}},
        {"cublasCher2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{1},
          0, -1, -1, "oneapi::mkl::blas::column_major::her2k"}},
        {"cublasZher2k",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{1},
          0, -1, -1, "oneapi::mkl::blas::column_major::her2k"}},
        {"cublasCsymm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::symm"}},
        {"cublasZsymm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::symm"}},
        {"cublasChemm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<float>", "std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::hemm"}},
        {"cublasZhemm",
         {std::vector<int>{5, 7, 10}, std::vector<int>{4, 9},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<int>{}, 1, 0, -1,
          "oneapi::mkl::blas::column_major::hemm"}},
        {"cublasCtrsm",
         {std::vector<int>{7, 9}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{2},
          1, 0, 3, "oneapi::mkl::blas::column_major::trsm"}},
        {"cublasZtrsm",
         {std::vector<int>{7, 9}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{2},
          1, 0, 3, "oneapi::mkl::blas::column_major::trsm"}},
        {"cublasCtrmm",
         {std::vector<int>{7, 9}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<float>",
                                   "std::complex<float>"},
          std::vector<std::string>{"std::complex<float>"}, std::vector<int>{2},
          1, 0, 3, "oneapi::mkl::blas::column_major::trmm"}},
        {"cublasZtrmm",
         {std::vector<int>{7, 9}, std::vector<int>{6},
          std::vector<std::string>{"std::complex<double>",
                                   "std::complex<double>"},
          std::vector<std::string>{"std::complex<double>"}, std::vector<int>{2},
          1, 0, 3, "oneapi::mkl::blas::column_major::trmm"}},
    };

// This map is only used for non-usm.
const std::map<std::string, std::map<int, std::string>>
    MapNames::MaySyncBLASFuncWithMultiArgs{
        {"cublasSrotg_v2",
         {{1, "float"}, {2, "float"}, {3, "float"}, {4, "float"}}},
        {"cublasDrotg_v2",
         {{1, "double"}, {2, "double"}, {3, "double"}, {4, "double"}}},
        {"cublasCrotg_v2",
         {{1, "std::complex<float>"},
          {2, "std::complex<float>"},
          {3, "float"},
          {4, "std::complex<float>"}}},
        {"cublasZrotg_v2",
         {{1, "std::complex<double>"},
          {2, "std::complex<double>"},
          {3, "double"},
          {4, "std::complex<double>"}}},
        {"cublasSrotmg_v2",
         {{1, "float"}, {2, "float"}, {3, "float"}, {5, "float"}}},
        {"cublasDrotmg_v2",
         {{1, "double"}, {2, "double"}, {3, "double"}, {5, "double"}}}};

// SOLVER functions names and parameters replacements information mapping
const std::map<std::string, MapNames::SOLVERFuncReplInfo>
    MapNames::SOLVERFuncReplInfoMap{
        {"cusolverDnSpotrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<float>")},
        {"cusolverDnDpotrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<double>")},
        {"cusolverDnCpotrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZpotrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgetrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<float>")},
        {"cusolverDnDgetrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<double>")},
        {"cusolverDnCgetrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgetrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgeqrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<float>")},
        {"cusolverDnDgeqrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<double>")},
        {"cusolverDnCgeqrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgeqrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSormqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormqr_scratchpad_size<float>")},
        {"cusolverDnDormqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormqr_scratchpad_size<double>")},
        {"cusolverDnCunmqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZunmqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::orgqr_scratchpad_size<float>")},
        {"cusolverDnDorgqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::orgqr_scratchpad_size<double>")},
        {"cusolverDnCungqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungqr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSsytrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::sytrd_scratchpad_size<float>")},
        {"cusolverDnDsytrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::sytrd_scratchpad_size<double>")},
        {"cusolverDnChetrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZhetrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSsytrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<float>")},
        {"cusolverDnDsytrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<double>")},
        {"cusolverDnCsytrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZsytrf_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgebrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<float>")},
        {"cusolverDnDgebrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<double>")},
        {"cusolverDnCgebrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgebrd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgbr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr_scratchpad_size<float>")},
        {"cusolverDnDorgbr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr_scratchpad_size<double>")},
        {"cusolverDnCungbr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungbr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSormtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormtr_scratchpad_size<float>")},
        {"cusolverDnDormtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormtr_scratchpad_size<double>")},
        {"cusolverDnCunmtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZunmtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::orgtr_scratchpad_size<float>")},
        {"cusolverDnDorgtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::orgtr_scratchpad_size<double>")},
        {"cusolverDnCungtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungtr_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgesvd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnCopyRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
             std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
             std::vector<int>{1, 1}, std::vector<bool>{false, false},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             std::vector<std::string>{"job_ct_mkl_jobu", "job_ct_mkl_jobvt"},
             "oneapi::mkl::lapack::gesvd_scratchpad_size<float>")},
        {"cusolverDnDgesvd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnCopyRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
             std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
             std::vector<int>{1, 1}, std::vector<bool>{false, false},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             std::vector<std::string>{"job_ct_mkl_jobu", "job_ct_mkl_jobvt"},
             "oneapi::mkl::lapack::gesvd_scratchpad_size<double>")},
        {"cusolverDnCgesvd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnCopyRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
             std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
             std::vector<int>{1, 1}, std::vector<bool>{false, false},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             std::vector<std::string>{"job_ct_mkl_jobu", "job_ct_mkl_jobvt"},
             "oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgesvd_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnCopyRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
             std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
             std::vector<int>{1, 1}, std::vector<bool>{false, false},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             std::vector<std::string>{"job_ct_mkl_jobu", "job_ct_mkl_jobvt"},
             "oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSpotri_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<float>")},
        {"cusolverDnDpotri_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<double>")},
        {"cusolverDnCpotri_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZpotri_bufferSize",
         MapNames::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSpotrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5}, std::vector<std::string>{"float", "float"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnDpotrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"double", "double"}, std::vector<int>{7},
             "oneapi::mkl::lapack::potrf")},
        {"cusolverDnCpotrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnZpotrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnSpotrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6}, std::vector<std::string>{"float", "float"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<float>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnDpotrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"double", "double"}, std::vector<int>{8},
             std::vector<int>{7}, std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<double>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnCpotrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnZpotrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnSpotri",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5}, std::vector<std::string>{"float", "float"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnDpotri",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"double", "double"}, std::vector<int>{7},
             "oneapi::mkl::lapack::potri")},
        {"cusolverDnCpotri",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnZpotri",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnSgetrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<float>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnDgetrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<double>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnCgetrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnZgetrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnSgetrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"float", "int", "float"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<float>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnDgetrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"double", "int", "double"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<double>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnCgetrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<float>", "int",
                                      "std::complex<float>"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnZgetrs",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<double>", "int",
                                      "std::complex<double>"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnSgeqrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "float", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnDgeqrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "double", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnCgeqrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnZgeqrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnSormqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"float", "float", "float", "float"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormqr")},
        {"cusolverDnDormqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "double",
                 "double",
                 "double",
                 "double",
             },
             std::vector<int>{13}, "oneapi::mkl::lapack::ormqr")},
        {"cusolverDnCunmqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<float>", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmqr")},
        {"cusolverDnZunmqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<double>", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmqr")},
        {"cusolverDnSorgqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{9}, "oneapi::mkl::lapack::orgqr")},
        {"cusolverDnDorgqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{9}, "oneapi::mkl::lapack::orgqr")},
        {"cusolverDnCungqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{9}, "oneapi::mkl::lapack::ungqr")},
        {"cusolverDnZungqr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{9}, "oneapi::mkl::lapack::ungqr")},
        {"cusolverDnSsytrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "int", "float"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnDsytrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "int", "double"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnCsytrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>", "int",
                                      "std::complex<float>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnZsytrf",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>", "int",
                                      "std::complex<double>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnSgebrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float", "float"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnDgebrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double", "double"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnCgebrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{
                 "std::complex<float>", "float", "float", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnZgebrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"std::complex<double>", "double",
                                      "double", "std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnSorgbr",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr")},
        {"cusolverDnDorgbr",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr")},
        {"cusolverDnCungbr",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr")},
        {"cusolverDnZungbr",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr")},
        {"cusolverDnSsytrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float"},
             std::vector<int>{10}, "oneapi::mkl::lapack::sytrd")},
        {"cusolverDnDsytrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double"},
             std::vector<int>{10}, "oneapi::mkl::lapack::sytrd")},
        {"cusolverDnChetrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"std::complex<float>", "float", "float",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{10}, "oneapi::mkl::lapack::hetrd")},
        {"cusolverDnZhetrd",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"std::complex<double>", "double",
                                      "double", "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{10}, "oneapi::mkl::lapack::hetrd")},
        {"cusolverDnSormtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"float", "float", "float", "float"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormtr")},
        {"cusolverDnDormtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"double", "double", "double", "double"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormtr")},
        {"cusolverDnCunmtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<float>", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmtr")},
        {"cusolverDnZunmtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<double>", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmtr")},
        {"cusolverDnSorgtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{8}, "oneapi::mkl::lapack::orgtr")},
        {"cusolverDnDorgtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{8}, "oneapi::mkl::lapack::orgtr")},
        {"cusolverDnCungtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::ungtr")},
        {"cusolverDnZungtr",
         MapNames::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::ungtr")},
        {"cusolverDnSgesvd",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnDgesvd",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnCgesvd",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{
                 "std::complex<float>", "float", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnZgesvd",
         MapNames::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{
                 "std::complex<double>", "double", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
    };

const std::map<std::string, std::string> MapNames::RandomGenerateFuncMap{
    {"curandGenerate", {"generate_uniform_bits"}},
    {"curandGenerateLongLong", {"generate_uniform_bits"}},
    {"curandGenerateLogNormal", {"generate_lognormal"}},
    {"curandGenerateLogNormalDouble", {"generate_lognormal"}},
    {"curandGenerateNormal", {"generate_gaussian"}},
    {"curandGenerateNormalDouble", {"generate_gaussian"}},
    {"curandGeneratePoisson", {"generate_poisson"}},
    {"curandGenerateUniform", {"generate_uniform"}},
    {"curandGenerateUniformDouble", {"generate_uniform"}},
};

const std::map<std::string, std::vector<unsigned int>>
    MapNames::FFTPlanAPINeedParenIdxMap{
        {"cufftPlan1d", {1}},
        {"cufftPlan2d", {1, 2}},
        {"cufftPlan3d", {1, 2, 3}},
        {"cufftPlanMany", {2, 3, 4, 6, 7}},
        {"cufftMakePlan1d", {1}},
        {"cufftMakePlan2d", {1, 2}},
        {"cufftMakePlan3d", {1, 2, 3}},
        {"cufftMakePlanMany", {2, 3, 4, 6, 7}},
        {"cufftMakePlanMany64", {2, 3, 4, 6, 7}}};

const MapNames::MapTy MapNames::Dim3MemberNamesMap{
    {"x", "[2]"}, {"y", "[1]"}, {"z", "[0]"},
    // ...
};

const std::map<unsigned, std::string> MapNames::ArrayFlagMap{
    {0, "standard"},
    {1, "array"},
};

std::unordered_map<std::string, MacroMigrationRule> MapNames::MacroRuleMap;

std::unordered_map<std::string, MetaRuleObject &> MapNames::HeaderRuleMap{};

// Files to not preprocess, i.e. ignore #include <file>
const MapNames::SetTy MapNames::ThrustFileExcludeSet{
    "thrust/detail/adjacent_difference.inl",
    "thrust/detail/binary_search.inl",
    "thrust/detail/complex/complex.inl",
    "thrust/detail/copy_if.h",
    "thrust/detail/count.inl",
    "thrust/detail/equal.inl",
    "thrust/detail/pair.inl",
    "thrust/detail/pointer.inl",
    "thrust/detail/sequence.inl",
    "thrust/detail/sort.inl",
    "thrust/detail/temporary_buffer.h"};

// Texture names mapping.
const MapNames::MapTy TextureRule::TextureMemberNames{
    {"addressMode", "addressing_mode"},
    {"filterMode", "filtering_mode"},
    {"normalized", "coordinate_normalization_mode"},
    {"normalizedCoords", "coordinate_normalization_mode"},
    {"channelDesc", "channel"},
    {"Format", "channel_type"},
    {"NumChannels", "channel_num"},
    {"Width", "x"},
    {"Height", "y"},
    {"flags", "coordinate_normalization_mode"},
    {"maxAnisotropy", "max_anisotropy"},
    {"mipmapFilterMode", "mipmap_filtering"},
    {"minMipmapLevelClamp", "min_mipmap_level_clamp"},
    {"maxMipmapLevelClamp", "max_mipmap_level_clamp"},
};

// DeviceProp names mapping.
const MapNames::MapTy DeviceInfoVarRule::PropNamesMap{
    {"clockRate", "max_clock_frequency"},
    {"major", "major_version"},
    {"minor", "minor_version"},
    {"integrated", "integrated"},
    {"warpSize", "max_sub_group_size"},
    {"multiProcessorCount", "max_compute_units"},
    {"maxThreadsPerBlock", "max_work_group_size"},
    {"maxThreadsPerMultiProcessor", "max_work_items_per_compute_unit"},
    {"name", "name"},
    {"totalGlobalMem", "global_mem_size"},
    {"sharedMemPerBlock", "local_mem_size"},
    {"sharedMemPerBlockOptin", "local_mem_size"},
    {"sharedMemPerMultiprocessor", "local_mem_size"},
    {"maxGridSize", "max_nd_range_size"},
    {"maxThreadsDim", "max_work_item_sizes"},
    {"memoryClockRate", "memory_clock_rate"},
    {"memoryBusWidth", "memory_bus_width"},
    {"pciDeviceID", "device_id"},
    {"uuid", "uuid"},
    {"l2CacheSize", "global_mem_cache_size"},
    {"maxTexture1D", "image1d_max"},
    {"maxTexture2D", "image2d_max"},
    {"maxTexture3D", "image3d_max"},
    {"regsPerBlock", "max_register_size_per_work_group"},
    // ...
};

const MapNames::MapTy MapNames::FunctionAttrMap{
    {"CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", "max_work_group_size"},
    {"CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",     "shared_size_bytes /* statically allocated shared memory per work-group in bytes */"},
    {"CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",      "local_size_bytes /* local memory per work-item in bytes */"},
    {"CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",      "const_size_bytes /* user-defined constant kernel memory in bytes */"},
    {"CU_FUNC_ATTRIBUTE_NUM_REGS",              "num_regs /* number of registers for each thread */"},
    // ...
};

// DeviceProp names mapping.
const MapNames::MapTy MapNames::MemberNamesMap{
    {"x", "x()"}, {"y", "y()"}, {"z", "z()"}, {"w", "w()"},
    // ...
};
const MapNames::MapTy MapNames::MArrayMemberNamesMap{
    {"x", "[0]"},
    {"y", "[1]"},
};

const MapNames::SetTy MapNames::HostAllocSet{
    "cudaHostAllocDefault",         "cudaHostAllocMapped",
    "cudaHostAllocPortable",        "cudaHostAllocWriteCombined",
    "CU_MEMHOSTALLOC_PORTABLE",     "CU_MEMHOSTALLOC_DEVICEMAP",
    "CU_MEMHOSTALLOC_WRITECOMBINED"};

// Function Attributes names migration
const MapNames::MapTy KernelFunctionInfoRule::AttributesNamesMap{
    {"maxThreadsPerBlock", "max_work_group_size"},
};

std::map<std::string, bool> MigrationStatistics::MigrationTable{
#define ENTRY(INTERFACENAME, APINAME, VALUE, FLAG, TARGET, COMMENT)            \
  {#APINAME, VALUE},
#define ENTRY_MEMBER_FUNCTION(INTERFACEOBJNAME, OBJNAME, INTERFACENAME,        \
                              APINAME, VALUE, FLAG, TARGET, COMMENT)           \
  {#OBJNAME "::" #APINAME, VALUE},
#include "APINames.inc"
#include "APINames_CUB.inc"
#include "APINames_NCCL.inc"
#include "APINames_NVML.inc"
#include "APINames_NVTX.inc"
#include "APINames_cuBLAS.inc"
#include "APINames_cuDNN.inc"
#include "APINames_cuFFT.inc"
#include "APINames_cuRAND.inc"
#include "APINames_cuSOLVER.inc"
#include "APINames_cuSPARSE.inc"
#include "APINames_cudnn_frontend.inc"
#include "APINames_nvGRAPH.inc"
#include "APINames_nvJPEG.inc"
#include "APINames_thrust.inc"
#include "APINames_wmma.inc"
#undef ENTRY_MEMBER_FUNCTION
#undef ENTRY
};

std::map<std::string, bool> MigrationStatistics::TypeMigrationTable{
#define ENTRY_TYPE(TYPENAME, VALUE, FLAG, TARGET, COMMENT) {#TYPENAME, VALUE},
#include "TypeNames.inc"
#undef ENTRY_TYPE
};

bool MigrationStatistics::IsMigrated(const std::string &APIName) {
  auto Search = MigrationTable.find(APIName);
  if (Search != MigrationTable.end()) {
    return Search->second;
  } else {
#ifdef DPCT_DEBUG_BUILD
    llvm::errs() << "[NOTE] Find new API \"" << APIName
                 << "\" , please update migrated API database.\n";
    ShowStatus(MigrationError);
    dpctExit(MigrationError);
#endif
    return false;
  }
}

std::vector<std::string> MigrationStatistics::GetAllAPINames(void) {
  std::vector<std::string> AllAPINames;
  for (const auto &APIName : MigrationTable) {
    AllAPINames.push_back(APIName.first);
  }

  return AllAPINames;
}
std::map<std::string, bool> &MigrationStatistics::GetTypeTable(void) {
  return TypeMigrationTable;
}

MapNames::MapTy TextureRule::ResourceTypeNames{{"devPtr", "data_ptr"},
                                               {"desc", "channel"},
                                               {"array", "data_ptr"},
                                               {"mipmap", "data_ptr"},
                                               {"width", "x"},
                                               {"height", "y"},
                                               {"pitchInBytes", "pitch"},
                                               {"sizeInBytes", "x"},
                                               {"hArray", "data_ptr"},
                                               {"format", "channel_type"},
                                               {"numChannels", "channel_num"}};

std::vector<MetaRuleObject::PatternRewriter> MapNames::PatternRewriters;

const MapNames::MapTy MemoryDataTypeRule::PitchMemberNames{
    {"pitch", "pitch"}, {"ptr", "data_ptr"}, {"xsize", "x"}, {"ysize", "y"}};
const MapNames::MapTy MemoryDataTypeRule::ExtentMemberNames{
    {"width", "[0]"}, {"height", "[1]"}, {"depth", "[2]"}};

const MapNames::MapTy MemoryDataTypeRule::ArrayDescMemberNames{
    {"Width", "width"},
    {"Height", "height"},
    {"Depth", "depth"},
    {"Format", "channel_type"},
    {"NumChannels", "num_channels"}};

const MapNames::MapTy MemoryDataTypeRule::DirectReplMemberNames{
    // cudaMemcpy3DParms fields.
    {"srcArray", "from.image"},
    {"srcPtr", "from.pitched"},
    {"srcPos", "from.pos"},
    {"dstArray", "to.image"},
    {"dstPtr", "to.pitched"},
    {"dstPos", "to.pos"},
    {"extent", "size"},
    {"kind", "direction"},
    // cudaMemcpy3DPeerParms fields.
    {"srcDevice", "from.dev_id"},
    {"dstDevice", "to.dev_id"},
    // CUDA_MEMCPY2D fields.
    {"Height", "size[1]"},
    {"WidthInBytes", "size_x_in_bytes"},
    {"dstXInBytes", "to.pos_x_in_bytes"},
    {"srcXInBytes", "from.pos_x_in_bytes"},
    {"dstY", "to.pos[1]"},
    {"srcY", "from.pos[1]"},
    // CUDA_MEMCPY3D fields.
    {"Depth", "size[2]"},
    {"dstZ", "to.pos[2]"},
    {"srcZ", "from.pos[2]"},
    // CUDA_MEMCPY3D_PEER fields.
    {"srcContext", "from.dev_id"},
    {"dstContext", "to.dev_id"},
};

const MapNames::MapTy MemoryDataTypeRule::GetSetReplMemberNames{
    // CUDA_MEMCPY2D fields.
    {"dstPitch", "pitch"},
    {"srcPitch", "pitch"},
    {"dstDevice", "data_ptr"},
    {"dstHost", "data_ptr"},
    {"srcDevice", "data_ptr"},
    {"srcHost", "data_ptr"},
    // CUDA_MEMCPY3D fields.
    {"dstHeight", "y"},
    {"srcHeight", "y"},
};

const std::vector<std::string> MemoryDataTypeRule::RemoveMember{
    "dstLOD", "srcLOD", "dstMemoryType", "srcMemoryType", "Flags"};

const std::unordered_set<std::string> MapNames::CooperativeGroupsAPISet{
    "this_thread_block",
    "this_grid",
    "sync",
    "tiled_partition",
    "thread_rank",
    "size",
    "shfl_down",
    "reduce",
    "num_threads",
    "shfl_up",
    "shfl",
    "shfl_xor",
    "meta_group_rank",
    "block_tile_memory",
    "thread_index",
    "group_index",
    "inclusive_scan",
    "exclusive_scan",
    "coalesced_threads",
    "num_blocks",
    "block_rank"};

const std::unordered_map<std::string, HelperFeatureEnum>
    MapNames::SamplingInfoToSetFeatureMap = {
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNames::SamplingInfoToGetFeatureMap = {
        {"addressing_mode", HelperFeatureEnum::device_ext},
        {"filtering_mode", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNames::ImageWrapperBaseToSetFeatureMap = {
        {"sampling_info", HelperFeatureEnum::device_ext},
        {"data", HelperFeatureEnum::device_ext},
        {"channel", HelperFeatureEnum::device_ext},
        {"channel_data_type", HelperFeatureEnum::device_ext},
        {"channel_size", HelperFeatureEnum::device_ext},
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext},
        {"channel_num", HelperFeatureEnum::device_ext},
        {"channel_type", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNames::ImageWrapperBaseToGetFeatureMap = {
        {"sampling_info", HelperFeatureEnum::device_ext},
        {"data", HelperFeatureEnum::device_ext},
        {"channel", HelperFeatureEnum::device_ext},
        {"channel_data_type", HelperFeatureEnum::device_ext},
        {"channel_size", HelperFeatureEnum::device_ext},
        {"addressing_mode", HelperFeatureEnum::device_ext},
        {"filtering_mode", HelperFeatureEnum::device_ext},
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext},
        {"channel_num", HelperFeatureEnum::device_ext},
        {"channel_type", HelperFeatureEnum::device_ext},
        {"sampler", HelperFeatureEnum::device_ext},
};
