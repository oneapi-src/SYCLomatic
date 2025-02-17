//===--------------- MapNames.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/MapNames.h"
#include "AnalysisInfo.h"

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

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

std::unordered_map<std::string, std::shared_ptr<ClassFieldRule>>
    MapNames::ClassFieldMap;

std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
    MapNames::TypeNamesMap;
std::unordered_set<std::string> MapNames::SYCLcompatUnsupportTypes;
std::unordered_map<std::string, std::shared_ptr<EnumNameRule>>
    MapNames::EnumNamesMap;
MapNames::MapTy MapNames::ITFName;

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
      {"cudaExternalMemoryDedicated",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaExternalMemoryDedicated", "0")},
      {"cudaArrayDefault",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaArrayDefault",
                          DpctGlobalInfo::useExtBindlessImages()
                              ? getExpNamespace() + "image_type::standard"
                              : "cudaArrayDefault")},
      {"cudaArrayLayered",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaArrayLayered",
                          DpctGlobalInfo::useExtBindlessImages()
                              ? getExpNamespace() + "image_type::array"
                              : "cudaArrayLayered")},
      {"cudaArrayCubemap",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaArrayCubemap",
                          DpctGlobalInfo::useExtBindlessImages()
                              ? getExpNamespace() + "image_type::cubemap"
                              : "cudaArrayCubemap")},
      {"cudaArraySurfaceLoadStore",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "cudaArraySurfaceLoadStore",
                          DpctGlobalInfo::useExtBindlessImages()
                              ? getExpNamespace() + "image_type::mipmap"
                              : "cudaArraySurfaceLoadStore")},
      {"CubDebug",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CubDebug", MapNames::getCheckErrorMacroName())},
      {"CubDebugExit",
       MacroMigrationRule("dpct_build_in_macro_rule", RulePriority::Fallback,
                          "CubDebugExit", MapNames::getCheckErrorMacroName())},
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
      {"CUjit_target", std::make_shared<TypeNameRule>("int")},
      {"CUjitInputType", std::make_shared<TypeNameRule>("int")},
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
      {"CUstreamCallback",
       std::make_shared<TypeNameRule>(getDpctNamespace() + "queue_callback",
                                      HelperFeatureEnum::device_ext)},
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
      {"thrust::device_ptr", std::make_shared<TypeNameRule>(
                                 getLibraryHelperNamespace() + "device_pointer",
                                 HelperFeatureEnum::device_ext)},
      {"thrust::device_reference",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "device_reference",
                                      HelperFeatureEnum::device_ext)},
      {"thrust::device_vector",
       std::make_shared<TypeNameRule>(getLibraryHelperNamespace() +
                                          "device_vector",
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
      {"cudaSurfaceObject_t",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::unsampled_image_handle"
               : getDpctNamespace() + "image_wrapper_base_p",
           HelperFeatureEnum::device_ext)},
      {"CUsurfObject",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::unsampled_image_handle"
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
      {"CUtexObject",
       std::make_shared<TypeNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getClNamespace() +
                     "ext::oneapi::experimental::sampled_image_handle"
               : getDpctNamespace() + "image_wrapper_base_p",
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
      {"CUmemAllocationProp",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                      "experimental::mem_prop")},
      {"CUmemGenericAllocationHandle",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                      "experimental::physical_mem_ptr")},
      {"CUmemAccessDesc",
       std::make_shared<TypeNameRule>(getDpctNamespace() +
                                      "experimental::mem_access_desc")},
      {"CUmemLocationType", std::make_shared<TypeNameRule>("int")},
      {"CUmemAllocationType", std::make_shared<TypeNameRule>("int")},
      {"CUmemAllocationGranularity_flags",
       std::make_shared<TypeNameRule>(
           getClNamespace() + "ext::oneapi::experimental::granularity_mode")},
      {"CUmemAccess_flags",
       std::make_shared<TypeNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::address_access_mode")},
      {"cudaGraphicsMapFlags", std::make_shared<TypeNameRule>("int")},
      {"cudaGraphicsRegisterFlags", std::make_shared<TypeNameRule>("int")},
      {"cudaExternalMemoryHandleType",
       std::make_shared<TypeNameRule>(getExpNamespace() +
                                      "external_mem_handle_type")},
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
      "cudaSurfaceObject_t",
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

  // Enum constants name mapping.
  MapNames::EnumNamesMap = {
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
      {"CU_MEM_ALLOCATION_TYPE_PINNED", std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_ALLOCATION_TYPE_INVALID", std::make_shared<EnumNameRule>("1")},
      {"CU_MEM_ALLOCATION_TYPE_MAX",
       std::make_shared<EnumNameRule>("0xFFFFFFFF")},
      {"CU_MEM_LOCATION_TYPE_DEVICE", std::make_shared<EnumNameRule>("1")},
      {"CU_MEM_LOCATION_TYPE_INVALID", std::make_shared<EnumNameRule>("0")},
      {"CU_MEM_LOCATION_TYPE_MAX",
       std::make_shared<EnumNameRule>("0xFFFFFFFF")},
      {"CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::address_access_mode::read_write")},
      {"CU_MEM_ACCESS_FLAGS_PROT_NONE",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::address_access_mode::none")},
      {"CU_MEM_ACCESS_FLAGS_PROT_MAX",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::address_access_mode::none")},
      {"CU_MEM_ACCESS_FLAGS_PROT_READ",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::address_access_mode::read")},
      {"CU_MEM_ALLOC_GRANULARITY_RECOMMENDED",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::granularity_mode::recommended")},
      {"CU_MEM_ALLOC_GRANULARITY_MINIMUM",
       std::make_shared<EnumNameRule>(
           getClNamespace() +
           "ext::oneapi::experimental::granularity_mode::minimum")},
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
      {"CU_DEVICE_ATTRIBUTE_MAX_PITCH",
       std::make_shared<EnumNameRule>("get_max_pitch",
                                      HelperFeatureEnum::device_ext)},
      {"CU_CTX_LMEM_RESIZE_TO_MAX", std::make_shared<EnumNameRule>("0")},
      {"CU_CTX_MAP_HOST", std::make_shared<EnumNameRule>("0")},
      {"CU_CTX_SCHED_BLOCKING_SYNC", std::make_shared<EnumNameRule>("0")},
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
      // enum cudaExternalMemoryHandleType
      {"cudaExternalMemoryHandleTypeOpaqueFd",
       std::make_shared<EnumNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getExpNamespace() + "external_mem_handle_type::opaque_fd"
               : "cudaExternalMemoryHandleTypeOpaqueFd")},
      {"cudaExternalMemoryHandleTypeOpaqueWin32",
       std::make_shared<EnumNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getExpNamespace() + "external_mem_handle_type::win32_nt_handle"
               : "cudaExternalMemoryHandleTypeOpaqueWin32")},
      {"cudaExternalMemoryHandleTypeD3D12Resource",
       std::make_shared<EnumNameRule>(
           DpctGlobalInfo::useExtBindlessImages()
               ? getExpNamespace() +
                     "external_mem_handle_type::win32_nt_dx12_resource"
               : "cudaExternalMemoryHandleTypeD3D12Resource")},
      // ...
  };

  ITFName = {
#define ENTRY(INTERFACENAME, APINAME, VALUE, FLAG, TARGET, COMMENT)            \
  {#APINAME, #INTERFACENAME},
#define ENTRY_MEMBER_FUNCTION(INTERFACEOBJNAME, OBJNAME, INTERFACENAME,        \
                              APINAME, VALUE, FLAG, TARGET, COMMENT)           \
  {#OBJNAME "::" #APINAME, #INTERFACEOBJNAME "::" #INTERFACENAME},
#include "SrcAPI/APINames.inc"
#include "SrcAPI/APINames_CUB.inc"
#include "SrcAPI/APINames_NCCL.inc"
#include "SrcAPI/APINames_cuBLAS.inc"
#include "SrcAPI/APINames_cuFFT.inc"
#include "SrcAPI/APINames_cuRAND.inc"
#include "SrcAPI/APINames_cuSOLVER.inc"
#include "SrcAPI/APINames_cuSPARSE.inc"
#include "SrcAPI/APINames_nvGRAPH.inc"
#include "SrcAPI/APINames_nvJPEG.inc"
#include "SrcAPI/APINames_thrust.inc"
#include "SrcAPI/APINames_wmma.inc"
#undef ENTRY_MEMBER_FUNCTION
#undef ENTRY
  };

  ClassFieldMap = {};
}

const MapNames::MapTy MapNames::RemovedAPIWarningMessage{
#define ENTRY(APINAME, MSG) {#APINAME, MSG},
#include "APINames_removed.inc"
#undef ENTRY
};

std::unordered_map<std::string, MacroMigrationRule> MapNames::MacroRuleMap;
std::unordered_map<std::string, MetaRuleObject &> MapNames::HeaderRuleMap{};
std::vector<MetaRuleObject::PatternRewriter> MapNames::PatternRewriters;
std::map<clang::dpct::HelperFuncCatalog, std::string>
    MapNames::CustomHelperFunctionMap;

} // namespace dpct
} // namespace clang
