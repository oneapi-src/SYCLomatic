//===--------------- GenHelperFunction.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GenHelperFunction.h"
#include "Utility.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <fstream>

namespace clang {
namespace dpct {

const std::string DpctAllContentStr =
#include "clang/DPCT/dpct.hpp.inc"
    ;
const std::string AtomicAllContentStr =
#include "clang/DPCT/atomic.hpp.inc"
    ;
const std::string BindlessImageAllContentStr =
#include "clang/DPCT/bindless_images.hpp.inc"
    ;
const std::string BlasUtilsAllContentStr =
#include "clang/DPCT/blas_utils.hpp.inc"
    ;
const std::string DnnlUtilsAllContentStr =
#include "clang/DPCT/dnnl_utils.hpp.inc"
    ;
const std::string DeviceAllContentStr =
#include "clang/DPCT/device.hpp.inc"
    ;
const std::string DplUtilsAllContentStr =
#include "clang/DPCT/dpl_utils.hpp.inc"
    ;
const std::string GraphAllContentStr =
#include "clang/DPCT/graph.hpp.inc"
    ;
const std::string ImageAllContentStr =
#include "clang/DPCT/image.hpp.inc"
    ;
const std::string KernelAllContentStr =
#include "clang/DPCT/kernel.hpp.inc"
    ;
const std::string MathAllContentStr =
#include "clang/DPCT/math.hpp.inc"
    ;
const std::string MemoryAllContentStr =
#include "clang/DPCT/memory.hpp.inc"
    ;
const std::string UtilAllContentStr =
#include "clang/DPCT/util.hpp.inc"
    ;
const std::string RngUtilsAllContentStr =
#include "clang/DPCT/rng_utils.hpp.inc"
    ;
const std::string LibCommonUtilsAllContentStr =
#include "clang/DPCT/lib_common_utils.hpp.inc"
    ;
const std::string CclUtilsAllContentStr =
#include "clang/DPCT/ccl_utils.hpp.inc"
    ;
const std::string SparseUtilsAllContentStr =
#include "clang/DPCT/sparse_utils.hpp.inc"
    ;
const std::string FftUtilsAllContentStr =
#include "clang/DPCT/fft_utils.hpp.inc"
    ;
const std::string LapackUtilsAllContentStr =
#include "clang/DPCT/lapack_utils.hpp.inc"
    ;
const std::string GroupUtilsAllContentStr =
#include "clang/DPCT/group_utils.hpp.inc"
    ;
const std::string BlasGemmUtilsAllContentStr =
#include "clang/DPCT/blas_gemm_utils.hpp.inc"
    ;
const std::string CompatServiceAllContentStr =
#include "clang/DPCT/compat_service.hpp.inc"
    ;
const std::string DplExtrasAlgorithmAllContentStr =
#include "clang/DPCT/dpl_extras/algorithm.h.inc"
    ;
const std::string DplExtrasFunctionalAllContentStr =
#include "clang/DPCT/dpl_extras/functional.h.inc"
    ;
const std::string DplExtrasIteratorsAllContentStr =
#include "clang/DPCT/dpl_extras/iterators.h.inc"
    ;
const std::string DplExtrasMemoryAllContentStr =
#include "clang/DPCT/dpl_extras/memory.h.inc"
    ;
const std::string DplExtrasNumericAllContentStr =
#include "clang/DPCT/dpl_extras/numeric.h.inc"
    ;
const std::string DplExtrasVectorAllContentStr =
#include "clang/DPCT/dpl_extras/vector.h.inc"
    ;
const std::string DplExtrasDpcppExtensionsAllContentStr =
#include "clang/DPCT/dpl_extras/dpcpp_extensions.h.inc"
    ;
const std::string DplExtrasIteratorAdaptorAllContentStr =
#include "clang/DPCT/dpl_extras/iterator_adaptor.h.inc"
    ;

const std::string CodePinAllContentStr =
#include "clang/DPCT/codepin/codepin.hpp.inc"
    ;

const std::string CodePinSerializationBasicAllContentStr =
#include "clang/DPCT/codepin/serialization/basic.hpp.inc"
    ;

const std::string CmakeAllContentStr =
#include "clang/DPCT/dpct.cmake.inc"
    ;

void genHelperFunction(const clang::tooling::UnifiedPath &OutRoot) {
  if (!llvm::sys::fs::is_directory(OutRoot.getCanonicalPath()))
    createDirectories(OutRoot);
  clang::tooling::UnifiedPath ToPath =
      appendPath(OutRoot.getCanonicalPath().str(), "include");
  if (!llvm::sys::fs::is_directory(ToPath.getCanonicalPath()))
    createDirectories(ToPath);
  ToPath = appendPath(ToPath.getCanonicalPath().str(), "dpct");
  if (!llvm::sys::fs::is_directory(ToPath.getCanonicalPath()))
    createDirectories(ToPath);
  if (!llvm::sys::fs::is_directory(
          appendPath(ToPath.getCanonicalPath().str(), "dpl_extras")))
    createDirectories(
        appendPath(ToPath.getCanonicalPath().str(), "dpl_extras"));
  if (!llvm::sys::fs::is_directory(
          appendPath(ToPath.getCanonicalPath().str(), "codepin")))
    createDirectories(appendPath(ToPath.getCanonicalPath().str(), "codepin"));
  if (!llvm::sys::fs::is_directory(
          appendPath(appendPath(ToPath.getCanonicalPath().str(), "codepin"),
                     "serialization")))
    createDirectories(
        appendPath(appendPath(ToPath.getCanonicalPath().str(), "codepin"),
                   "serialization"));

#define GENERATE_ALL_FILE_CONTENT(VAR_NAME, FOLDER_NAME, FILE_NAME)            \
  {                                                                            \
    std::ofstream VAR_NAME##File(                                              \
        appendPath(appendPath(ToPath.getCanonicalPath().str(), FOLDER_NAME),   \
                   #FILE_NAME),                                                \
        std::ios::out | std::ios::trunc);                                      \
    std::string Code = VAR_NAME##AllContentStr;                                \
    VAR_NAME##File << Code;                                                    \
    VAR_NAME##File.flush();                                                    \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic, ".", atomic.hpp)
  GENERATE_ALL_FILE_CONTENT(BindlessImage, ".", bindless_images.hpp)
  GENERATE_ALL_FILE_CONTENT(BlasUtils, ".", blas_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(Device, ".", device.hpp)
  GENERATE_ALL_FILE_CONTENT(Dpct, ".", dpct.hpp)
  GENERATE_ALL_FILE_CONTENT(DplUtils, ".", dpl_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(DnnlUtils, ".", dnnl_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(Graph, ".", graph.hpp)
  GENERATE_ALL_FILE_CONTENT(Image, ".", image.hpp)
  GENERATE_ALL_FILE_CONTENT(Kernel, ".", kernel.hpp)
  GENERATE_ALL_FILE_CONTENT(Math, ".", math.hpp)
  GENERATE_ALL_FILE_CONTENT(Memory, ".", memory.hpp)
  GENERATE_ALL_FILE_CONTENT(Util, ".", util.hpp)
  GENERATE_ALL_FILE_CONTENT(RngUtils, ".", rng_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(LibCommonUtils, ".", lib_common_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(CclUtils, ".", ccl_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(SparseUtils, ".", sparse_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(FftUtils, ".", fft_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(LapackUtils, ".", lapack_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(GroupUtils, ".", group_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(BlasGemmUtils, ".", blas_gemm_utils.hpp)
  GENERATE_ALL_FILE_CONTENT(CompatService, ".", compat_service.hpp)
  GENERATE_ALL_FILE_CONTENT(CodePin, "codepin", codepin.hpp)
  GENERATE_ALL_FILE_CONTENT(CodePinSerializationBasic, "codepin/serialization",
                            basic.hpp)
  GENERATE_ALL_FILE_CONTENT(DplExtrasAlgorithm, "dpl_extras", algorithm.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasFunctional, "dpl_extras", functional.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasIterators, "dpl_extras", iterators.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasMemory, "dpl_extras", memory.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasNumeric, "dpl_extras", numeric.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasVector, "dpl_extras", vector.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasDpcppExtensions, "dpl_extras",
                            dpcpp_extensions.h)
  GENERATE_ALL_FILE_CONTENT(DplExtrasIteratorAdaptor, "dpl_extras",
                            iterator_adaptor.h)
#undef GENERATE_ALL_FILE_CONTENT
}

void genCmakeHelperFunction(const clang::tooling::UnifiedPath &OutRoot) {
  if (!llvm::sys::fs::is_directory(OutRoot.getCanonicalPath()))
    createDirectories(OutRoot);

  clang::tooling::UnifiedPath ToPath =
      appendPath(OutRoot.getCanonicalPath().str(), ".");

#define GENERATE_ALL_FILE_CONTENT(VAR_NAME, FOLDER_NAME, FILE_NAME)            \
  {                                                                            \
    std::ofstream VAR_NAME##File(                                              \
        appendPath(appendPath(ToPath.getCanonicalPath().str(), FOLDER_NAME),   \
                   #FILE_NAME),                                                \
        std::ios::out | std::ios::trunc);                                      \
    std::string Code = VAR_NAME##AllContentStr;                                \
    VAR_NAME##File << Code;                                                    \
    VAR_NAME##File.flush();                                                    \
  }
  GENERATE_ALL_FILE_CONTENT(Cmake, ".", dpct.cmake)

#undef GENERATE_ALL_FILE_CONTENT
}

} // namespace dpct
} // namespace clang
