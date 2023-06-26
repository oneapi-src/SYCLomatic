//===--------------- GenHelperFunction.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GenHelperFunction.h"

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

void replaceAllOccurredStrsInStr(std::string &StrNeedProcess,
                                 const std::string &Pattern,
                                 const std::string &Repl) {
  if (StrNeedProcess.empty() || Pattern.empty()) {
    return;
  }

  size_t PatternLen = Pattern.size();
  size_t ReplLen = Repl.size();
  size_t Offset = 0;
  Offset = StrNeedProcess.find(Pattern, Offset);

  while (Offset != std::string::npos) {
    StrNeedProcess.replace(Offset, PatternLen, Repl);
    Offset = Offset + ReplLen;
    Offset = StrNeedProcess.find(Pattern, Offset);
  }
}

void replaceEndOfLine(std::string &StrNeedProcess) {
#ifdef _WIN64
  replaceAllOccurredStrsInStr(StrNeedProcess, "\n", "\r\n");
#endif
}

void genHelperFunction(const std::string &OutRoot) {
  if (!llvm::sys::fs::is_directory(OutRoot))
    llvm::sys::fs::create_directory(llvm::Twine(OutRoot));
  std::string ToPath = OutRoot + "/include";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(llvm::Twine(ToPath));
  ToPath = ToPath + "/dpct";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(llvm::Twine(ToPath));
  if (!llvm::sys::fs::is_directory(llvm::Twine(ToPath + "/dpl_extras")))
    llvm::sys::fs::create_directory(llvm::Twine(ToPath + "/dpl_extras"));

#define GENERATE_ALL_FILE_CONTENT(VAR_NAME, FILE_NAME)                         \
  {                                                                            \
    std::ofstream VAR_NAME##File(ToPath + "/" + #FILE_NAME, std::ios::binary); \
    std::string Code = VAR_NAME##AllContentStr;                                \
    replaceEndOfLine(Code);                                                    \
    VAR_NAME##File << Code;                                                    \
    VAR_NAME##File.flush();                                                    \
  }
#define GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(VAR_NAME, FILE_NAME)              \
  {                                                                            \
    std::ofstream VAR_NAME##File(ToPath + "/dpl_extras/" + #FILE_NAME,         \
                                 std::ios::binary);                            \
    std::string Code = VAR_NAME##AllContentStr;                                \
    replaceEndOfLine(Code);                                                    \
    VAR_NAME##File << Code;                                                    \
    VAR_NAME##File.flush();                                                    \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic, "atomic.hpp")
  GENERATE_ALL_FILE_CONTENT(BlasUtils, "blas_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(Device, "device.hpp")
  GENERATE_ALL_FILE_CONTENT(Dpct, "dpct.hpp")
  GENERATE_ALL_FILE_CONTENT(DplUtils, "dpl_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(DnnlUtils, "dnnl_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(Image, "image.hpp")
  GENERATE_ALL_FILE_CONTENT(Kernel, "kernel.hpp")
  GENERATE_ALL_FILE_CONTENT(Math, "math.hpp")
  GENERATE_ALL_FILE_CONTENT(Memory, "memory.hpp")
  GENERATE_ALL_FILE_CONTENT(Util, "util.hpp")
  GENERATE_ALL_FILE_CONTENT(RngUtils, "rng_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(LibCommonUtils, "lib_common_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(CclUtils, "ccl_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(SparseUtils, "sparse_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(FftUtils, "fft_utils.hpp")
  GENERATE_ALL_FILE_CONTENT(LapackUtils, "lapack_utils.hpp")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasAlgorithm, "algorithm.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasFunctional, "functional.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasIterators, "iterators.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasMemory, "memory.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasNumeric, "numeric.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasVector, "vector.h")
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasDpcppExtensions, "dpcpp_extensions.h")
#undef GENERATE_ALL_FILE_CONTENT
#undef GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT
}

} // namespace dpct
} // namespace clang
