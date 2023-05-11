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
#include "clang/DPCT/dpct.all.inc"
    ;
const std::string AtomicAllContentStr =
#include "clang/DPCT/atomic.all.inc"
    ;
const std::string BlasUtilsAllContentStr =
#include "clang/DPCT/blas_utils.all.inc"
    ;
const std::string DnnlUtilsAllContentStr =
#include "clang/DPCT/dnnl_utils.all.inc"
    ;
const std::string DeviceAllContentStr =
#include "clang/DPCT/device.all.inc"
    ;
const std::string DplUtilsAllContentStr =
#include "clang/DPCT/dpl_utils.all.inc"
    ;
const std::string ImageAllContentStr =
#include "clang/DPCT/image.all.inc"
    ;
const std::string KernelAllContentStr =
#include "clang/DPCT/kernel.all.inc"
    ;
const std::string MathAllContentStr =
#include "clang/DPCT/math.all.inc"
    ;
const std::string MemoryAllContentStr =
#include "clang/DPCT/memory.all.inc"
    ;
const std::string UtilAllContentStr =
#include "clang/DPCT/util.all.inc"
    ;
const std::string RngUtilsAllContentStr =
#include "clang/DPCT/rng_utils.all.inc"
    ;
const std::string LibCommonUtilsAllContentStr =
#include "clang/DPCT/lib_common_utils.all.inc"
    ;
const std::string CclUtilsAllContentStr =
#include "clang/DPCT/ccl_utils.all.inc"
    ;
const std::string SparseUtilsAllContentStr =
#include "clang/DPCT/sparse_utils.all.inc"
    ;
const std::string FftUtilsAllContentStr =
#include "clang/DPCT/fft_utils.all.inc"
    ;
const std::string LapackUtilsAllContentStr =
#include "clang/DPCT/lapack_utils.all.inc"
    ;
const std::string DplExtrasAlgorithmAllContentStr =
#include "clang/DPCT/dpl_extras/algorithm.all.inc"
    ;
const std::string DplExtrasFunctionalAllContentStr =
#include "clang/DPCT/dpl_extras/functional.all.inc"
    ;
const std::string DplExtrasIteratorsAllContentStr =
#include "clang/DPCT/dpl_extras/iterators.all.inc"
    ;
const std::string DplExtrasMemoryAllContentStr =
#include "clang/DPCT/dpl_extras/memory.all.inc"
    ;
const std::string DplExtrasNumericAllContentStr =
#include "clang/DPCT/dpl_extras/numeric.all.inc"
    ;
const std::string DplExtrasVectorAllContentStr =
#include "clang/DPCT/dpl_extras/vector.all.inc"
    ;
const std::string DplExtrasDpcppExtensionsAllContentStr =
#include "clang/DPCT/dpl_extras/dpcpp_extensions.all.inc"
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

namespace {
std::unordered_map<std::string, std::string> HelperFileNameMap{
#define HELPERFILE(PATH, FILENAME, UNIQUE_ENUM) {#UNIQUE_ENUM, #FILENAME},
#include "../../runtime/dpct-rt/include/HelperFileNames.inc"
#undef HELPERFILE
};
} // namespace

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

#define GENERATE_ALL_FILE_CONTENT(FILE_NAME)                                   \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/" + HelperFileNameMap.at(#FILE_NAME), std::ios::binary);    \
    std::string Code = FILE_NAME##AllContentStr;                               \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(FILE_NAME)                        \
  {                                                                            \
    std::ofstream FILE_NAME##File(ToPath + "/dpl_extras/" +                    \
                                      HelperFileNameMap.at(#FILE_NAME),        \
                                  std::ios::binary);                           \
    std::string Code = FILE_NAME##AllContentStr;                               \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic)
  GENERATE_ALL_FILE_CONTENT(BlasUtils)
  GENERATE_ALL_FILE_CONTENT(Device)
  GENERATE_ALL_FILE_CONTENT(Dpct)
  GENERATE_ALL_FILE_CONTENT(DplUtils)
  GENERATE_ALL_FILE_CONTENT(DnnlUtils)
  GENERATE_ALL_FILE_CONTENT(Image)
  GENERATE_ALL_FILE_CONTENT(Kernel)
  GENERATE_ALL_FILE_CONTENT(Math)
  GENERATE_ALL_FILE_CONTENT(Memory)
  GENERATE_ALL_FILE_CONTENT(Util)
  GENERATE_ALL_FILE_CONTENT(RngUtils)
  GENERATE_ALL_FILE_CONTENT(LibCommonUtils)
  GENERATE_ALL_FILE_CONTENT(CclUtils)
  GENERATE_ALL_FILE_CONTENT(SparseUtils)
  GENERATE_ALL_FILE_CONTENT(FftUtils)
  GENERATE_ALL_FILE_CONTENT(LapackUtils)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasVector)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasDpcppExtensions)
#undef GENERATE_ALL_FILE_CONTENT
#undef GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT
}

} // namespace dpct
} // namespace clang
