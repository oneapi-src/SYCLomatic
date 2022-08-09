//===--------------- FFTAPIMigration.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FFTAPIMigration.h"

namespace clang {
namespace dpct {

TextModification* processFunctionPointer(const UnaryOperator *UO) {
  if (!UO)
    return nullptr;
  const DeclRefExpr *DRE = dyn_cast_or_null<DeclRefExpr>(UO->getSubExpr());
  if (!DRE)
    return nullptr;
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return nullptr;
  StringRef FuncNameRef = FD->getName();
  std::string ParameterTypes = "std::shared_ptr<" +
                               MapNames::getDpctNamespace() +
                               "fft::fft_solver> solver";
  std::string Dir;
  if (FuncNameRef == "cufftExecC2C") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 in, " + MapNames::getClNamespace() +
                     "float2 out, " + MapNames::getDpctNamespace() +
                     "fft::fft_dir dir";
    Dir = "dir";
  } else if (FuncNameRef == "cufftExecZ2Z") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 in, " + MapNames::getClNamespace() +
                     "double2 out, " + MapNames::getDpctNamespace() +
                     "fft::fft_dir dir";
    Dir = "dir";
  } else if (FuncNameRef == "cufftExecR2C") {
    ParameterTypes = ParameterTypes + ", float in, " +
                     MapNames::getClNamespace() + "float2 out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_dir::forward";
  } else if (FuncNameRef == "cufftExecC2R") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 in, float out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_dir::backward";
  } else if (FuncNameRef == "cufftExecD2Z") {
    ParameterTypes = ParameterTypes + ", double in, " +
                     MapNames::getClNamespace() + "double2 out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_dir::forward";
  } else if (FuncNameRef == "cufftExecZ2D") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 in, double out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_dir::backward";
  } else {
    return nullptr;
  }
  std::string ReplStr = "[](" + ParameterTypes + "){" + getNL() +
                        "  desc->compute(in, out, " + Dir + ");" + getNL() +
                        "  return 0;" + getNL() + "}";
  ReplaceStmt *TM = new ReplaceStmt(UO, ReplStr);
  TM->setBlockLevelFormatFlag(true);
  return TM;
}


} // namespace dpct
} // namespace clang
