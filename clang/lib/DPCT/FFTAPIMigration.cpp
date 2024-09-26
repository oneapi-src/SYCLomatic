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
  std::string ParameterTypes = MapNames::getLibraryHelperNamespace() +
                               "fft::fft_engine_ptr engine";
  requestFeature(HelperFeatureEnum::device_ext);
  std::string Dir;
  std::string NewFuncName;
  if (FuncNameRef == "cufftExecC2C") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 *in, " + MapNames::getClNamespace() +
                     "float2 *out, " + MapNames::getLibraryHelperNamespace() +
                     "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "float2, " +
                  MapNames::getClNamespace() + "float2>";
  } else if (FuncNameRef == "cufftExecZ2Z") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 *in, " + MapNames::getClNamespace() +
                     "double2 *out, " + MapNames::getLibraryHelperNamespace() +
                     "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "double2, " +
                  MapNames::getClNamespace() + "double2>";
  } else if (FuncNameRef == "cufftExecR2C") {
    ParameterTypes = ParameterTypes + ", float *in, " +
                     MapNames::getClNamespace() + "float2 *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::forward";
    NewFuncName = "compute<float, " + MapNames::getClNamespace() + "float2>";
  } else if (FuncNameRef == "cufftExecC2R") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 *in, float *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::backward";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "float2, float>";
  } else if (FuncNameRef == "cufftExecD2Z") {
    ParameterTypes = ParameterTypes + ", double *in, " +
                     MapNames::getClNamespace() + "double2 *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::forward";
    NewFuncName = "compute<double, " + MapNames::getClNamespace() + "double2>";
  } else if (FuncNameRef == "cufftExecZ2D") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 *in, double *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::backward";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "double2, double>";
  } else {
    return nullptr;
  }
  std::string ReplStr = "[](" + ParameterTypes + "){" + getNL() + "  engine->" +
                        NewFuncName + "(in, out, " + Dir + ");" + getNL() +
                        "  return 0;" + getNL() + "}";
  ReplaceStmt *TM = new ReplaceStmt(UO, ReplStr);
  TM->setBlockLevelFormatFlag(true);
  return TM;
}


} // namespace dpct
} // namespace clang
