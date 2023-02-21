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

TextModification *processFunctionPointer(const UnaryOperator *UO) {
  if (!UO)
    return nullptr;
  const DeclRefExpr *DRE = dyn_cast_or_null<DeclRefExpr>(UO->getSubExpr());
  if (!DRE)
    return nullptr;
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return nullptr;
  StringRef FuncNameRef = FD->getName();
  std::string ParameterTypes =
      MapNames::getDpctNamespace() + "fft::fft_engine_ptr engine";
  requestFeature(HelperFeatureEnum::FftUtils_fft_engine, UO->getBeginLoc());
  std::string Dir;
  std::string NewFuncName;
  if (FuncNameRef == "cufftExecC2C") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 *in, " + MapNames::getClNamespace() +
                     "float2 *out, " + MapNames::getDpctNamespace() +
                     "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "mfloat2, " +
                  MapNames::getClNamespace() + "mfloat2>";
  } else if (FuncNameRef == "cufftExecZ2Z") {
    ParameterTypes = ParameterTypes + ", " +
                     MapNames::TypeNamesMap["double2"]->NewName + " *in, " +
                     MapNames::TypeNamesMap["double2"]->NewName + " *out, " +
                     MapNames::getDpctNamespace() + "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::TypeNamesMap["double2"]->NewName +
                  ", " + MapNames::TypeNamesMap["double2"]->NewName + ">";
  } else if (FuncNameRef == "cufftExecR2C") {
    ParameterTypes = ParameterTypes + ", float *in, " +
                     MapNames::TypeNamesMap["float2"]->NewName + " *out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_direction::forward";
    NewFuncName =
        "compute<float, " + MapNames::TypeNamesMap["float2"]->NewName + ">";
  } else if (FuncNameRef == "cufftExecC2R") {
    ParameterTypes = ParameterTypes + ", " +
                     MapNames::TypeNamesMap["float2"]->NewName +
                     " *in, float *out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_direction::backward";
    NewFuncName =
        "compute<" + MapNames::TypeNamesMap["float2"]->NewName + ", float>";
  } else if (FuncNameRef == "cufftExecD2Z") {
    ParameterTypes = ParameterTypes + ", double *in, " +
                     MapNames::TypeNamesMap["double2"]->NewName + " *out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_direction::forward";
    NewFuncName =
        "compute<double, " + MapNames::TypeNamesMap["double2"]->NewName + ">";
  } else if (FuncNameRef == "cufftExecZ2D") {
    ParameterTypes = ParameterTypes + ", " +
                     MapNames::TypeNamesMap["double2"]->NewName +
                     " *in, double *out";
    Dir = MapNames::getDpctNamespace() + "fft::fft_direction::backward";
    NewFuncName =
        "compute<" + MapNames::TypeNamesMap["double2"]->NewName + ", double>";
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
