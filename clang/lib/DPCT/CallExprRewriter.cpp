//===--------------- CallExprRewriter.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "AnalysisInfo.h"
#include "BLASAPIMigration.h"
#include "ExprAnalysis.h"
#include "MapNames.h"
#include "Utility.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LangOptions.h"
#include <cstdarg>

namespace clang {
namespace dpct {

std::string CallExprRewriter::getMigratedArg(unsigned Idx) {
  Analyzer.setCallSpelling(Call);
  Analyzer.analyze(Call->getArg(Idx));
  return Analyzer.getRewritePrefix() + Analyzer.getRewriteString() +
         Analyzer.getRewritePostfix();
}

std::vector<std::string> CallExprRewriter::getMigratedArgs() {
  std::vector<std::string> ArgList;
  Analyzer.setCallSpelling(Call);
  for (unsigned i = 0; i < Call->getNumArgs(); ++i)
    ArgList.emplace_back(getMigratedArg(i));
  return ArgList;
}

Optional<std::string> FuncCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  return buildRewriteString();
}

Optional<std::string> FuncCallExprRewriter::buildRewriteString() {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << TargetCalleeName << "(";
  for (auto &Arg : RewriteArgList)
    OS << Arg << ", ";
  OS.flush();
  return RewriteArgList.empty() ? Result.append(")")
                                : Result.replace(Result.length() - 2, 2, ")");
}

Optional<std::string> MathCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  setTargetCalleeName(SourceCalleeName.str());
  return buildRewriteString();
}

void MathCallExprRewriter::reportUnsupportedRoundingMode() {
  if (SourceCalleeName.endswith("_rd") || SourceCalleeName.endswith("_rn") ||
      SourceCalleeName.endswith("_ru") || SourceCalleeName.endswith("_rz")) {
    report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, false);
  }
}

// Judge if a function is declared in user code or not
bool isTargetMathFunction(const FunctionDecl *FD) {
  if (!FD)
    return false;
  auto FilePath = DpctGlobalInfo::getLocInfo(FD).first;
  if (isChildOrSamePath(DpctGlobalInfo::getInRoot(), FilePath))
    return false;
  return true;
}

Optional<std::string> MathFuncNameRewriter::rewrite() {
  // If the function is not a target math function, do not migrate it
  if (!isTargetMathFunction(Call->getDirectCallee())) {
    // No actions needed here, just return an empty string
    return {};
  }

  reportUnsupportedRoundingMode();
  RewriteArgList = getMigratedArgs();
  auto NewFuncName = getNewFuncName();

  if (NewFuncName.empty() || NewFuncName == SourceCalleeName)
    return {};

  setTargetCalleeName(NewFuncName);
  return buildRewriteString();
}

Optional<std::string> NoRewriteFuncNameRewriter::rewrite() {
  // If the function is not a target math function, do not migrate it
  if (!isTargetMathFunction(Call->getDirectCallee())) {
    // No actions needed here, just return an empty string
    return {};
  }

  reportUnsupportedRoundingMode();
  RewriteArgList = getMigratedArgs();
  auto NewFuncName = getNewFuncName();

  if (NewFuncName == SourceCalleeName)
    return {};

  return NewFuncName;
}

/// Returns true if E is one of the forms:
/// (blockDim/blockIdx/threadIdx/gridDim).(x/y/z)
bool isTargetPseudoObjectExpr(const Expr *E) {
  if (auto POE = dyn_cast<PseudoObjectExpr>(E->IgnoreImpCasts())) {
    auto RE = POE->getResultExpr();
    if (auto CE = dyn_cast<CallExpr>(RE)) {
      auto FD = CE->getDirectCallee();
      auto Name = FD->getNameAsString();
      if (Name == "__fetch_builtin_x" || Name == "__fetch_builtin_y" ||
          Name == "__fetch_builtin_z")
        return true;
    }
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts())) {
    auto VarDecl = DRE->getDecl();
    if (VarDecl && (VarDecl->getNameAsString() == "warpSize")) {
      return !DpctGlobalInfo::isInRoot(VarDecl->getLocation());
    }
  }
  return false;
}

/// Policies to migrate math functions:
/// 1) Functions with the "std" namespace are treated as host functions;
/// 2) Functions with __device__ attribute but without __host__
///    attribute are treated as device functions;
/// 3) Functions whose calling functions are augmented with __device__
///    or __global__ attributes are treated as device functions;
/// 4) Other functions are treated as host functions.
///    eg. "__host__ __device__ fabs()" falls in 4) if fabs is not called in
///    device or kernel
std::string MathFuncNameRewriter::getNewFuncName() {
  auto FD = Call->getDirectCallee();
  std::string NewFuncName;
  if (!FD) {
    NewFuncName = SourceCalleeName.str();
  } else {
    NewFuncName = TargetCalleeName;
    std::string NamespaceStr;
    auto DRE = dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImpCasts());
    if (DRE) {
      auto Qualifier = DRE->getQualifier();
      if (Qualifier) {
        auto Namespace = Qualifier->getAsNamespace();
        if (Namespace)
          NamespaceStr = Namespace->getName().str();
      }
    }

    if (dpct::DpctGlobalInfo::isInRoot(FD->getBeginLoc())) {
      return "";
    }

    auto ContextFD = getImmediateOuterFuncDecl(Call);
    if (NamespaceStr == "std" && ContextFD &&
        !ContextFD->hasAttr<CUDADeviceAttr>() &&
        !ContextFD->hasAttr<CUDAGlobalAttr>()) {
      return "";
    }
    // For device functions
    else if ((FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAHostAttr>()) ||
             (ContextFD && (ContextFD->hasAttr<CUDADeviceAttr>() ||
                            ContextFD->hasAttr<CUDAGlobalAttr>()))) {
      if (SourceCalleeName == "abs") {
        // further check the type of the args.
        if (!Call->getArg(0)->getType()->isIntegerType()) {
          NewFuncName = MapNames::getClNamespace(false, true) + "fabs";
        }
      }

      if (SourceCalleeName == "min" || SourceCalleeName == "max") {
        LangOptions LO;
        std::string FT =
            Call->getType().getCanonicalType().getAsString(PrintingPolicy(LO));
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i);
          auto ArgExprClass = Arg->getStmtClass();
          if (isTargetPseudoObjectExpr(Arg)) {
            RewriteArgList[i] = "(" + FT + ")" + RewriteArgList[i];
          } else {
            std::string ArgT = Arg->getType().getCanonicalType().getAsString(
                PrintingPolicy(LO));
            auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
            if (ArgT != FT || ArgExprClass == Stmt::BinaryOperatorClass) {
              if (DRE)
                RewriteArgList[i] = "(" + FT + ")" + RewriteArgList[i];
              else
                RewriteArgList[i] = "(" + FT + ")(" + RewriteArgList[i] + ")";
            }
          }
        }
      }

      if (SourceCalleeName == "__clz" || SourceCalleeName == "__clzll") {
        LangOptions LO;
        auto Arg = Call->getArg(0);
        std::string ArgT =
            Arg->IgnoreImplicit()->getType().getAsString(PrintingPolicy(LO));
        auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
        auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
        if (SourceCalleeName == "__clz") {
          if (ArgT != "int") {
            if (DRE || IL)
              RewriteArgList[0] = "(int)" + RewriteArgList[0];
            else
              RewriteArgList[0] = "(int)(" + RewriteArgList[0] + ")";
          }
        } else {
          if (ArgT != "long long") {
            if (DRE || IL)
              RewriteArgList[0] = "(long long)" + RewriteArgList[0];
            else
              RewriteArgList[0] = "(long long)(" + RewriteArgList[0] + ")";
          }
        }
      } else if (SourceCalleeName == "__mul24" || SourceCalleeName == "mul24" ||
                 SourceCalleeName == "__umul24" ||
                 SourceCalleeName == "umul24" ||
                 SourceCalleeName == "__mulhi" ||
                 SourceCalleeName == "__hadd") {
        std::string ParamType = "int";
        if (SourceCalleeName == "__umul24" || SourceCalleeName == "umul24")
          ParamType = "unsigned int";
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i)->IgnoreImpCasts();
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getAsString(PrintingPolicy(LO));
          auto ArgExpr = Arg->getStmtClass();
          if (ArgExpr == Stmt::PseudoObjectExprClass) {
            // The type of (blockDim/blockIdx/threadIdx/gridDim).(x/y/z) is
            // unsigned int but it is migrated to size_t (unsigned long in
            // typical 64-bit systems). However, sycl::mul24 only takes 32-bit
            // integers, so it is necessary to convert the migrated type to
            // int or unsigned int.
            if (isTargetPseudoObjectExpr(Arg))
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
          } else {
            auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
            auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
            auto FL = dyn_cast<FloatingLiteral>(Arg->IgnoreCasts());
            if (ArgT != ParamType) {
              if (DRE || IL || FL)
                RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
              else
                RewriteArgList[i] =
                    "(" + ParamType + ")(" + RewriteArgList[i] + ")";
            }
          }
        }
      } else if (MapNames::MathTypeCastingMap.count(SourceCalleeName.str())) {
        auto TypePair = MapNames::MathTypeCastingMap[SourceCalleeName.str()];
        bool NeedFromType = false;
        if (auto ICE = dyn_cast_or_null<ImplicitCastExpr>(Call->getArg(0))) {
          if (ICE->getCastKind() != CastKind::CK_LValueToRValue)
            NeedFromType = true;
        }
        if (NeedFromType)
          NewFuncName =
              NewFuncName + "<" + TypePair.first + ", " + TypePair.second + ">";
        else
          NewFuncName = NewFuncName + "<" + TypePair.first + ">";
      }

      if (std::find(SingleFuctions.begin(), SingleFuctions.end(),
                    SourceCalleeName) != SingleFuctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          if (SourceCalleeName == "ldexpf" && i == 1)
            continue;
          auto Arg = Call->getArg(i);
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getCanonicalType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "float";
          auto PVD = FD->getParamDecl(i);
          if (PVD) {
            ParamType = PVD->getType()
                            .getCanonicalType()
                            .getUnqualifiedType()
                            .getAsString();
          }
          // Since isnan is overloaded for both float and double, so there is no
          // need to add type conversions for isnan.
          if (ArgT != ParamType && SourceCalleeName != "isnan") {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] =
                  "(" + ParamType + ")(" + RewriteArgList[i] + ")";
          } else if (ParamType == "int" || ParamType == "unsigned int") {
            if (DRE || IL)
              RewriteArgList[i] = "(float)" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(float)(" + RewriteArgList[i] + ")";
          }
        }
      } else if (std::find(DoubleFuctions.begin(), DoubleFuctions.end(),
                           SourceCalleeName) != DoubleFuctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          if (SourceCalleeName == "ldexp" && i == 1)
            continue;
          auto Arg = Call->getArg(i);
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getCanonicalType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "double";
          auto PVD = FD->getParamDecl(i);
          if (PVD) {
            ParamType = PVD->getType()
                            .getCanonicalType()
                            .getUnqualifiedType()
                            .getAsString();
          }
          if (ArgT != ParamType) {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] =
                  "(" + ParamType + ")(" + RewriteArgList[i] + ")";
          } else if (ParamType == "int" || ParamType == "unsigned int") {
            if (DRE || IL)
              RewriteArgList[i] = "(double)" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(double)(" + RewriteArgList[i] + ")";
          }
        }
      }
    }
    // For host functions
    else {
      // Insert "#include <cmath>" to migrated code
      DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(), HT_Math);
      NewFuncName = SourceCalleeName.str();
      if (SourceCalleeName == "abs") {
        auto *BT =
            dyn_cast<BuiltinType>(Call->getArg(0)->IgnoreImpCasts()->getType());
        if (BT) {
          auto K = BT->getKind();
          if (K == BuiltinType::Float) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "f";
          } else if (K == BuiltinType::Double) {
            NewFuncName = "f" + SourceCalleeName.str();
          } else if (K == BuiltinType::LongDouble) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "l";
          }
        }
      } else if (SourceCalleeName == "max" || SourceCalleeName == "min") {
        if (Call->getNumArgs() != 2) {
          return SourceCalleeName.str();
        }
        auto Arg0 = Call->getArg(0)->IgnoreImpCasts();
        auto Arg1 = Call->getArg(1)->IgnoreImpCasts();
        auto *BT0 = dyn_cast<BuiltinType>(Arg0->getType());
        auto *BT1 = dyn_cast<BuiltinType>(Arg1->getType());
        // Deal with cases where types of arguments are typedefs, e.g.,
        // 1) typdef int INT;
        // 2) using int_t = int;
        const TypedefType *TT0 = nullptr, *TT1 = nullptr;
        if (!BT0) {
          TT0 = dyn_cast<TypedefType>(Arg0->getType());
          if (TT0)
            BT0 = dyn_cast<BuiltinType>(TT0->desugar().getTypePtr());
        }
        if (!BT1) {
          TT1 = dyn_cast<TypedefType>(Arg1->getType());
          if (TT1)
            BT1 = dyn_cast<BuiltinType>(TT1->desugar().getTypePtr());
        }
        if (BT0 && BT1) {
          auto K0 = BT0->getKind();
          auto K1 = BT1->getKind();
          if (K0 == BuiltinType::LongDouble || K1 == BuiltinType::LongDouble) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "l";
          } else if (K0 == BuiltinType::Double || K1 == BuiltinType::Double) {
            NewFuncName = "f" + SourceCalleeName.str();
          } else if (K0 == BuiltinType::Float || K1 == BuiltinType::Float) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "f";
          } else if (BT0->isInteger() && BT0->isInteger()) {
            // Host max/min functions with integer parameters are in <algorithm>
            // instead of <cmath>, so we need to migrate them to std versions
            // and do necessary type conversions.
            bool MigrateToStd = true;
            std::string TypeName;
            // Deal with integer types in this branch
            PrintingPolicy PP{LangOptions()};
            if (K0 == K1) {
              // Nothing to do: no type conversion needed
            } else if (BT0->isSignedInteger() && BT1->isSignedInteger()) {
              // Only deal with short, int, long, and long long
              if (K0 < BuiltinType::Short || K0 > BuiltinType::LongLong ||
                  K1 < BuiltinType::Short || K1 > BuiltinType::LongLong)
                return NewFuncName;
              // Convert shorter types to longer types
              if (K0 < K1) {
                TypeName = BT1->getNameAsCString(PP);
              } else {
                TypeName = BT0->getNameAsCString(PP);
              }
            } else if (BT0->isUnsignedInteger() && BT1->isUnsignedInteger()) {
              // Only deal with unsigned short, unsigned int, unsigned long,
              // and unsigned long long
              if (K0 < BuiltinType::UShort || K0 > BuiltinType::ULongLong ||
                  K1 < BuiltinType::UShort || K1 > BuiltinType::ULongLong)
                return NewFuncName;
              // Convert shorter types to longer types
              if (K0 < K1) {
                TypeName = BT1->getNameAsCString(PP);
              } else {
                TypeName = BT0->getNameAsCString(PP);
              }
            } else {
              // Convert signed types to unsigned types if the bit width of
              // the signed is equal or smaller than that of the unsigned;
              // otherwise, do not migrate them. Overflow is not considered.
              const BuiltinType *UnsignedType;
              const TypedefType *UnsignedTypedefType;
              BuiltinType::Kind UnsignedKind, SignedKind;
              if (BT0->isSignedInteger() && BT1->isUnsignedInteger()) {
                UnsignedType = BT1;
                UnsignedTypedefType = TT1;
                UnsignedKind = K1;
                SignedKind = K0;
              } else if (BT0->isUnsignedInteger() && BT1->isSignedInteger()) {
                UnsignedType = BT0;
                UnsignedTypedefType = TT0;
                UnsignedKind = K0;
                SignedKind = K1;
              }
              auto GetType = [=]() -> std::string {
                std::string TypeName;
                if (UnsignedTypedefType) {
                  if (auto TND = UnsignedTypedefType->getDecl())
                    TypeName = TND->getNameAsString();
                  else
                    TypeName = UnsignedType->getNameAsCString(PP);
                } else {
                  TypeName = UnsignedType->getNameAsCString(PP);
                }
                return TypeName;
              };
              switch (UnsignedKind) {
              case BuiltinType::ULongLong:
                switch (SignedKind) {
                case BuiltinType::LongLong:
                case BuiltinType::Long:
                case BuiltinType::Int:
                case BuiltinType::Short:
                  TypeName = GetType();
                  break;
                default:
                  MigrateToStd = false;
                }
                break;
              case BuiltinType::ULong:
                switch (SignedKind) {
                case BuiltinType::Long:
                case BuiltinType::Int:
                case BuiltinType::Short:
                  TypeName = GetType();
                  break;
                default:
                  MigrateToStd = false;
                }
                break;
              case BuiltinType::UInt:
                switch (SignedKind) {
                case BuiltinType::Int:
                case BuiltinType::Short:
                  TypeName = GetType();
                  break;
                default:
                  MigrateToStd = false;
                }
                break;
              case BuiltinType::UShort:
                switch (SignedKind) {
                case BuiltinType::Short:
                  TypeName = GetType();
                  break;
                default:
                  MigrateToStd = false;
                }
                break;
              default:
                MigrateToStd = false;
              }
            }

            if (NamespaceStr.empty() && MigrateToStd) {
              DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(),
                                                         HT_Algorithm);
              NewFuncName = "std::" + SourceCalleeName.str();
              if (!TypeName.empty())
                NewFuncName += "<" + TypeName + ">";
            }
          }
        }
      }

      if (!NamespaceStr.empty())
        NewFuncName = NamespaceStr + "::" + NewFuncName;
    }
  }
  return NewFuncName;
}

Optional<std::string> MathUnsupportedRewriter::rewrite() {
  report(Diagnostics::API_NOT_MIGRATED, false,
         MapNames::ITFName.at(SourceCalleeName.str()));
  return Base::rewrite();
}

Optional<std::string> MathTypeCastRewriter::rewrite() {
  auto FD = Call->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return Base::rewrite();

  using SSMap = std::map<std::string, std::string>;
  static SSMap RoundingModeMap{{"", "automatic"},
                               {"rd", "rtn"},
                               {"rn", "rte"},
                               {"ru", "rtp"},
                               {"rz", "rtz"}};
  const StringRef &FuncName = SourceCalleeName;
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);

  auto MigratedArg0 = getMigratedArg(0);
  if (FuncName == "__float22half2_rn") {
    OS << MigratedArg0
       << ".convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__float2half2_rn") {
    OS << MapNames::getClNamespace() + "float2{" << MigratedArg0 << ","
       << MigratedArg0
       << "}.convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__floats2half2_rn") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "float2{" << MigratedArg0 << ","
       << MigratedArg1
       << "}.convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__half22float2") {
    OS << MigratedArg0
       << ".convert<float, " + MapNames::getClNamespace() +
              "rounding_mode::automatic>()";
  } else if (FuncName == "__half2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << ","
       << MigratedArg0 << "}";
  } else if (FuncName == "__halves2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << ","
       << MigratedArg1 << "}";
  } else if (FuncName == "__high2float") {
    OS << MigratedArg0 << "[0]";
  } else if (FuncName == "__high2half") {
    OS << MigratedArg0 << "[0]";
  } else if (FuncName == "__high2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[0], "
       << MigratedArg0 << "[0]}";
  } else if (FuncName == "__highs2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[0], "
       << MigratedArg1 << "[0]}";
  } else if (FuncName == "__low2float") {
    OS << MigratedArg0 << "[1]";
  } else if (FuncName == "__low2half") {
    OS << MigratedArg0 << "[1]";
  } else if (FuncName == "__low2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg0 << "[1]}";
  } else if (FuncName == "__lowhigh2highlow") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg0 << "[0]}";
  } else if (FuncName == "__lows2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg1 << "[1]}";
  } else {
    //__half2short_rd and __half2float
    static SSMap TypeMap{{"ll", "long long"},
                         {"ull", "unsigned long long"},
                         {"ushort", "unsigned short"},
                         {"uint", "unsigned int"},
                         {"half", MapNames::getClNamespace() + "half"}};
    std::string RoundingMode;
    if (FuncName[FuncName.size() - 3] == '_')
      RoundingMode = FuncName.substr(FuncName.size() - 2).str();
    auto FN = FuncName.substr(2, FuncName.find('_', 2) - 2).str();
    auto Types = split(FN, '2');
    assert(Types.size() == 2);
    MapNames::replaceName(TypeMap, Types[0]);
    MapNames::replaceName(TypeMap, Types[1]);
    OS << MapNames::getClNamespace() + "vec<" << Types[0] << ", 1>{"
       << MigratedArg0 << "}.convert<" << Types[1]
       << ", " + MapNames::getClNamespace() + "rounding_mode::"
       << RoundingModeMap[RoundingMode] << ">()[0]";
  }
  OS.flush();
  return ReplStr;
}

bool isArgMigratedToAccessor(const CallExpr *Call, unsigned Index) {
  if (auto DRE = dyn_cast<DeclRefExpr>(Call->getArg(Index)->IgnoreImpCasts())) {
    if (!DRE->getDecl()->hasAttrs())
      return false;
    for (auto A : DRE->getDecl()->getAttrs()) {
      auto K = A->getKind();
      if (K == attr::CUDAConstant || K == attr::CUDADevice ||
          K == attr::CUDAShared)
        return true;
    }
  }
  return false;
}

std::string getTypecastName(const CallExpr *Call) {
  auto Arg0TypeName = Call->getArg(0)->getType().getAsString();
  auto Arg1TypeName = Call->getArg(1)->getType().getAsString();
  auto RetTypeName = Call->getType().getAsString();
  bool B0 = isArgMigratedToAccessor(Call, 0);
  bool B1 = isArgMigratedToAccessor(Call, 1);
  if (B0 && !B1)
    return Arg1TypeName;
  if (!B0 && B1)
    return Arg0TypeName;
  if (B0 && B1)
    return RetTypeName;
  return {};
}

Optional<std::string> MathSimulatedRewriter::rewrite() {
  std::string NamespaceStr;
  auto DRE = dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImpCasts());
  if (DRE) {
    auto Qualifier = DRE->getQualifier();
    if (Qualifier) {
      auto Namespace = Qualifier->getAsNamespace();
      if (Namespace)
        NamespaceStr = Namespace->getName().str();
    }
  }

  Analyzer.setCallSpelling(Call);

  auto FD = Call->getDirectCallee();
  if (!FD)
    return Base::rewrite();

  if (dpct::DpctGlobalInfo::isInRoot(FD->getBeginLoc())) {
    return {};
  }

  auto ContextFD = getImmediateOuterFuncDecl(Call);
  if (NamespaceStr == "std" && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>()) {
    return {};
  }

  if (!FD->hasAttr<CUDADeviceAttr>() && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>())
    return Base::rewrite();

  // Do need to report warnings for pow or funnelshift migrations
  if (SourceCalleeName != "pow" && SourceCalleeName != "powf" &&
      SourceCalleeName != "__powf" && SourceCalleeName != "__funnelshift_l" &&
      SourceCalleeName != "__funnelshift_lc" &&
      SourceCalleeName != "__funnelshift_r" &&
      SourceCalleeName != "__funnelshift_rc")
    report(Diagnostics::MATH_EMULATION, false,
           MapNames::ITFName.at(SourceCalleeName.str()), TargetCalleeName);

  const std::string FuncName = SourceCalleeName.str();
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  auto MigratedArg0 = getMigratedArg(0);

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*Call);
  bool IsInReturnStmt = false;
  if (Parents.size())
    if (auto ParentStmt = getParentStmt(Call))
      if (ParentStmt->getStmtClass() == Stmt::StmtClass::ReturnStmtClass)
          IsInReturnStmt = true;

  if (FuncName == "frexp" || FuncName == "frexpf") {
    auto Arg = Call->getArg(0);
    std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    std::string ArgExpr = Arg->getStmtClassName();
    auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
    if (ArgT == "int") {
      if (FuncName == "frexpf") {
        if (DRE)
          MigratedArg0 = "(float)" + MigratedArg0;
        else
          MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      } else {
        if (DRE)
          MigratedArg0 = "(double)" + MigratedArg0;
        else
          MigratedArg0 = "(double)(" + MigratedArg0 + ")";
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace(false, true) + "frexp(" << MigratedArg0
       << ", " + MapNames::getClNamespace() + "make_ptr<int, "
       << MapNames::getClNamespace() + "access::address_space::" +
              getAddressSpace(Call->getArg(1), MigratedArg1) + ">("
       << MigratedArg1 << "))";
  } else if (FuncName == "modf" || FuncName == "modff") {
    auto Arg = Call->getArg(0);
    std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    std::string ArgExpr = Arg->getStmtClassName();
    auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
    if (ArgT == "int") {
      if (FuncName == "modff") {
        if (DRE)
          MigratedArg0 = "(float)" + MigratedArg0;
        else
          MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      } else {
        if (DRE)
          MigratedArg0 = "(double)" + MigratedArg0;
        else
          MigratedArg0 = "(double)(" + MigratedArg0 + ")";
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace(false, true) + "modf(" << MigratedArg0;
    if (FuncName == "modf")
      OS << ", " + MapNames::getClNamespace() + "make_ptr<double, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(1), MigratedArg1) + ">(";
    else
      OS << ", " + MapNames::getClNamespace() + "make_ptr<float, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(1), MigratedArg1) + ">(";
    OS << MigratedArg1 << "))";
  } else if (FuncName == "nan" || FuncName == "nanf") {
    OS << MapNames::getClNamespace(false, true) + "nan(0u)";
  } else if (FuncName == "sincos" || FuncName == "sincosf" ||
             FuncName == "__sincosf") {
    std::string Buf;
    llvm::raw_string_ostream RSO(Buf);

    auto Arg = Call->getArg(0);
    std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    std::string ArgExpr = Arg->getStmtClassName();
    auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
    if (ArgT == "int") {
      if (FuncName == "sincosf" || FuncName == "__sincosf") {
        if (DRE)
          MigratedArg0 = "(float)" + MigratedArg0;
        else
          MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      } else {
        if (DRE)
          MigratedArg0 = "(double)" + MigratedArg0;
        else
          MigratedArg0 = "(double)(" + MigratedArg0 + ")";
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      RSO << MigratedArg1.substr(1);
    else
      RSO << "*(" + MigratedArg1 + ")";
    RSO << " = " + MapNames::getClNamespace(false, true) + "sincos("
       << MigratedArg0;

    if (FuncName == "sincos")
      RSO << ", " + MapNames::getClNamespace() + "make_ptr<double, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(2), MigratedArg2) + ">(";
    else
      RSO << ", " + MapNames::getClNamespace() + "make_ptr<float, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(2), MigratedArg2) + ">(";
    RSO << MigratedArg2 << "))";

    if(IsInReturnStmt) {
      OS << "[&](){ " << Buf << ";"<< " }()";
      BlockLevelFormatFlag = true;
    } else {
      OS << Buf;
    }

  } else if (FuncName == "sincospi" || FuncName == "sincospif") {
    std::string Buf;
    llvm::raw_string_ostream RSO(Buf);

    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      RSO << MigratedArg1.substr(1);
    else
      RSO << "*(" + MigratedArg1 + ")";
    RSO << " = " + MapNames::getClNamespace(false, true) + "sincos("
       << MigratedArg0;
    if (FuncName == "sincospi") {
      RSO << " * DPCT_PI";
      requestFeature(HelperFeatureEnum::Dpct_dpct_pi, Call);
    } else {
      RSO << " * DPCT_PI_F";
      requestFeature(HelperFeatureEnum::Dpct_dpct_pi_f, Call);
    }

    if (FuncName == "sincospi")
      RSO << ", " + MapNames::getClNamespace() + "make_ptr<double, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(2), MigratedArg2) + ">(";
    else
      RSO << ", " + MapNames::getClNamespace() + "make_ptr<float, " +
                MapNames::getClNamespace() + "access::address_space::" +
                getAddressSpace(Call->getArg(2), MigratedArg2) + ">(";
    RSO << MigratedArg2 << "))";
    if(IsInReturnStmt) {
      OS << "[&](){ " << Buf << ";"<< " }()";
      BlockLevelFormatFlag = true;
    } else {
      OS << Buf;
    }
  } else if (FuncName == "remquo" || FuncName == "remquof") {
    {
      auto Arg = Call->getArg(0);
      std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
          PrintingPolicy(LangOptions()));
      std::string ArgExpr = Arg->getStmtClassName();
      auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
      if (ArgT == "int") {
        if (FuncName == "remquof") {
          if (DRE)
            MigratedArg0 = "(float)" + MigratedArg0;
          else
            MigratedArg0 = "(float)(" + MigratedArg0 + ")";
        } else {
          if (DRE)
            MigratedArg0 = "(double)" + MigratedArg0;
          else
            MigratedArg0 = "(double)(" + MigratedArg0 + ")";
        }
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    {
      auto Arg = Call->getArg(1);
      std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
          PrintingPolicy(LangOptions()));
      std::string ArgExpr = Arg->getStmtClassName();
      auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
      if (ArgT == "int") {
        if (FuncName == "remquof") {
          if (DRE)
            MigratedArg1 = "(float)" + MigratedArg1;
          else
            MigratedArg1 = "(float)(" + MigratedArg1 + ")";
        } else {
          if (DRE)
            MigratedArg1 = "(double)" + MigratedArg1;
          else
            MigratedArg1 = "(double)(" + MigratedArg1 + ")";
        }
      }
    }
    auto MigratedArg2 = getMigratedArg(2);
    OS << MapNames::getClNamespace(false, true) + "remquo(" << MigratedArg0
       << ", " << MigratedArg1
       << ", " + MapNames::getClNamespace() + "make_ptr<int, " +
              MapNames::getClNamespace() + "access::address_space::" +
              getAddressSpace(Call->getArg(2), MigratedArg2) + ">("
       << MigratedArg2 << "))";
  } else if (FuncName == "nearbyint" || FuncName == "nearbyintf") {
    OS << MapNames::getClNamespace(false, true) + "floor(" << MigratedArg0
       << " + 0.5)";
  } else if (FuncName == "rhypot" || FuncName == "rhypotf") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "1 / " + MapNames::getClNamespace(false, true) + "hypot("
       << MigratedArg0 << ", " << MigratedArg1 << ")";
  } else if (SourceCalleeName == "pow" || SourceCalleeName == "powf" ||
             SourceCalleeName == "__powf") {
    RewriteArgList = getMigratedArgs();
    if (Call->getNumArgs() != 2) {
      TargetCalleeName = SourceCalleeName.str();
      return buildRewriteString();
    }
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto PP = Context.getPrintingPolicy();
    PP.PrintCanonicalTypes = 1;
    auto Arg0 = Call->getArg(0);
    auto Arg1 = Call->getArg(1);

    std::string T0, T1;
    if (auto CXXFCE = dyn_cast<CXXFunctionalCastExpr>(Call->getArg(0))) {
      T0 = CXXFCE->getType().getAsString(PP);
    } else {
      T0 = Arg0->IgnoreCasts()->getType().getAsString(PP);
    }

    if (auto CXXFCE = dyn_cast<CXXFunctionalCastExpr>(Call->getArg(1))) {
      T1 = CXXFCE->getType().getAsString(PP);
    } else {
      T1 = Arg1->IgnoreCasts()->getType().getAsString(PP);
    }

    auto IL1 = dyn_cast<IntegerLiteral>(Arg1->IgnoreCasts());
    auto FL1 = dyn_cast<FloatingLiteral>(Arg1->IgnoreCasts());
    auto DRE0 = dyn_cast<DeclRefExpr>(Arg0->IgnoreCasts());
    // For integer literal 2 or floating literal 2.0/2.0f, expand pow to
    // multiply expression:
    // pow(x, 2) ==> x * x, if x is an expression that has no side effects.
    // pow(x, 2.0) ==> x * x, if x is an expression that has no side effects.
    // pow(x, 2.0f) ==> x * x, if x is an expression that has no side effects.
    bool IsExponentTwo = false;
    if (IL1) {
      if (IL1->getValue().getZExtValue() == 2)
        IsExponentTwo = true;
    } else if (FL1) {
      auto &SM = DpctGlobalInfo::getSourceManager();
      if (!FL1->getBeginLoc().isMacroID() && !FL1->getEndLoc().isMacroID()) {
        auto B = SM.getCharacterData(FL1->getBeginLoc());
        auto E = SM.getCharacterData(
            Lexer::getLocForEndOfToken(FL1->getEndLoc(), 0, SM, LangOptions()));
        std::string Exponent(B, E);
        if (Exponent == "2.0" || Exponent == "2.0f")
          IsExponentTwo = true;
      }
    }
    SideEffectsAnalysis SEA(Arg0);
    SEA.analyze();
    bool Arg0HasSideEffects = SEA.hasSideEffects();
    if (!Arg0HasSideEffects && IsExponentTwo) {
      auto Arg0Str = SEA.getReplacedString();
      if (!needExtraParens(Arg0))
        return Arg0Str + " * " + Arg0Str;
      else
        return "(" + Arg0Str + ") * (" + Arg0Str + ")";
    }
    if (IL1 || T1 == "int" || T1 == "unsigned int" || T1 == "char" ||
        T1 == "unsigned char" || T1 == "short" || T1 == "unsigned short") {
      if (T0 == "int") {
        if (DRE0) {
          RewriteArgList[0] = "(float)" + RewriteArgList[0];
        } else {
          RewriteArgList[0] = "(float)(" + RewriteArgList[0] + ")";
        }
      }
      if (T1 != "int") {
        RewriteArgList[1] = "(int)" + RewriteArgList[1];
      }
      TargetCalleeName = MapNames::getClNamespace(false, true) + "pown";
    }
    return buildRewriteString();
  } else if (FuncName == "erfcx" || FuncName == "erfcxf") {
    OS << MapNames::getClNamespace(false, true) << "exp(" << MigratedArg0 << "*"
       << MigratedArg0 << ")*" << TargetCalleeName << "(" << MigratedArg0
       << ")";
  } else if (FuncName == "norm3d" || FuncName == "norm3df") {
    OS << TargetCalleeName << "(" << MapNames::getClNamespace() << "float3("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << "))";
  } else if (FuncName == "norm4d" || FuncName == "norm4df") {
    OS << TargetCalleeName << "(" << MapNames::getClNamespace() << "float4("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << ", " << getMigratedArg(3) << "))";
  } else if (FuncName == "rcbrt" || FuncName == "rcbrtf") {
    OS << MapNames::getClNamespace(false, true) << "native::recip((float)"
       << TargetCalleeName << "(" << getMigratedArg(0) << "))";
  } else if (FuncName == "rnorm3d" || FuncName == "rnorm3df") {
    OS << MapNames::getClNamespace(false, true) << "native::recip("
       << TargetCalleeName << "(" << MapNames::getClNamespace() << "float3("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << ")))";
  } else if (FuncName == "rnorm4d" || FuncName == "rnorm4df") {
    OS << MapNames::getClNamespace(false, true) << "native::recip("
       << TargetCalleeName << "(" << MapNames::getClNamespace() << "float4("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << ", " << getMigratedArg(3) << ")))";
  } else if (FuncName == "scalbln" || FuncName == "scalblnf" ||
             FuncName == "scalbn" || FuncName == "scalbnf") {
    OS << MigratedArg0 << "*(2<<" << getMigratedArg(1) << ")";
  } else if (FuncName == "__double2hiint") {
    requestFeature(HelperFeatureEnum::Util_cast_double_to_int, Call);
    OS << MapNames::getDpctNamespace() << "cast_double_to_int(" << MigratedArg0
       << ")";
  } else if (FuncName == "__double2loint") {
    requestFeature(HelperFeatureEnum::Util_cast_double_to_int, Call);
    OS << MapNames::getDpctNamespace() << "cast_double_to_int(" << MigratedArg0
       << ", false)";
  } else if (FuncName == "__hiloint2double") {
    requestFeature(HelperFeatureEnum::Util_cast_ints_to_double, Call);
    OS << MapNames::getDpctNamespace() << "cast_ints_to_double(" << MigratedArg0
       << ", " << getMigratedArg(1) << ")";
  } else if (FuncName == "__sad" || FuncName == "__usad") {
    OS << TargetCalleeName << "(" << MigratedArg0 << ", " << getMigratedArg(1)
       << ")"
       << "+" << getMigratedArg(2);
  } else if (FuncName == "__drcp_rd" || FuncName == "__drcp_rn" ||
             FuncName == "__drcp_ru" || FuncName == "__drcp_rz") {
    auto Arg0 = Call->getArg(0);
    auto T0 = Arg0->IgnoreCasts()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    auto DRE0 = dyn_cast<DeclRefExpr>(Arg0->IgnoreCasts());
    report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, false);
    OS << TargetCalleeName;
    if (T0 != "float") {
      if (DRE0)
        OS << "((float)" << MigratedArg0 << ")";
      else
        OS << "((float)(" << MigratedArg0 << "))";
    } else {
      OS << "(" << MigratedArg0 << ")";
    }
  } else if (FuncName == "norm") {
    Expr::EvalResult ER;
    if (Call->getArg(0)->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      std::string MigratedArg1 = getMigratedArg(1);
      int64_t Value = ER.Val.getInt().getExtValue();
      switch (Value) {
      case 0:
        return std::string("0");
      case 1:
        OS << TargetCalleeName << "((float)" << MigratedArg1 << "[0])";
        break;
      case 2:
        OS << TargetCalleeName << "(" << MapNames::getClNamespace() << "float2("
           << MigratedArg1 << "[0], " << MigratedArg1 << "[1]))";
        break;
      case 3:
        OS << TargetCalleeName << "(" << MapNames::getClNamespace() << "float3("
           << MigratedArg1 << "[0], " << MigratedArg1 << "[1], " << MigratedArg1
           << "[2]))";
        break;
      case 4:
        OS << TargetCalleeName << "(" << MapNames::getClNamespace() << "float4("
           << MigratedArg1 << "[0], " << MigratedArg1 << "[1], " << MigratedArg1
           << "[2], " << MigratedArg1 << "[3]))";
        break;
      default:
        requestFeature(HelperFeatureEnum::Util_fast_length, Call);
        OS << MapNames::getDpctNamespace() << "fast_length("
           << "(float *)" << getMigratedArg(1) << ", " << MigratedArg0 << ")";
      }
    } else {
      requestFeature(HelperFeatureEnum::Util_fast_length, Call);
      OS << MapNames::getDpctNamespace() << "fast_length("
         << "(float *)" << getMigratedArg(1) << ", " << MigratedArg0 << ")";
    }
  } else if (FuncName == "__funnelshift_l" || FuncName == "__funnelshift_lc" ||
             FuncName == "__funnelshift_r" || FuncName == "__funnelshift_rc") {
    report(Diagnostics::MATH_EMULATION_EXPRESSION, false,
           MapNames::ITFName.at(SourceCalleeName.str()), TargetCalleeName);
    auto Namespace = MapNames::getClNamespace();
    auto Low = getMigratedArg(0);
    auto High = getMigratedArg(1);
    auto Shift = getMigratedArg(2);
    if (FuncName == "__funnelshift_l") {
      OS << "((" << Namespace << "upsample<unsigned>(" << High
         << ", " << Low << ") << (" << Shift << " & 31)) >> 32)";
    } else if (FuncName == "__funnelshift_lc") {
      OS << "((" << Namespace << "upsample<unsigned>(" << High
         << ", " << Low << ") << " << Namespace << "min(" << Shift
         << ", 32)) >> 32)";
    } else if (FuncName == "__funnelshift_r") {
      OS << "((" << Namespace << "upsample<unsigned>(" << High
         << ", " << Low << ") >> (" << Shift << " & 31)) & 0xFFFFFFFF)";
    } else if (FuncName == "__funnelshift_rc") {
      OS << "((" << Namespace << "upsample<unsigned>(" << High
         << ", " << Low << ") >> " << Namespace << "min(" << Shift
         << ", 32)) & 0xFFFFFFFF)";
    }
  }

  OS.flush();
  return ReplStr;
}

Optional<std::string> MathBinaryOperatorRewriter::rewrite() {
  reportUnsupportedRoundingMode();
  if (SourceCalleeName == "__hneg" || SourceCalleeName == "__hneg2") {
    setLHS("");
    if (needExtraParens(Call->getArg(0)))
      setRHS("(" + getMigratedArg(0) + ")");
    else
      setRHS(getMigratedArg(0));
  } else {
    if (needExtraParens(Call->getArg(0)))
      setLHS("(" + getMigratedArg(0) + ")");
    else
      setLHS(getMigratedArg(0));
    if (needExtraParens(Call->getArg(1)))
      setRHS("(" + getMigratedArg(1) + ")");
    else
      setRHS(getMigratedArg(1));
  }
  return buildRewriteString();
}

// In AST, &SubExpr could be recognized as UnaryOperator or CXXOperatorCallExpr.
// To get the SubExpr from the original Expr, both cases need to be handled.
const Expr *getDereferencedExpr(const Expr *E) {
  E = E->IgnoreImplicitAsWritten()->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      return UO->getSubExpr()->IgnoreImplicitAsWritten();
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      return COCE->getArg(0)->IgnoreImplicitAsWritten();
    }
  }
  return nullptr;
}

DerefExpr DerefExpr::create(const Expr *E) {
  DerefExpr D;
  // If E is UnaryOperator or CXXOperatorCallExpr D.E will has value
  D.E = getDereferencedExpr(E);
  if (D.E) {
    D.E = D.E->IgnoreParens();
    D.AddrOfRemoved = true;
  } else {
    D.E = E;
  }

  D.NeedParens = needExtraParens(E);
  return D;
}

class DerefStreamExpr {
  const Expr *E;

  bool isDefaultStream() const {
    auto Expression = E->IgnoreImpCasts();
    if (auto Paren = dyn_cast<ParenExpr>(Expression)) {
      Expression = Paren->getSubExpr()->IgnoreImpCasts();
    }
    if (auto TypeCast = dyn_cast<ExplicitCastExpr>(Expression)) {
      Expression = TypeCast->getSubExpr()->IgnoreImpCasts();
    }
    Expr::EvalResult Result;
    if (!Expression->isValueDependent() &&
        Expression->EvaluateAsInt(Result, DpctGlobalInfo::getContext())) {
      auto Val = Result.Val.getInt().getZExtValue();
      return Val < 3; // 0 or 1 (cudaStreamLegacy) or 2 (cudaStreamPerThread)
                      // all migrated to default queue;
    }
    Expression->dumpPretty(DpctGlobalInfo::getContext());
    return false;
  }

  template <class StreamT> void printDefaultQueue(StreamT &Stream) const {
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, E, HelperFuncType::HFT_DefaultQueue);
    Stream << "{{NEEDREPLACEQ" << Index << "}}";
  }

public:
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    if (isDefaultStream())
      printDefaultQueue(Stream);
    else
      DerefExpr::create(E).printArg(Stream, A);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    if (isDefaultStream()) {
      printDefaultQueue(Stream);
      Stream << ".";
    } else {
      DerefExpr::create(E).printMemberBase(Stream);
    }
  }

  template <class StreamT> void print(StreamT &Stream) const {
    if (isDefaultStream())
      printDefaultQueue(Stream);
    else
      DerefExpr::create(E).print(Stream);
  }

  DerefStreamExpr(const Expr *Expression) : E(Expression) {}
};

std::function<DerefStreamExpr(const CallExpr *)>
makeDerefStreamExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> DerefStreamExpr {
    return DerefStreamExpr(C->getArg(Idx));
  };
}

std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> DerefExpr {
    return DerefExpr::create(C->getArg(Idx));
  };
}

std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(
    std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
        F) {
  return [=](const CallExpr *C) -> DerefExpr {
    return DerefExpr::create(F(C).second);
  };
}

std::function<BLASEnumExpr(const CallExpr *)>
makeBLASEnumCallArgCreator(unsigned Idx, BLASEnumExpr::BLASEnumType BET) {
  return [=](const CallExpr *C) -> BLASEnumExpr {
    return BLASEnumExpr::create(C->getArg(Idx), BET);
  };
}

std::function<const Expr *(const CallExpr *)> makeCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> const Expr * { return C->getArg(Idx); };
}

std::function<const StringRef(const CallExpr *)>
makeCallArgCreator(std::string Str) {
  return [=](const CallExpr *C) -> const StringRef { return StringRef(Str); };
}

std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
makeCallArgCreatorWithCall(unsigned Idx) {
  return [=](const CallExpr *C) -> std::pair<const CallExpr *, const Expr *> {
    return std::pair<const CallExpr *, const Expr *>(C, C->getArg(Idx));
  };
}

const Expr *removeCStyleCast(const Expr *E) {
  if (auto CSCE = dyn_cast<ExplicitCastExpr>(E)) {
    return CSCE->getSubExpr()->IgnoreImplicitAsWritten();
  } else {
    return E;
  }
}

// Prepare the arg for deref by removing the CStyleCast
// Should be used when the cast information is not relevant.
// e.g. migrating cudaMallocHost((void**)ptr, size) to
// *ptr = sycl::malloc_host<float>(size, q_ct1);
std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
makeDerefArgCreatorWithCall(unsigned Idx) {
  return [=](const CallExpr *C) -> std::pair<const CallExpr *, const Expr *> {
    return std::pair<const CallExpr *, const Expr *>(
        C, removeCStyleCast(C->getArg(Idx)));
  };
}

template <class T1, class T2>
std::function<std::pair<T1, T2>(const CallExpr *)>
makeCombinedArg(std::function<T1(const CallExpr *)> Part1,
                std::function<T2(const CallExpr *)> Part2) {
  return [=](const CallExpr *C) -> std::pair<T1, T2> {
    return std::make_pair(Part1(C), Part2(C));
  };
}

std::function<std::vector<RenameWithSuffix>(const CallExpr *)>
makeStructDismantler(unsigned Idx, const std::vector<std::string> &Suffixes) {
  return [=](const CallExpr *C) -> std::vector<RenameWithSuffix> {
    std::vector<RenameWithSuffix> Ret;
    if (auto DRE = dyn_cast_or_null<DeclRefExpr>(
            getDereferencedExpr(C->getArg(Idx)))) {
      Ret.reserve(Suffixes.size());
      auto Origin = DRE->getDecl()->getName();
      std::transform(Suffixes.begin(), Suffixes.end(), std::back_inserter(Ret),
                     [&](StringRef Suffix) -> RenameWithSuffix {
                       return RenameWithSuffix(Origin, Suffix);
                     });
    }
    return Ret;
  };
}

std::function<std::string(const CallExpr *)>
makeExtendStr(unsigned Idx, const std::string Suffix) {
  return [=](const CallExpr *C) -> std::string {
    ArgumentAnalysis AA;
    AA.setCallSpelling(C);
    AA.analyze(C->getArg(Idx));
    std::string S = "(std::string(" + AA.getRewriteString() + ") + \"" +
                    Suffix + "\").c_str()";
    return S;
  };
}

std::function<std::string(const CallExpr *)> makeQueueStr() {
  return [=](const CallExpr *C) -> std::string {
    int Index = getPlaceholderIdx(C);
    if (Index == 0) {
      Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    }

    buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
    std::string S = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    return S;
  };
}

std::function<std::string(const CallExpr *)> makeDeviceStr() {
  return [=](const CallExpr *C) -> std::string {
    int Index = getPlaceholderIdx(C);
    if (Index == 0) {
      Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    }

    buildTempVariableMap(Index, C, HelperFuncType::HFT_CurrentDevice);
    std::string S = "{{NEEDREPLACED" + std::to_string(Index) + "}}";
    return S;
  };
}

std::function<std::string(const CallExpr *)>
makeMappedThrustPolicyEnum(unsigned Idx) {
  return [=](const CallExpr *C) -> std::string {
    auto E = C->getArg(Idx);
    if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
      if (auto DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
        std::string EnumName = DRE->getNameInfo().getName().getAsString();
        if (EnumName == "device") {
          return "oneapi::dpl::execution::make_device_policy(" +
                 makeQueueStr()(C) + ")";
        } else if (EnumName == "seq") {
          return "oneapi::dpl::execution::seq";
        }
      }
    }
    return "oneapi::dpl::execution::make_device_policy(" + makeQueueStr()(C) +
           ")";
  };
}

template <class BaseT, class... CallArgsT>
using MemberCallPrinterCreator =
    PrinterCreator<MemberCallPrinter<BaseT, StringRef, CallArgsT...>,
                   std::function<BaseT(const CallExpr *)>, bool, std::string,
                   std::function<CallArgsT(const CallExpr *)>...>;

template <class BaseT, class... CallArgsT>
std::function<
    MemberCallPrinter<BaseT, StringRef, CallArgsT...>(const CallExpr *)>
makeMemberCallCreator(std::function<BaseT(const CallExpr *)> BaseFunc,
                      bool IsArrow, std::string Member,
                      std::function<CallArgsT(const CallExpr *)>... Args) {
  return MemberCallPrinterCreator<BaseT, CallArgsT...>(BaseFunc, IsArrow,
                                                       Member, Args...);
}

template <class... StmtT>
std::function<
    LambdaPrinter<StmtT...>(const CallExpr *)>
makeLambdaCreator(bool IsCaptureRef,
                      std::function<StmtT(const CallExpr *)>... Stmts) {
  return PrinterCreator<LambdaPrinter<StmtT...>, bool,
                        std::function<StmtT(const CallExpr *)>...>(
                        IsCaptureRef, Stmts...);
}

auto getTemplateArgsList =
    [](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
  ArrayRef<TemplateArgumentLoc> TemplateArgsList;
  std::vector<TemplateArgumentInfo> Ret;
  auto Callee = C->getCallee()->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
    TemplateArgsList = DRE->template_arguments();
  } else if (auto ULE = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    TemplateArgsList = ULE->template_arguments();
  }
  for (auto Arg : TemplateArgsList) {
    Ret.emplace_back(Arg);
  }
  return Ret;
};

std::function<TemplatedCallee(const CallExpr *)>
makeTemplatedCalleeCreator(std::string CalleeName,
                           std::vector<size_t> Indexes) {
  return PrinterCreator<
      TemplatedCallee, std::string,
      std::function<std::vector<TemplateArgumentInfo>(const CallExpr *)>>(
      CalleeName, [=](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
        std::vector<TemplateArgumentInfo> Ret;
        auto List = getTemplateArgsList(C);
        for (auto Idx : Indexes) {
          if (Idx < List.size()) {
            Ret.emplace_back(List[Idx]);
          }
        }
        return Ret;
      });
}

std::function<TemplateArgumentInfo(const CallExpr *)>
makeCallArgCreatorFromTemplateArg(unsigned Idx) {
  return [=](const CallExpr *CE) -> TemplateArgumentInfo {
    return getTemplateArgsList(CE)[Idx];
  };
}

template <class First>
void setTemplateArgumentInfo(const CallExpr *C,
                             std::vector<TemplateArgumentInfo> &Vec,
                             std::function<First(const CallExpr *)> F) {
  TemplateArgumentInfo TAI;
  TAI.setAsType(F(C));
  Vec.emplace_back(TAI);
}

template <class First, class... ArgsT>
void setTemplateArgumentInfo(const CallExpr *C,
                             std::vector<TemplateArgumentInfo> &Vec,
                             std::function<First(const CallExpr *)> F,
                             ArgsT... Args) {
  TemplateArgumentInfo TAI;
  TAI.setAsType(F(C));
  Vec.emplace_back(TAI);
  setTemplateArgumentInfo(C, Vec, Args...);
}

template <class... CallArgsT>
std::function<TemplatedCallee(const CallExpr *)>
makeTemplatedCalleeWithArgsCreator(
    std::string Callee, std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<
      TemplatedCallee, std::string,
      std::function<std::vector<TemplateArgumentInfo>(const CallExpr *)>>(
      Callee, [=](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
        std::vector<TemplateArgumentInfo> Ret;
        setTemplateArgumentInfo(C, Ret, Args...);
        return Ret;
      });
}

template <BinaryOperatorKind Op, class LValueT, class RValueT>
std::function<BinaryOperatorPrinter<Op, LValueT, RValueT>(const CallExpr *)>
makeBinaryOperatorCreator(std::function<LValueT(const CallExpr *)> L,
                          std::function<RValueT(const CallExpr *)> R) {
  return PrinterCreator<BinaryOperatorPrinter<Op, LValueT, RValueT>,
                        std::function<LValueT(const CallExpr *)>,
                        std::function<RValueT(const CallExpr *)>>(std::move(L),
                                                                  std::move(R));
}

template <class CalleeT, class... CallArgsT>
std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>
makeCallExprCreator(std::function<CalleeT(const CallExpr *)> Callee,
                    std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<CallExprPrinter<CalleeT, CallArgsT...>,
                        std::function<CalleeT(const CallExpr *)>,
                        std::function<CallArgsT(const CallExpr *)>...>(Callee,
                                                                       Args...);
}

template <class... CallArgsT>
std::function<CallExprPrinter<StringRef, CallArgsT...>(const CallExpr *)>
makeCallExprCreator(std::string Callee,
                    std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<CallExprPrinter<StringRef, CallArgsT...>, std::string,
                        std::function<CallArgsT(const CallExpr *)>...>(Callee,
                                                                       Args...);
}

std::function<std::string(const CallExpr *)> makeLiteral(std::string Str) {
  return [=](const CallExpr *) { return Str; };
}

template <class BaseT, class MemberT>
std::function<MemberExprPrinter<BaseT, MemberT>(const CallExpr *)>
makeMemberExprCreator(std::function<BaseT(const CallExpr *)> Base, bool IsArrow,
                      std::function<MemberT(const CallExpr *)> Member) {
  return PrinterCreator<MemberExprPrinter<BaseT, MemberT>,
                        std::function<BaseT(const CallExpr *)>, bool,
                        std::function<MemberT(const CallExpr *)>>(Base, IsArrow,
                                                                  Member);
}

template <class TypeInfoT, class SubExprT>
std::function<CastExprPrinter<TypeInfoT, SubExprT>(const CallExpr *)>
makeCastExprCreator(std::function<TypeInfoT(const CallExpr *)> TypeInfo,
                    std::function<SubExprT(const CallExpr *)> Sub) {
  return PrinterCreator<CastExprPrinter<TypeInfoT, SubExprT>,
                        std::function<TypeInfoT(const CallExpr *)>,
                        std::function<SubExprT(const CallExpr *)>>(TypeInfo,
                                                                   Sub);
}

template <class... ArgsT>
std::function<NewExprPrinter<ArgsT...>(const CallExpr *)>
makeNewExprCreator(std::string TypeName,
                   std::function<ArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<NewExprPrinter<ArgsT...>, std::string,
                        std::function<ArgsT(const CallExpr *)>...>(TypeName,
                                                                   Args...);
}

bool isCallAssigned(const CallExpr *C) { return isAssigned(C); }

template <unsigned int Idx>
unsigned int getSizeFromCallArg(const CallExpr *C, std::string &Var) {
  // Default sub group size align with cuda warp size
  if (Idx == UINT_MAX) {
    return 32;
  }
  auto SizeExpr = C->getArg(Idx);
  Expr::EvalResult Result;
  if (!SizeExpr->isValueDependent() &&
      SizeExpr->EvaluateAsInt(Result, DpctGlobalInfo::getContext())) {
    return Result.Val.getInt().getZExtValue();
  } else {
    ExprAnalysis EA(SizeExpr);
    Var = EA.getReplacedString();
    return UINT_MAX;
  }
}

/// If the input \p QT is a pointer type or an array type, this function will
/// return the deref-ed type. Otherwise an empty QualType object will be
/// returned. The caller needs to check if the return value is null using
/// isNull().
QualType DerefQualType(QualType QT) {
  QualType DerefQT;
  if (QT->isPointerType()) {
    DerefQT = QT->getPointeeType();
  } else if (QT->isArrayType()) {
    DerefQT = dyn_cast<ArrayType>(QT.getTypePtr())->getElementType();
  }
  return DerefQT;
}

// Get the replaced type of a function call argument
// For example, foo(x) where x is an int2, this function will return sycl::int2
std::function<std::string(const CallExpr *C)> getReplacedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";
    return DpctGlobalInfo::getReplacedTypeName(C->getArg(Idx)->getType());
  };
}

// Get the derefed type name of an arg while getDereferencedExpr is get the
// derefed expr.
std::function<std::string(const CallExpr *C)> getDerefedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";
    auto TE = removeCStyleCast(C->getArg(Idx));
    // Deref by removing the "&" of &SubExpr
    auto DE = getDereferencedExpr(TE);
    bool NeedDeref = true;
    // If getDereferencedExpr returns value, DE is the derefed TE.
    if (DE) {
      NeedDeref = false;
      TE = DE;
    }
    TE = TE->IgnoreParens();

    QualType DerefQT;
    if (auto ArraySub = dyn_cast<ArraySubscriptExpr>(TE)) {
      // Handle cases like A[3] where A is an array or pointer
      QualType BaseType = ArraySub->getBase()->getType();
      if (BaseType->isArrayType()) {
        if (auto Array = BaseType->getAsArrayTypeUnsafe()) {
          DerefQT = Array->getElementType();
        }
      } else if (BaseType->isPointerType()) {
        DerefQT = BaseType->getPointeeType();
      }
    }

    // All other cases
    if (DerefQT.isNull()) {
      DerefQT = TE->getType();
    }

    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
    if (TypeStr == "<dependent type>") {
      if (NeedDeref) {
        return "typename std::remove_pointer<decltype(" +
               ExprAnalysis::ref(TE) + ")>::type";
      } else {
        return "typename std::remove_reference<decltype(" +
               ExprAnalysis::ref(TE) + ")>::type";
      }
    }

    if (NeedDeref) {
      DerefQT = DerefQualType(DerefQT);
      if (DerefQT.isNull())
        return "";
    }

    return DpctGlobalInfo::getReplacedTypeName(DerefQT);
  };
}

// Can only be used if CheckCanUseTemplateMalloc is true.
std::function<std::string(const CallExpr *C)> getDoubleDerefedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";

    // Remove CStyleCast if any
    auto TE = removeCStyleCast(C->getArg(Idx));

    // Deref twice
    QualType DerefQT = TE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return "";
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return "";
    std::string ReplType = DpctGlobalInfo::getReplacedTypeName(DerefQT);

    return ReplType;
  };
}

// Remove sizeof(T) if using template version.
// Can only be used if CheckCanUseTemplateMalloc is true.
std::function<std::string(const CallExpr *C)> getSizeForMalloc(size_t PtrIdx,
                                                               size_t SizeIdx) {
  return [=](const CallExpr *C) -> std::string {
    auto AllocatedExpr = C->getArg(PtrIdx);
    auto SizeExpr = C->getArg(SizeIdx);
    const Expr *AE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
      AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    } else {
      AE = AllocatedExpr;
    }

    ArgumentAnalysis AA;
    AA.setCallSpelling(C);
    AA.analyze(SizeExpr);
    std::string OrginalStr =
        AA.getRewritePrefix() + AA.getRewriteString() + AA.getRewritePostfix();

    // Deref twice
    QualType DerefQT = AE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return OrginalStr;
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return OrginalStr;

    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);

    auto BO = dyn_cast<BinaryOperator>(SizeExpr);
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
      std::string Repl;
      if (!isContainMacro(BO->getLHS()) &&
          isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
        // case 1: sizeof(b) * a
        ArgumentAnalysis AASize;
        AASize.setCallSpelling(C);
        AASize.analyze(BO->getRHS());
        Repl = AASize.getRewritePrefix() + AASize.getRewriteString() +
               AASize.getRewritePostfix();
        return Repl;
      } else if (!isContainMacro(BO->getRHS()) &&
                 isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
        // case 2: a * sizeof(b)
        ArgumentAnalysis AASize;
        AASize.setCallSpelling(C);
        AASize.analyze(BO->getLHS());
        Repl = AASize.getRewritePrefix() + AASize.getRewriteString() +
               AASize.getRewritePostfix();
        return Repl;
      } else {
        return OrginalStr;
      }
    } else {
      // case 3: sizeof(b)
      if (!isContainMacro(SizeExpr) &&
          isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
        return "1";
      }
    }

    return OrginalStr;
  };
}

std::function<bool(const CallExpr *C)> checkIsUSM() {
  return [](const CallExpr *C) -> bool {
    return DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted;
  };
}

std::function<bool(const CallExpr *C)> checkIsArgStream(size_t index) {
  return [=](const CallExpr *C) -> bool {
    return !(isPredefinedStreamHandle(C->getArg(index)));
  };
}

std::function<bool(const CallExpr *C)> checkArgSpelling(size_t index,
                                                        std::string str) {
  return [=](const CallExpr *C) -> bool {
    return getStmtSpelling(C->getArg(index)) == str;
  };
}

std::function<bool(const CallExpr *C)> checkIsCallExprOnly() {
  return [=](const CallExpr *C) -> bool {
    auto parentStmt = getParentStmt(C);
    if (parentStmt != nullptr && (dyn_cast<CompoundStmt>(parentStmt) ||
                          dyn_cast<ExprWithCleanups>(parentStmt)))
      return true;
    return false;
    };
}

std::function<bool(const CallExpr *C)> checkIsArgIntegerLiteral(size_t index) {
  return [=](const CallExpr *C) -> bool {
    auto Arg2Expr = C->getArg(index);
    if (auto NamedCaster = dyn_cast<ExplicitCastExpr>(Arg2Expr)) {
      if (NamedCaster->getTypeAsWritten()->isIntegerType()) {
        Arg2Expr = NamedCaster->getSubExpr();
      }
    }
    return Arg2Expr->getStmtClass() == Stmt::IntegerLiteralClass;
  };
}

template <size_t Idx>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFactoryWithSubGroupSizeRequest(
    std::string NewFuncName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Inner) {
  return std::make_pair(
      std::move(Inner.first),
      std::make_shared<RewriterFactoryWithSubGroupSize>(
          getSizeFromCallArg<Idx>, std::move(NewFuncName), Inner.second));
}

template <size_t Idx, class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFactoryWithSubGroupSizeRequest(
    std::string NewFuncName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Inner,
    T) {
  return createFactoryWithSubGroupSizeRequest<Idx>(std::move(NewFuncName),
                                                   std::move(Inner));
}

template <class... StmtPrinters>
std::shared_ptr<CallExprRewriterFactoryBase> createMultiStmtsRewriterFactory(
    const std::string &SourceName,
    std::function<StmtPrinters(const CallExpr *)> &&...Creators) {
  return std::make_shared<ConditionalRewriterFactory>(
      isCallAssigned,
      std::make_shared<AssignableRewriterFactory>(
          std::make_shared<CallExprRewriterFactory<
              PrinterRewriter<CommaExprPrinter<StmtPrinters...>>,
              std::function<StmtPrinters(const CallExpr *)>...>>(SourceName,
                                                                 Creators...)),
      std::make_shared<CallExprRewriterFactory<
          PrinterRewriter<MultiStmtsPrinter<StmtPrinters...>>,
          std::function<StmtPrinters(const CallExpr *)>...>>(SourceName,
                                                             Creators...));
}

/// Create AssignExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p LValueCreator use to get lhs from original call expr.
/// \p RValueCreator use to get rhs from original call expr.
template <BinaryOperatorKind BO, class LValue, class RValue>
std::shared_ptr<CallExprRewriterFactoryBase> creatBinaryOpRewriterFactory(
    const std::string &SourceName,
    std::function<LValue(const CallExpr *)> &&LValueCreator,
    std::function<RValue(const CallExpr *)> &&RValueCreator) {
  return std::make_shared<
      CallExprRewriterFactory<BinaryOpRewriter<BO, LValue, RValue>,
                              std::function<LValue(const CallExpr *)>,
                              std::function<RValue(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<LValue(const CallExpr *)>>(LValueCreator),
      std::forward<std::function<RValue(const CallExpr *)>>(RValueCreator));
}

template <class BaseT, class MemberT>
std::shared_ptr<CallExprRewriterFactoryBase> creatMemberExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const CallExpr *)> &&BaseCreator, bool IsArrow,
    std::function<MemberT(const CallExpr *)> &&MemberCreator) {
  return std::make_shared<
      CallExprRewriterFactory<MemberExprRewriter<BaseT, MemberT>,
                              std::function<BaseT(const CallExpr *)>, bool,
                              std::function<MemberT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<BaseT(const CallExpr *)>>(BaseCreator),
      IsArrow,
      std::forward<std::function<MemberT(const CallExpr *)>>(MemberCreator));
}

std::shared_ptr<CallExprRewriterFactoryBase> creatIfElseRewriterFactory(
    const std::string &SourceName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        PredCreator,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        IfCreator,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        ElseCreator,
    int i) {
  return std::make_shared<CallExprRewriterFactory<
      IfElseRewriter, std::shared_ptr<CallExprRewriterFactoryBase>,
      std::shared_ptr<CallExprRewriterFactoryBase>,
      std::shared_ptr<CallExprRewriterFactoryBase>>>(
      SourceName, PredCreator.second, IfCreator.second, ElseCreator.second);
}

/// Create CallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p ArgsCreator use to get call args from original call expr.
template <class CalleeT, class... CallArgsT>
std::shared_ptr<CallExprRewriterFactoryBase> creatCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>
        Args) {
  return std::make_shared<CallExprRewriterFactory<
      SimpleCallExprRewriter<CalleeT, CallArgsT...>,
      std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>>>(
      SourceName, Args);
}

/// Create TemplatedCallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p CalleeCreator use to get templated callee from original call expr.
/// \p ArgsCreator use to get call args from original call expr.
template <class... ArgsT>
std::shared_ptr<CallExprRewriterFactoryBase>
createTemplatedCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<TemplatedCallee(const CallExpr *)> CalleeCreator,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<
      CallExprRewriterFactory<TemplatedCallExprRewriter<ArgsT...>,
                              std::function<TemplatedCallee(const CallExpr *)>,
                              std::function<ArgsT(const CallExpr *)>...>>(
      SourceName,
      std::forward<std::function<TemplatedCallee(const CallExpr *)>>(
          CalleeCreator),
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

/// Create MemberCallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p BaseCreator use to get base expr from original call expr.
/// \p IsArrow the member operator is arrow or dot as default.
/// \p ArgsCreator use to get call args from original call expr.
template <class BaseT, class... ArgsT>
std::shared_ptr<CallExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const CallExpr *)> BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>,
      std::function<BaseT(const CallExpr *)>, bool, std::string,
      std::function<ArgsT(const CallExpr *)>...>>(
      SourceName,
      std::forward<std::function<BaseT(const CallExpr *)>>(BaseCreator),
      IsArrow, MemberName,
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class... ArgsT>
std::shared_ptr<CallExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName, BaseT BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>, BaseT, bool, std::string,
      std::function<ArgsT(const CallExpr *)>...>>(
      SourceName, BaseCreator, IsArrow, MemberName,
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

template <class... ArgsT>
std::shared_ptr<CallExprRewriterFactoryBase> createReportWarningRewriterFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        Factory,
    const std::string &FuncName, Diagnostics MsgId, ArgsT... ArgsCreator) {
  return std::make_shared<ReportWarningRewriterFactory<ArgsT...>>(
      Factory.second, FuncName, MsgId, ArgsCreator...);
}

template <class ArgT>
std::shared_ptr<CallExprRewriterFactoryBase>
createDeleterCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<ArgT(const CallExpr *)> &&ArgCreator) {
  return std::make_shared<CallExprRewriterFactory<
      DeleterCallExprRewriter<ArgT>, std::function<ArgT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgT(const CallExpr *)>>(ArgCreator));
}

template <class ArgT>
std::shared_ptr<CallExprRewriterFactoryBase> createToStringExprRewriterFactory(
    const std::string &SourceName,
    std::function<ArgT(const CallExpr *)> &&ArgCreator) {
  return std::make_shared<CallExprRewriterFactory<
      ToStringExprRewriter<ArgT>, std::function<ArgT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgT(const CallExpr *)>>(ArgCreator));
}

std::shared_ptr<CallExprRewriterFactoryBase>
createRemoveAPIRewriterFactory(const std::string &SourceName) {
  return std::make_shared<CallExprRewriterFactory<RemoveAPIRewriter>>(SourceName);
}

/// Create AssignableRewriterFactory key-value pair with inner key-value.
/// If the call expr's return value is used, will insert around "(" and ", 0)".
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<AssignableRewriterFactory>(Input.second));
}
/// Create AssignableRewriterFactory key-value pair with inner key-value.
/// If the call expr's return value is used, will insert around "(" and ", 0)".
template <class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createAssignableFactory(std::move(Input));
}

std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertAroundFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    std::string &&Prefix, std::string &&Suffix) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<InsertAroundRewriterFactory>(
          Input.second, std::move(Prefix), std::move(Suffix)));
}

/// Create RewriterFactoryWithFeatureRequest key-value pair with inner
/// key-value. Will call requestFeature when used to create CallExprRewriter.
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RewriterFactoryWithFeatureRequest>(Feature,
                                                          Input.second));
}
/// Create RewriterFactoryWithFeatureRequest key-value pair with inner
/// key-value. Will call requestFeature when used to create CallExprRewriter.
template <class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createFeatureRequestFactory(Feature, std::move(Input));
}


/// Create RewriterFactoryWithHeaderFile key-value pair with inner
/// key-value. Will call insertHeader when used to create CallExprRewriter.
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertHeaderFactory(
    HeaderType Header,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RewriterFactoryWithHeaderFile>(Header,
                                                          Input.second));
}
/// Create RewriterFactoryWithHeaderFile key-value pair with inner
/// key-value. Will call insertHeader when used to create CallExprRewriter.
template <class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertHeaderFactory(
    HeaderType Header,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createInsertHeaderFactory(Header, std::move(Input));
}

/// Create ConditonalRewriterFactory key-value pair with two key-value
/// candidates and predicate.
/// If predicate result is true, \p First will be used, else \p Second will be
/// used.
/// Also check the key of \p First and \p Second is same in debug build.
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createConditionalFactory(
    std::function<bool(const CallExpr *)> Pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&First,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Second) {
#ifdef DPCT_DEBUG_BUILD
  if (First.first != Second.first) {
    llvm::errs() << "Condtional factory has different name: [" << First.first
                 << "] : [" << Second.first << "]\n";
    assert(0 && "Condtional factory has different name");
  }
#endif // DPCT_DEBUG_BUILD
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(First.first), std::make_shared<ConditionalRewriterFactory>(
                                  Pred, First.second, Second.second));
}

/// Create ConditonalRewriterFactory key-value pair with two key-value
/// candidates and predicate.
/// If predicate result is true, \p First will be used, else \p Second will be
/// used.
/// Also check the key of \p First and \p Second is same in debug build.
template <class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createConditionalFactory(
    std::function<bool(const CallExpr *)> Pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&First,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Second,
    T) {
  return createConditionalFactory(std::move(Pred), std::move(First),
                                  std::move(Second));
}

std::function<bool(const CallExpr *)> makePointerChecker(unsigned Idx) {
  return [=](const CallExpr *C) -> bool {
    return C->getArg(Idx)->getType()->isPointerType();
  };
}

/// Create rewriter factory for migration of cudaBindTexture APIs.
/// \p StartIdx is represent the first available argument's index.
/// For cudaBindTexture and cudaBindTexture2D, it is 1.
/// For  cudaBindTextureToArray, it is 0.
/// The first predicate will check the \p StartIdx 'th argument whether is
/// pointer. If it is true, the call expr will be migrated to member call expr.
/// e.g.: cudaBindTexture(0, &tex, data, &desc, size) -> tex.attach(data, size,
/// desc) with template arguments: <1, 2>. Else will check the second predicate:
/// If \p Start + 2 'th argument's type whether is cudaChannelFormatDesc.
/// If it is true, e.g.: cudaBindTexture(0, tex, data, desc, size) ->
/// tex.attach(data, size, desc).
/// Else, e.g.: cudaBindTexture(0, tex, data, size) ->tex.attach(data, size,
/// desc).
template <size_t StartIdx, size_t... Idx>
std::shared_ptr<CallExprRewriterFactoryBase>
createBindTextureRewriterFactory(const std::string &Source) {
  std::function<bool(const CallExpr *)> TypeChecker =
      [=](const CallExpr *C) -> bool {
    if (C->getNumArgs() > StartIdx + 2)
      return DpctGlobalInfo::getUnqualifiedTypeName(
                 C->getArg(StartIdx + 2)->getType()) == "cudaChannelFormatDesc";
    return false;
  };

  return std::make_shared<ConditionalRewriterFactory>(
      makePointerChecker(StartIdx + 0),
      createMemberCallExprRewriterFactory(
          Source, makeDerefExprCreator(StartIdx + 0), true, "attach",
          makeCallArgCreator(StartIdx + 1),
          makeCallArgCreator(StartIdx + Idx + 1)...,
          makeDerefExprCreator(StartIdx + 2)),
      std::make_shared<ConditionalRewriterFactory>(
          TypeChecker,
          createMemberCallExprRewriterFactory(
              Source, makeCallArgCreatorWithCall(StartIdx + 0), false, "attach",
              makeCallArgCreatorWithCall(StartIdx + 1),
              makeCallArgCreatorWithCall(StartIdx + Idx + 1)...,
              makeCallArgCreatorWithCall(StartIdx + 2)),
          createMemberCallExprRewriterFactory(
              Source, makeCallArgCreatorWithCall(StartIdx + 0), false, "attach",
              makeCallArgCreatorWithCall(StartIdx + 1),
              makeCallArgCreatorWithCall(StartIdx + Idx)...)));
}

void setTextureInfo(const CallExpr *C, int TexType, int ObjIdx, QualType QT) {
  if (auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(C)) {
    if (auto ObjInfo =
            DeviceFunctionDecl::LinkRedecls(FD)
                ->addCallee(C)
                ->addTextureObjectArg(
                    ObjIdx, dyn_cast<DeclRefExpr>(
                                C->getArg(ObjIdx)->IgnoreImpCasts()))) {
      ObjInfo->setType(DpctGlobalInfo::getUnqualifiedTypeName(QT), TexType);
    }
  }
}

/// Create rewriter factory for texture reader APIs.
/// Predicate: check the first arg if is pointer and set texture info with
/// corresponding data. Migrate the call expr to an assign expr if Pred result
/// is true; e.g.: tex1D(&u, tex, 1.0f) -> u = tex.read(1.0f) Migrate the call
/// expr to an assign expr if Pred result is false; e.g.: tex1D(tex, 1.0f) ->
/// tex.read(1.0f) The template arguments is the member call arguments' index in
/// original call expr.
template <size_t... Idx>
std::shared_ptr<CallExprRewriterFactoryBase>
createTextureReaderRewriterFactory(const std::string &Source, int TextureType) {
  std::function<bool(const CallExpr *)> Pred = [=](const CallExpr *C) -> bool {
    if (C->getArg(0)->getType()->isPointerType()) {
      setTextureInfo(C, TextureType, 1,
                     C->getArg(0)->getType()->getPointeeType());
      return true;
    } else {
      setTextureInfo(C, TextureType, 0, C->getType());
      return false;
    }
  };
  return std::make_shared<ConditionalRewriterFactory>(
      Pred,
      creatBinaryOpRewriterFactory<BinaryOperatorKind::BO_Assign>(
          Source, makeDerefExprCreator(0),
          makeMemberCallCreator(makeCallArgCreator(1), false, "read",
                                makeCallArgCreator(Idx + 1)...)),
      createMemberCallExprRewriterFactory(Source, makeCallArgCreator(0), false,
                                          "read", makeCallArgCreator(Idx)...));
}

template <class... MsgArgs>
std::shared_ptr<CallExprRewriterFactoryBase>
createUnsupportRewriterFactory(const std::string &Source, Diagnostics MsgID,
                               MsgArgs &&...Args) {
  return std::make_shared<UnsupportFunctionRewriterFactory<MsgArgs...>>(
      Source, MsgID, std::forward<MsgArgs>(Args)...);
}

class CheckWarning1073 {
  unsigned Idx;

public:
  CheckWarning1073(unsigned I) : Idx(I) {}
  bool operator()(const CallExpr *C) {
    auto DerefE = getDereferencedExpr(C->getArg(Idx));
    return DerefE && isa<DeclRefExpr>(DerefE);
  }
};

std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedRewriterFactory(const std::string &Source, MetaRuleObject &R) {
  return std::make_shared<UserDefinedRewriterFactory>(R);
}

std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedMethodRewriterFactory(
    const std::string &Source, MetaRuleObject &R,
    std::shared_ptr<MetaRuleObject::ClassMethod> MethodPtr) {
  return std::make_shared<UserDefinedRewriterFactory>(R, MethodPtr);
}

// sycl has 2 overloading of malloc_device
// 1. sycl::malloc_device(Addr, Size)
// 2. sycl::malloc_device<type>(Addr, Size)
// DPCT will use the template version if following constraints hold:
// 1. The Addr can be derefed twice. The derefed type is type_1
// 2. The Size argument contains sizeof(type_2)
// 3. type_1 and type_2 are the same
// 4. The Size argument does not contain macro
class CheckCanUseTemplateMalloc {
  unsigned AddrArgIdx;
  unsigned SizeArgIdx;

public:
  CheckCanUseTemplateMalloc(unsigned AddrIdx, unsigned SizeIdx)
      : AddrArgIdx(AddrIdx), SizeArgIdx(SizeIdx) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() <= AddrArgIdx)
      return false;
    auto AllocatedExpr = C->getArg(AddrArgIdx);
    const Expr *AE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
      AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    } else {
      AE = AllocatedExpr;
    }

    // Try to deref twice to avoid the type is an unresolved template
    QualType DerefQT = AE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return false;
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return false;

    if (C->getNumArgs() <= SizeArgIdx)
      return false;
    auto SizeExpr = C->getArg(SizeArgIdx);

    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
    auto BO = dyn_cast<BinaryOperator>(SizeExpr);
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
      std::string Repl;
      if (!isContainMacro(BO->getLHS()) &&
          isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
        // case 1: sizeof(b) * a
        return true;
      } else if (!isContainMacro(BO->getRHS()) &&
                 isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
        // case 2: a * sizeof(b)
        return true;
      }
      return false;
    } else {
      // case 3: sizeof(b)
      if (!isContainMacro(SizeExpr) &&
          isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
        return true;
      }
    }

    return false;
  }
};

class CheckArgCount {
  unsigned Count;

public:
  CheckArgCount(unsigned I) : Count(I) {}
  bool operator()(const CallExpr *C) { return C->getNumArgs() == Count; }
};

class CheckBaseType {
  std::string TypeName;

public:
  CheckBaseType(std::string Name) : TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    auto BaseType = getBaseTypeStr(C);
    if (BaseType.empty())
      return false;
    return TypeName == BaseType;
  }
};

auto UseNDRangeBarrier = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useNdRangeBarrier();
};
auto UseLogicalGroup = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useLogicalGroup();
};

class CheckDerefedTypeBeforeCast {
  unsigned Idx;
  std::string TypeName;

public:
  CheckDerefedTypeBeforeCast(unsigned I, std::string Name)
      : Idx(I), TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() > Idx) {
      std::ostringstream OS;
      std::string Type;
      printDerefOp(OS, C->getArg(Idx)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != TypeName) {
        return false;
      }
    }
    return true;
  }
};

class CheckArgIsConstantIntWithValue {
  int value;
  int index;

public:
  CheckArgIsConstantIntWithValue(int idx, int val) : value(val), index(idx) {}
  bool operator()(const CallExpr *C) {
    auto Arg = C->getArg(index);
    Expr::EvalResult Result;
    if (!Arg->isValueDependent() &&
        Arg->EvaluateAsInt(Result, DpctGlobalInfo::getContext()) &&
        Result.Val.getInt().getSExtValue() == value) {
      return true;
    }
    return false;
  }
};

class CheckIsPtr {
  unsigned Idx;

public:
  CheckIsPtr(unsigned I) : Idx(I) {}
  // Normally, we will deref the ptr after we know it's a ptr,
  // so this check should return false in cases like template.
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() > Idx) {
      if (!C->getDirectCallee())
        return false;
      if (!C->getDirectCallee()->getParamDecl(Idx))
        return false;
      return C->getDirectCallee()
          ->getParamDecl(Idx)
          ->getType()
          ->isPointerType();
    }
    return false;
  }
};

template <class F, class S> class CheckAnd {
  F Fir;
  S Sec;

public:
  CheckAnd(F Fir, S Sec) : Fir(Fir), Sec(Sec) {}
  bool operator()(const CallExpr *C) { return Fir(C) && Sec(C); }
};

template <class F, class S> class CheckOr {
  F Fir;
  S Sec;

public:
  CheckOr(F Fir, S Sec) : Fir(Fir), Sec(Sec) {}
  bool operator()(const CallExpr *C) { return Fir(C) || Sec(C); }
};

template <class F, class S> CheckAnd<F, S> makeCheckAnd(F Fir, S Sec) {
  return CheckAnd<F, S>(Fir, Sec);
}

template <class F, class S> CheckOr<F, S> makeCheckOr(F Fir, S Sec) {
  return CheckOr<F, S>(Fir, Sec);
}

template <class T> class CheckNot {
  T Expr;

public:
  CheckNot(T Expr) : Expr(Expr) {}
  bool operator()(const CallExpr *C) { return !Expr(C); }
};

template <class T> CheckNot<T> makeCheckNot(T Expr) {
  return CheckNot<T>(Expr);
}

class CompareArgType {
  unsigned Idx1, Idx2;

public:
  CompareArgType(unsigned I1, unsigned I2) : Idx1(I1), Idx2(I2) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() > Idx1 && C->getNumArgs() > Idx2) {
      if (!C->getDirectCallee())
        return true;
      if (!C->getDirectCallee()->getParamDecl(Idx1))
        return true;
      if (!C->getDirectCallee()->getParamDecl(Idx2))
        return true;
      std::string ArgType1 = C->getDirectCallee()
                                 ->getParamDecl(Idx1)
                                 ->getType()
                                 .getCanonicalType()
                                 .getUnqualifiedType()
                                 .getAsString();
      std::string ArgType2 = C->getDirectCallee()
                                 ->getParamDecl(Idx2)
                                 ->getType()
                                 .getCanonicalType()
                                 .getUnqualifiedType()
                                 .getAsString();
      return ArgType1 != ArgType2;
    }
    return true;
  }
};

std::function<std::string(const CallExpr *)> MemberExprBase() {
  return [=](const CallExpr *C) -> std::string {
    auto ME = dyn_cast<MemberExpr>(C->getCallee()->IgnoreImpCasts());
    if (!ME)
      return "";
    auto Base = ME->getBase()->IgnoreImpCasts();
    if (!Base)
      return "";
    return getStmtSpelling(Base);
  };
}

#define ASSIGNABLE_FACTORY(x) createAssignableFactory(x 0),
#define INSERT_AROUND_FACTORY(x, PREFIX, SUFFIX)                               \
  createInsertAroundFactory(x PREFIX, SUFFIX),
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define HEADER_INSERT_FACTORY(HEADER, x)                                       \
  createInsertHeaderFactory(HEADER, x 0),
#define SUBGROUPSIZE_FACTORY(IDX, NEWFUNCNAME, x)                              \
  createFactoryWithSubGroupSizeRequest<IDX>(NEWFUNCNAME, x 0),
#define STREAM(x) makeDerefStreamExprCreator(x)
#define DEREF(x) makeDerefExprCreator(x)
#define STRUCT_DISMANTLE(idx, ...) makeStructDismantler(idx, {__VA_ARGS__})
#define ARG(x) makeCallArgCreator(x)
#define ARG_WC(x) makeDerefArgCreatorWithCall(x)
#define BLAS_ENUM_ARG(x, BLAS_ENUM_TYPE)                                       \
  makeBLASEnumCallArgCreator(x, BLAS_ENUM_TYPE)
#define EXTENDSTR(idx, str) makeExtendStr(idx, str)
#define QUEUESTR makeQueueStr()
#define BO(Op, L, R) makeBinaryOperatorCreator<Op>(L, R)
#define MEMBER_CALL(...) makeMemberCallCreator(__VA_ARGS__)
#define LAMBDA(...) makeLambdaCreator(__VA_ARGS__)
#define CALL(...) makeCallExprCreator(__VA_ARGS__)
#define CAST(T, S) makeCastExprCreator(T, S)
#define NEW(...) makeNewExprCreator(__VA_ARGS__)
#define SUBGROUP                                                               \
  std::function<SubGroupPrinter(const CallExpr *)>(SubGroupPrinter::create)
#define NDITEM std::function<ItemPrinter(const CallExpr *)>(ItemPrinter::create)
#define GROUP                                                                  \
  std::function<GroupPrinter(const CallExpr *)>(GroupPrinter::create)
#define POINTER_CHECKER(x) makePointerChecker(x)
#define LITERAL(x) makeLiteral(x)
#define TEMPLATED_CALLEE(FuncName, ...)                                        \
  makeTemplatedCalleeCreator(FuncName, {__VA_ARGS__})
#define TEMPLATED_CALLEE_WITH_ARGS(FuncName, ...)                              \
  makeTemplatedCalleeWithArgsCreator(FuncName, __VA_ARGS__)
#define CONDITIONAL_FACTORY_ENTRY(Pred, First, Second)                         \
  createConditionalFactory(Pred, First Second 0),
#define IFELSE_FACTORY_ENTRY(FuncName, Pred, IfBlock, ElseBlock)               \
  {FuncName, creatIfElseRewriterFactory(FuncName, Pred IfBlock ElseBlock 0)},
#define TEMPLATED_CALL_FACTORY_ENTRY(FuncName, ...)                            \
  {FuncName, createTemplatedCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define ASSIGN_FACTORY_ENTRY(FuncName, L, R)                                   \
  {FuncName, creatBinaryOpRewriterFactory<BinaryOperatorKind::BO_Assign>(      \
                 FuncName, L, R)},
#define BINARY_OP_FACTORY_ENTRY(FuncName, OP, L, R)                            \
  {FuncName, creatBinaryOpRewriterFactory<OP>(FuncName, L, R)},
#define MEM_EXPR_ENTRY(FuncName, B, IsArrow, M)                                \
  {FuncName, creatMemberExprRewriterFactory(FuncName, B, IsArrow, M)},
#define CALL_FACTORY_ENTRY(FuncName, C)                                        \
  {FuncName, creatCallExprRewriterFactory(FuncName, C)},
#define MEMBER_CALL_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMemberCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define DELETER_FACTORY_ENTRY(FuncName, Arg)                                   \
  {FuncName, createDeleterCallExprRewriterFactory(FuncName, Arg)},
#define UNSUPPORT_FACTORY_ENTRY(FuncName, MsgID, ...)                          \
  {FuncName, createUnsupportRewriterFactory(FuncName, MsgID, __VA_ARGS__)},
#define MULTI_STMTS_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMultiStmtsRewriterFactory(FuncName, __VA_ARGS__)},
#define WARNING_FACTORY_ENTRY(FuncName, Factory, ...)                          \
  {FuncName, createReportWarningRewriterFactory(Factory FuncName, __VA_ARGS__)},
#define TOSTRING_FACTORY_ENTRY(FuncName, ...)                                  \
  {FuncName, createToStringExprRewriterFactory(FuncName, __VA_ARGS__)},
#define REMOVE_API_FACTORY_ENTRY(FuncName)                                  \
  {FuncName, createRemoveAPIRewriterFactory(FuncName)},

///***************************************************************
/// Examples:
/// cudaBindTexture(0, &tex21, d_data21, &desc21, 32 * sizeof(uint2))
/// -> tex21.attach(d_data21, 32 * sizeof(uint32)
/// using: MEMBER_CALL_FACTORY_ENTRY("cudaBindTexture", DEREF(1), true/* member
/// operator default is arrow*/, "attach", ARG(2), ARG(4), DEREF(3))
///
/// tex2DLayered(&data, tex, 2.0f, 2.0f, 10) -> data = tex.read(10, 2.0f, 2.0f)
/// using: BINARY_OP_FACTORY_ENTRY("tex2DLayered",
/// BinaryOperatorKind::BO_Assign, DEREF(0), MEMBER_ALL(ARG(1), false/* member
/// operator default is dot */, "read", ARG(4), ARG(2), ARG(3))
///
/// Macro ASSIGNABLE_FACTORY(x) is used for migration of call expr that has
/// valid return value.
///
/// Macro CONDITIONAL_FACTORY_ENTRY(pred, first, second) is used for conditonal
/// migration. \p pred is expr that can convert to std::function<bool(const
/// CallExpr *)>. \p first and \p second is two candidate factories. If \p pred
/// result is true, \p first will be used to create rewrite, else \p second will
/// be used.
///***************************************************************

#define TEX_FUNCTION_FACTORY_ENTRY(FuncName, TexType, ...)                     \
  {FuncName,                                                                   \
   createTextureReaderRewriterFactory<__VA_ARGS__>(FuncName, TexType)},
#define BIND_TEXTURE_FACTORY_ENTRY(FuncName, ...)                              \
  {FuncName, createBindTextureRewriterFactory<__VA_ARGS__>(FuncName)},

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define MATH_FUNCNAME_FACTORY_ENTRY(FuncName, RewriterName)                    \
  REWRITER_FACTORY_ENTRY(FuncName, MathFuncNameRewriterFactory, RewriterName)
#define NO_REWRITE_FUNCNAME_FACTORY_ENTRY(FuncName, RewriterName)              \
  REWRITER_FACTORY_ENTRY(FuncName, NoRewriteFuncNameRewriterFactory,           \
                         RewriterName)
#define MATH_SIMULATED_FUNC_FACTORY_ENTRY(FuncName, RewriterName)              \
  REWRITER_FACTORY_ENTRY(FuncName, MathSimulatedRewriterFactory, RewriterName)
#define MATH_TYPECAST_FACTORY_ENTRY(FuncName)                                  \
  REWRITER_FACTORY_ENTRY(FuncName, MathTypeCastRewriterFactory, FuncName)
#define MATH_BO_FACTORY_ENTRY(FuncName, OpKind)                                \
  REWRITER_FACTORY_ENTRY(FuncName, MathBinaryOperatorRewriterFactory, OpKind)
#define MATH_UNSUPPORTED_FUNC_FACTORY_ENTRY(FuncName)                          \
  REWRITER_FACTORY_ENTRY(FuncName, MathUnsupportedRewriterFactory, FuncName)
#define WARP_FUNC_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, WarpFunctionRewriterFactory, RewriterName)
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportFunctionRewriterFactory<>, MsgID)

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
    CallExprRewriterFactoryBase::RewriterMap;

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
    CallExprRewriterFactoryBase::MethodRewriterMap;

void CallExprRewriterFactoryBase::initRewriterMap() {
  RewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)                 \
  NO_REWRITE_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)                           \
  MATH_SIMULATED_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND) MATH_BO_FACTORY_ENTRY(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME) MATH_TYPECAST_FACTORY_ENTRY(APINAME)
#define ENTRY_UNSUPPORTED(APINAME) MATH_UNSUPPORTED_FUNC_FACTORY_ENTRY(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED

#define ENTRY_WARP(SOURCEAPINAME, TARGETAPINAME)                               \
  WARP_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)

#undef ENTRY_WARP

#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_TEXTURE(SOURCEAPINAME, TEXTYPE, ...)                             \
  TEX_FUNCTION_FACTORY_ENTRY(SOURCEAPINAME, TEXTYPE, __VA_ARGS__)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#define ENTRY_BIND(SOURCEAPINAME, ...)                                         \
  BIND_TEXTURE_FACTORY_ENTRY(SOURCEAPINAME, __VA_ARGS__)
#define ENTRY_TEMPLATED(SOURCEAPINAME, ...)                                    \
  TEMPLATED_CALL_FACTORY_ENTRY(SOURCEAPINAME, __VA_ARGS__)
#include "APINamesCUBLAS.inc"
#include "APINamesCUFFT.inc"
#include "APINamesCURAND.inc"
#include "APINamesComplex.inc"
#include "APINamesDriver.inc"
#include "APINamesMemory.inc"
#include "APINamesNccl.inc"
#include "APINamesStream.inc"
#include "APINamesTexture.inc"
#include "APINamesThrust.inc"
#include "APINamesWarp.inc"
#include "APINamesCUDNN.inc"
#include "APINamesErrorHandling.inc"
#define FUNCTION_CALL
#define CLASS_METHOD_CALL
#include "APINamesCooperativeGroups.inc"
#undef FUNCTION_CALL
#undef CLASS_METHOD_CALL
#undef ENTRY_RENAMED
#undef ENTRY_TEXTURE
#undef ENTRY_UNSUPPORTED
#undef ENTRY_TEMPLATED
#undef ENTRY_BIND

#define ENTRY_HOST(from, to, policy)
#define ENTRY_DEVICE(SOURCEAPINAME, TARGETAPINAME, EXTR)                       \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_BOTH(SOURCEAPINAME, TARGETAPINAME, EXTR)                         \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesMapThrust.inc"
#undef ENTRY_HOST
#undef ENTRY_DEVICE
#undef ENTRY_BOTH
      }));
}

void CallExprRewriterFactoryBase::initMethodRewriterMap() {
  MethodRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define CLASS_METHOD_CALL
#include "APINamesCooperativeGroups.inc"
#undef CLASS_METHOD_CALL
      }));
}

const std::vector<std::string> MathFuncNameRewriter::SingleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
};

const std::vector<std::string> MathFuncNameRewriter::DoubleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
};

} // namespace dpct
} // namespace clang
