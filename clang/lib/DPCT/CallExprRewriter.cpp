//===--- CallExprRewriter.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//


#include "clang/AST/Attr.h"
#include "CallExprRewriter.h"
#include "AnalysisInfo.h"
#include "MapNames.h"

namespace clang {
namespace dpct {

std::string CallExprRewriter::getMigratedArg(unsigned Idx) {
  Analyzer.analyze(Call->getArg(Idx));
  return Analyzer.getRewritePrefix() + Analyzer.getReplacedString() +
         Analyzer.getRewritePostfix();
}

std::vector<std::string> CallExprRewriter::getMigratedArgs() {
  std::vector<std::string> ArgList;
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
  auto FilePath = DpctGlobalInfo::getLocInfo(FD).first;
  if (isChildOrSamePath(DpctGlobalInfo::getInRoot(), FilePath))
    return false;
  return true;
}

Optional<std::string> MathFuncNameRewriter::rewrite() {
  // If the function is not a target math function, do not migrate it
  if (!isTargetMathFunction(Call->getDirectCallee())) {
    RewriteArgList = getMigratedArgs();
    setTargetCalleeName(getSourceCalleeName().str());
    return buildRewriteString();
  }

  reportUnsupportedRoundingMode();
  RewriteArgList = getMigratedArgs();
  setTargetCalleeName(getNewFuncName());
  // When the function name is in macro define,
  // add extra replacement to migrate the function name
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto It = DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
    SM.getCharacterData(Call->getCallee()->getBeginLoc()));
  if (It != DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    ExprAnalysis EA;
    EA.analyze(Call->getCallee());
    EA.addReplacement(Call->getCallee(), getNewFuncName());
    if (auto R = EA.getReplacement()) {
    DpctGlobalInfo::getInstance().addReplacement(
      R->getReplacement(DpctGlobalInfo::getContext()));
    }
  }
  return buildRewriteString();
}

/// Policies to migrate math functions:
/// 1) Functions with the "std" namespace are treated as host functions;
/// 2) Functions with __device__ attribute but without __host__
///    attribute are treated as device functions;
/// 3) Functions whose calling functions are augmented with __device__
///    or __global__ attributes are treated as device functions;
/// 4) Other functions are treated as host functions.
///    eg. "__host__ __device__ fabs()" falls in 4) if fabs is not called in device or kernel
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

    auto ContextFD = getImmediateOuterFuncDecl(Call);
    if (NamespaceStr == "std" && ContextFD &&
        !ContextFD->hasAttr<CUDADeviceAttr>() &&
        !ContextFD->hasAttr<CUDAGlobalAttr>()) {
      NewFuncName = "std::" + SourceCalleeName.str();
    }
    // For device functions
    else if ((FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAHostAttr>()) ||
             (ContextFD && (ContextFD->hasAttr<CUDADeviceAttr>() ||
                            ContextFD->hasAttr<CUDAGlobalAttr>()))) {
      if (SourceCalleeName == "abs") {
        // further check the type of the args.
        if (!Call->getArg(0)->getType()->isIntegerType()) {
          NewFuncName = MapNames::getClNamespace() + "::fabs";
        }
      }

      if (SourceCalleeName == "min" || SourceCalleeName == "max") {
        LangOptions LO;
        std::string FT = Call->getType().getAsString(PrintingPolicy(LO));
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i);
          auto ArgExpr = Arg->getStmtClass();
          if (ArgExpr == Stmt::PseudoObjectExprClass) {
            auto POE = dyn_cast<PseudoObjectExpr>(Arg->IgnoreImpCasts());
            auto RE = POE->getResultExpr();
            if (auto CE = dyn_cast<CallExpr>(RE)) {
              auto FD = CE->getDirectCallee();
              auto Name = FD->getNameAsString();
              // Force typecast threadIdx/blockIdx/blockDim./x/y/z to return
              // types of math functions
              if (Name == "__fetch_builtin_x" || Name == "__fetch_builtin_y" ||
                  Name == "__fetch_builtin_z") {
                RewriteArgList[i] = "(" + FT + ")" + RewriteArgList[i];
              }
            }
          } else {
            std::string ArgT = Arg->getType().getAsString(PrintingPolicy(LO));
            auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
            if (ArgT != FT || ArgExpr == Stmt::BinaryOperatorClass) {
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
        std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
            PrintingPolicy(LO));
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
                 SourceCalleeName == "__mulhi" || SourceCalleeName == "__hadd") {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i);
          std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
              PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          if (ArgT != "int") {
            if (DRE || IL)
              RewriteArgList[i] = "(int)" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(int)(" + RewriteArgList[i] + ")";
          }
        }
      }

      if (std::find(SingleFuctions.begin(), SingleFuctions.end(),
                    SourceCalleeName) != SingleFuctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          if (SourceCalleeName == "ldexpf" && i == 1)
            continue;
          auto Arg = Call->getArg(i);
          std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
              PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "float";
          auto PVD = FD->getParamDecl(i);
          if (PVD)
            ParamType = PVD->getType().getAsString();
          // Since isnan is overloaded for both float and double, so there is no
          // need to add type conversions for isnan.
          if (ArgT != ParamType && SourceCalleeName != "isnan") {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(" + ParamType + ")(" + RewriteArgList[i] + ")";
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
          std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
              PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "double";
          auto PVD = FD->getParamDecl(i);
          if (PVD)
            ParamType = PVD->getType().getAsString();
          if (ArgT != ParamType) {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(" + ParamType + ")(" + RewriteArgList[i] + ")";
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
      DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(), Math);
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
        // Deal with cases where types of arguements are typedefs, e.g.,
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
              // Convert signed types to unsigned types if the bitwidth of
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
                                                         Algorithm);
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
  report(Diagnostics::NOTSUPPORTED, false,
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
       << ".convert<" + MapNames::getClNamespace() + "::half, " +
              MapNames::getClNamespace() + "::rounding_mode::rte>()";
  } else if (FuncName == "__float2half2_rn") {
    OS << MapNames::getClNamespace() + "::float2{" << MigratedArg0 << ","
       << MigratedArg0
       << "}.convert<" + MapNames::getClNamespace() + "::half, " +
              MapNames::getClNamespace() + "::rounding_mode::rte>()";
  } else if (FuncName == "__floats2half2_rn") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "::float2{" << MigratedArg0 << ","
       << MigratedArg1
       << "}.convert<" + MapNames::getClNamespace() + "::half, " +
              MapNames::getClNamespace() + "::rounding_mode::rte>()";
  } else if (FuncName == "__half22float2") {
    OS << MigratedArg0
       << ".convert<float, " + MapNames::getClNamespace() +
              "::rounding_mode::automatic>()";
  } else if (FuncName == "__half2half2") {
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0 << ","
       << MigratedArg0 << "}";
  } else if (FuncName == "__halves2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0 << ","
       << MigratedArg1 << "}";
  } else if (FuncName == "__high2float") {
    OS << MigratedArg0 << "[0]";
  } else if (FuncName == "__high2half") {
    OS << MigratedArg0 << "[0]";
  } else if (FuncName == "__high2half2") {
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0
       << "[0], " << MigratedArg0 << "[0]}";
  } else if (FuncName == "__highs2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0
       << "[0], " << MigratedArg1 << "[0]}";
  } else if (FuncName == "__low2float") {
    OS << MigratedArg0 << "[1]";
  } else if (FuncName == "__low2half") {
    OS << MigratedArg0 << "[1]";
  } else if (FuncName == "__low2half2") {
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0
       << "[1], " << MigratedArg0 << "[1]}";
  } else if (FuncName == "__lowhigh2highlow") {
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0
       << "[1], " << MigratedArg0 << "[0]}";
  } else if (FuncName == "__lows2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "::half2{" << MigratedArg0
       << "[1], " << MigratedArg1 << "[1]}";
  } else {
    //__half2short_rd and __half2float
    static SSMap TypeMap{{"ll", "long long"},
                         {"ull", "unsigned long long"},
                         {"ushort", "unsigned short"},
                         {"uint", "unsigned int"},
                         {"half", MapNames::getClNamespace() + "::half"}};
    std::string RoundingMode;
    if (FuncName[FuncName.size() - 3] == '_')
      RoundingMode = FuncName.substr(FuncName.size() - 2).str();
    auto FN = FuncName.substr(2, FuncName.find('_', 2) - 2).str();
    auto Types = split(FN, '2');
    assert(Types.size() == 2);
    MapNames::replaceName(TypeMap, Types[0]);
    MapNames::replaceName(TypeMap, Types[1]);
    OS << MapNames::getClNamespace() + "::vec<" << Types[0] << ", 1>{"
       << MigratedArg0 << "}.convert<" << Types[1]
       << ", " + MapNames::getClNamespace() + "::rounding_mode::"
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

  auto FD = Call->getDirectCallee();
  if (!FD)
    return Base::rewrite();

  auto ContextFD = getImmediateOuterFuncDecl(Call);
  if (NamespaceStr == "std" && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>()) {
    std::string NewFuncName = "std::" + SourceCalleeName.str();
    SourceCalleeName = StringRef(NewFuncName);
    return Base::rewrite();
  }

  if (!FD->hasAttr<CUDADeviceAttr>() && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>())
    return Base::rewrite();

  // Do need to report warnings for pow migration
  if (SourceCalleeName != "pow" && SourceCalleeName != "powf"&&
      SourceCalleeName != "__powf")
    report(Diagnostics::MATH_EMULATION, false,
           MapNames::ITFName.at(SourceCalleeName.str()), TargetCalleeName);

  const std::string FuncName = SourceCalleeName.str();
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  auto MigratedArg0 = getMigratedArg(0);

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
    OS << MapNames::getClNamespace() + "::frexp(" << MigratedArg0
       << ", " + MapNames::getClNamespace() + "::make_ptr<int, "
       << MapNames::getClNamespace() +
              "::access::address_space::global_space>("
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
    OS << MapNames::getClNamespace() + "::modf(" << MigratedArg0;
    if (FuncName == "modf")
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<double, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    else
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<float, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    OS << MigratedArg1 << "))";
  } else if (FuncName == "nan" || FuncName == "nanf") {
    OS << MapNames::getClNamespace() + "::nan(0u)";
  } else if (FuncName == "sincos" || FuncName == "sincosf" ||
             FuncName == "__sincosf") {
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
      OS << MigratedArg1.substr(1);
    else
      OS << "*(" + MigratedArg1 + ")";
    OS << " = " + MapNames::getClNamespace() + "::sincos("
       << MigratedArg0;
    if (FuncName == "sincos")
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<double, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    else
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<float, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    OS << MigratedArg2 << "))";
  } else if (FuncName == "sincospi" || FuncName == "sincospif") {
    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      OS << MigratedArg1.substr(1);
    else
      OS << "*(" + MigratedArg1 + ")";
    OS << " = " + MapNames::getClNamespace() + "::sincos("
       << MigratedArg0;
    if (FuncName == "sincospi")
      OS << " * DPCT_PI";
    else
      OS << " * DPCT_PI_F";

    if (FuncName == "sincospi")
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<double, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    else
      OS << ", " + MapNames::getClNamespace() + "::make_ptr<float, " +
                MapNames::getClNamespace() +
                "::access::address_space::global_space>(";
    OS << MigratedArg2 << "))";
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
    OS << MapNames::getClNamespace() + "::remquo(" << MigratedArg0 << ", "
       << MigratedArg1
       << ", " + MapNames::getClNamespace() + "::make_ptr<int, " +
              MapNames::getClNamespace() +
              "::access::address_space::global_space>("
       << MigratedArg2 << "))";
  } else if (FuncName == "nearbyint" || FuncName == "nearbyintf") {
    OS << MapNames::getClNamespace() + "::floor(" << MigratedArg0
       << " + 0.5)";
  } else if (FuncName == "rhypot" || FuncName == "rhypotf") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "1 / " + MapNames::getClNamespace() + "::hypot(" << MigratedArg0
       << ", " << MigratedArg1 << ")";
  } else if (SourceCalleeName == "pow" || SourceCalleeName == "powf" ||
             SourceCalleeName == "__powf") {
    RewriteArgList = getMigratedArgs();
    if (Call->getNumArgs() != 2) {
      TargetCalleeName = SourceCalleeName.str();
      return buildRewriteString();
    }
    LangOptions LO;
    auto Arg0 = Call->getArg(0);
    auto Arg1 = Call->getArg(1);
    auto T0 = Arg0->IgnoreCasts()->getType().getAsString(PrintingPolicy(LO));
    auto T1 = Arg1->IgnoreCasts()->getType().getAsString(PrintingPolicy(LO));
    auto IL1 = dyn_cast<IntegerLiteral>(Arg1->IgnoreCasts());
    auto DRE0 = dyn_cast<DeclRefExpr>(Arg0->IgnoreCasts());
    // For integer literal 2, expand to multiply expression:
    // pow(x, 2) ==> x * x.
    if (IL1 && DRE0 && IL1->getValue().getZExtValue() == 2) {
      auto Arg0Str = ExprAnalysis::ref(Arg0);
      return Arg0Str + " * " + Arg0Str;
    }
    // For i of integer type or integer literals, migrate to sycl::pown:
    // pow(x, i) ==> pown(x, i);
    // otherwise, migrate to sycl::pow(x, i).
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
      TargetCalleeName =  MapNames::getClNamespace() + "::pown";
    } else if (T1 == "long" || T1 == "unsigned long" || T1 == "long long" ||
               T1 == "unsigned long long") {
      if (DRE0)
        RewriteArgList[1] = "(float)" + RewriteArgList[1];
      else
        RewriteArgList[1] = "(float)(" + RewriteArgList[1] + ")";
    } else if (T1 == "float") {
      if (T0 == "int") {
        if (DRE0)
          RewriteArgList[0] = "(float)" + RewriteArgList[0];
        else
          RewriteArgList[0] = "(float)(" + RewriteArgList[0] + ")";
      } else if (T0 == "double") {
        if (DRE0)
          RewriteArgList[1] = "(double)" + RewriteArgList[1];
        else
          RewriteArgList[1] = "(double)(" + RewriteArgList[1] + ")";
      }
      TargetCalleeName = MapNames::getClNamespace() + "::pow";
    } else if (T1 == "double") {
      if (T0 == "int" || T0 == "float") {
        if (DRE0)
          RewriteArgList[0] = "(double)" + RewriteArgList[0];
        else
          RewriteArgList[0] = "(double)(" + RewriteArgList[0] + ")";
      }
      TargetCalleeName = MapNames::getClNamespace() + "::pow";
    }
    return buildRewriteString();
  } else if (FuncName == "erfcx" || FuncName == "erfcxf") {
    OS << "sycl::exp(" << MigratedArg0 << "*" << MigratedArg0 << ")*"
       << TargetCalleeName << "(" << MigratedArg0 << ")";
  } else if (FuncName == "norm3d" || FuncName == "norm3df") {
    OS << TargetCalleeName << "(sycl::float3(" << MigratedArg0 << ", "
       << getMigratedArg(1) << ", " << getMigratedArg(2) << "))";
  } else if (FuncName == "norm4d" || FuncName == "norm4df") {
    OS << TargetCalleeName << "(sycl::float4(" << MigratedArg0 << ", "
       << getMigratedArg(1) << ", " << getMigratedArg(2) << ", "
       << getMigratedArg(3) << "))";
  } else if (FuncName == "rcbrt" || FuncName == "rcbrtf") {
    OS << "sycl::native::recip((float)" << TargetCalleeName << "(" << getMigratedArg(0) << "))";
  } else if (FuncName == "rnorm3d" || FuncName == "rnorm3df") {
    OS << "sycl::native::recip(" << TargetCalleeName << "(sycl::float3("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << ")))";
  } else if (FuncName == "rnorm4d" || FuncName == "rnorm4df") {
    OS << "sycl::native::recip(" << TargetCalleeName << "(sycl::float4("
       << MigratedArg0 << ", " << getMigratedArg(1) << ", " << getMigratedArg(2)
       << ", " << getMigratedArg(3) << ")))";
  } else if (FuncName == "scalbln" || FuncName == "scalblnf" ||
             FuncName == "scalbn" || FuncName == "scalbnf") {
    OS << MigratedArg0 << "*(2<<" << getMigratedArg(1) << ")";
  } else if (FuncName == "__double2hiint") {
    OS << "dpct::cast_double_to_int(" << MigratedArg0 << ")";
  } else if (FuncName == "__double2loint") {
    OS << "dpct::cast_double_to_int(" << MigratedArg0 << ", false)";
  } else if (FuncName == "__hiloint2double") {
    OS << "dpct::cast_ints_to_double(" << MigratedArg0 << ", " << getMigratedArg(1) << ")";
  } else if (FuncName == "__sad" || FuncName == "__usad") {
    OS << TargetCalleeName << "(" << MigratedArg0 << ", " << getMigratedArg(1)
       << ")" << "+" << getMigratedArg(2);
  } else if (FuncName == "__drcp_rd" || FuncName == "__drcp_rn" ||
             FuncName == "__drcp_ru" || FuncName == "__drcp_rz") {
    auto Arg0 = Call->getArg(0);
    auto T0 = Arg0->IgnoreCasts()->getType().getAsString(PrintingPolicy(LangOptions()));
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
        OS << TargetCalleeName << "(sycl::float2(" << MigratedArg1
           << "[0], " << MigratedArg1 << "[1]))";
        break;
      case 3:
        OS << TargetCalleeName << "(sycl::float3(" << MigratedArg1
           << "[0], " << MigratedArg1 << "[1], " << MigratedArg1 << "[2]))";
        break;
      case 4:
        OS << TargetCalleeName << "(sycl::float4(" << MigratedArg1
           << "[0], " << MigratedArg1 << "[1], " << MigratedArg1 << "[2], "
           << MigratedArg1 << "[3]))";
        break;
      default:
        OS << "dpct::fast_length(" << "(float *)" << getMigratedArg(1) << ", "
           << MigratedArg0 << ")";
      }
    } else {
      OS << "dpct::fast_length(" << "(float *)" << getMigratedArg(1) << ", "
         << MigratedArg0 << ")";
    }
  }

  OS.flush();
  return ReplStr;
}

Optional<std::string> MathBinaryOperatorRewriter::rewrite() {
  reportUnsupportedRoundingMode();
  if (SourceCalleeName == "__hneg" || SourceCalleeName == "__hneg2") {
    setLHS("");
    setRHS(getMigratedArg(0));
  } else {
    setLHS(getMigratedArg(0));
    setRHS(getMigratedArg(1));
  }
  return buildRewriteString();
}

Optional<std::string> WarpFunctionRewriter::rewrite() {
  if (SourceCalleeName == "__activemask" || SourceCalleeName == "__ballot" ||
      SourceCalleeName == "__ballot_sync") {
    report(Diagnostics::NOTSUPPORTED, false,
           MapNames::ITFName.at(SourceCalleeName.str()));
    RewriteArgList = getMigratedArgs();
    setTargetCalleeName(SourceCalleeName.str());
  } else {
    if (SourceCalleeName == "__all" || SourceCalleeName == "__any") {
      RewriteArgList.emplace_back(DpctGlobalInfo::getItemName() + ".get_group()");
      RewriteArgList.emplace_back(getMigratedArg(0));
    } else if (SourceCalleeName == "__all_sync" ||
               SourceCalleeName == "__any_sync") {
      reportNoMaskWarning();
      RewriteArgList.emplace_back(DpctGlobalInfo::getItemName() + ".get_group()");
      RewriteArgList.emplace_back(getMigratedArg(1));
      setTargetCalleeName(
          MapNames::findReplacedName(WarpFunctionsMap, SourceCalleeName.str()));
    } else if (SourceCalleeName.endswith("_sync")) {
      reportNoMaskWarning();
      RewriteArgList.emplace_back(getMigratedArg(1));
      RewriteArgList.emplace_back(getMigratedArg(2));
      setTargetCalleeName(buildString(
          DpctGlobalInfo::getItemName(), ".get_sub_group().",
          MapNames::findReplacedName(WarpFunctionsMap, SourceCalleeName.str())));
    } else {
      RewriteArgList.emplace_back(getMigratedArg(0));
      RewriteArgList.emplace_back(getMigratedArg(1));
      setTargetCalleeName(buildString(
          DpctGlobalInfo::getItemName(), ".get_sub_group().",
          MapNames::findReplacedName(WarpFunctionsMap, SourceCalleeName.str())));
    }
  }
  return buildRewriteString();
}

Optional<std::string> ReorderFunctionRewriter::rewrite() {
  for (auto ArgIdx : RewriterArgsIdx) {
    if (ArgIdx >= Call->getNumArgs())
      continue;
    if ((SourceCalleeName == "cudaBindTexture" ||
         SourceCalleeName == "cudaBindTexture2D") &&
        Call->getArg(1)->getType()->isPointerType() &&
        (ArgIdx == 1 || ArgIdx == 3)) {
      std::ostringstream OS;
      printDerefOp(OS, Call->getArg(ArgIdx));
      appendRewriteArg(OS.str());
    } else {
      appendRewriteArg(getMigratedArg(ArgIdx));
    }
  }
  return buildRewriteString();
}

void TemplatedCallExprRewriter::buildTemplateArgsList() {
  auto Callee = Call->getCallee()->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
    return buildTemplateArgsList(DRE->template_arguments());
  } else if (auto ULE = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    return buildTemplateArgsList(ULE->template_arguments());
  }
}

void TemplatedCallExprRewriter::buildTemplateArgsList(
    const ArrayRef<TemplateArgumentLoc> &Args) {
  ExprAnalysis EA;
  for (auto &Arg : Args) {
    EA.analyze(Arg);
    TemplateArgs.push_back(EA.getReplacedString());
  }
}

Optional<std::string> TemplatedCallExprRewriter::rewrite() {
  buildTemplateArgsList();
  RewriteArgList = getMigratedArgs();
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << TargetCalleeName;
  if (!TemplateArgs.empty()) {
    OS << "<";
    for (size_t i = 0; i < TemplateArgs.size(); ++i) {
      if (i)
        OS << ", ";
      OS << TemplateArgs[i];
    }
    OS << ">";
  }
  OS << "(";
  for (size_t i = 0; i < RewriteArgList.size(); ++i) {
    if (i)
      OS << ", ";
    OS << RewriteArgList[i];
  }
  OS << ")";
  return OS.str();
}

DerefExpr DerefExpr::create(const Expr *E) {
  DerefExpr D;
  E = E->IgnoreImplicitAsWritten();
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      E = UO->getSubExpr()->IgnoreImplicitAsWritten();
      D.AddrOfRemoved = true;
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      E = COCE->getArg(0)->IgnoreImplicitAsWritten();
      D.AddrOfRemoved = true;
    }
  }

  D.E = E;
  D.NeedParens = needExtraParens(E);
  return D;
}

std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> DerefExpr {
    return DerefExpr::create(C->getArg(Idx));
  };
}

std::function<const Expr *(const CallExpr *)> makeCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> const Expr * { return C->getArg(Idx); };
}

template <class Printer, class... Ts> class PrinterCreator {
  std::tuple<Ts...> Creators;

  template <class T> T create(T Val, const CallExpr *) { return Val; }
  StringRef create(const std::string &Val, const CallExpr *) { return Val; }
  template <class T>
  T create(std::function<T(const CallExpr *)> &Func, const CallExpr *C) {
    return Func(C);
  }
  template <size_t... Idx>
  Printer createPrinter(const CallExpr *C, std::index_sequence<Idx...>) {
    return Printer(create(std::get<Idx>(Creators), C)...);
  }

public:
  PrinterCreator(Ts... Args) : Creators(Args...) {}
  Printer operator()(const CallExpr *C) {
    return createPrinter(C, std::index_sequence_for<Ts...>());
  }
};

template <class BaseT, class... CallArgsT>
using MemberCallPrinterCreator =
    PrinterCreator<MemberCallPrinter<BaseT, CallArgsT...>,
                   std::function<BaseT(const CallExpr *)>, bool, std::string,
                   std::function<CallArgsT(const CallExpr *)>...>;

template <class BaseT, class... CallArgsT>
std::function<MemberCallPrinter<BaseT, CallArgsT...>(const CallExpr *)>
makeMemberCallCreator(std::function<BaseT(const CallExpr *)> BaseFunc,
                      bool IsArrow, std::string Member,
                      std::function<CallArgsT(const CallExpr *)>... Args) {
  return MemberCallPrinterCreator<BaseT, CallArgsT...>(
      std::forward<std::function<BaseT(const CallExpr *)>>(BaseFunc), IsArrow,
      Member,
      std::forward<std::function<CallArgsT(const CallExpr *)>>(Args)...);
}

/// Create AssignExprRewriterFactory with given argumens.
/// \p SourceName the source callee name of original call expr.
/// \p LValueCreator use to get lhs from original call expr.
/// \p RValueCreator use to get rhs from original call expr.
template <class LValue, class RValue>
std::shared_ptr<CallExprRewriterFactoryBase>
creatAssignExprRewriterFactory(
    const std::string &SourceName,
    std::function<LValue(const CallExpr *)> &&LValueCreator,
    std::function<RValue(const CallExpr *)> &&RValueCreator) {
  return std::make_shared<CallExprRewriterFactory<AssignExprRewriter<LValue, RValue>,
                                 std::function<LValue(const CallExpr *)>,
                                 std::function<RValue(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<LValue(const CallExpr *)>>(LValueCreator),
      std::forward<std::function<RValue(const CallExpr *)>>(RValueCreator));
}

/// Create MemberCallExprRewriterFactory with given argumens.
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
/// \p StartIdx is represent the first availbe argument's index.
/// For cudaBindTexture and cudaBindTexture2D, it is 1.
/// For  cudaBindTextureToArray, it is 0.
/// The first predicate will check the \p StartIdx 'th argument whether is pointer.
/// If it is true, the call expr will be migrated to member call expr.
/// e.g.: cudaBindTexture(0, &tex, data, &desc, size) -> tex.attach(data, size, desc)
/// with template arguments: <1, 2>.
/// Else will check the second predicate:
/// If \p Start + 2 'th argument's type whether is cudaChannelFormatDesc.
/// If it is true, e.g.: cudaBindTexture(0, tex, data, desc, size) ->
/// tex.attach(data, size, desc).
/// Else, e.g.: cudaBindTexture(0, tex, data, size) ->tex.attach(data, size, desc).
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
              Source, makeCallArgCreator(StartIdx + 0), false, "attach",
              makeCallArgCreator(StartIdx + 1),
              makeCallArgCreator(StartIdx + Idx + 1)...,
              makeCallArgCreator(StartIdx + 2)),
          createMemberCallExprRewriterFactory(
              Source, makeCallArgCreator(StartIdx + 0), false, "attach",
              makeCallArgCreator(StartIdx + 1),
              makeCallArgCreator(StartIdx + Idx)...)));
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
/// Predicate: check the first arg if is pointer and set texture info with corresponding
/// data.
/// Migrate the call expr to an assign expr if Pred result is true;
/// e.g.: tex1D(&u, tex, 1.0f) -> u = tex.read(1.0f)
/// Migrate the call expr to an assign expr if Pred result is false;
/// e.g.: tex1D(tex, 1.0f) -> tex.read(1.0f)
/// The template arguments is the member call arguments' index in original call expr.
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
      creatAssignExprRewriterFactory(
          Source, makeDerefExprCreator(0),
          makeMemberCallCreator(makeCallArgCreator(1),
                                              false, "read",
                                              makeCallArgCreator(Idx + 1)...)),
      createMemberCallExprRewriterFactory(Source, makeCallArgCreator(0), false,
                                          "read", makeCallArgCreator(Idx)...));
}

#define ASSIGNABLE_FACTORY(x) createAssignableFactory(x 0),
#define DEREF(x) makeDerefExprCreator(x)
#define ARG(x) makeCallArgCreator(x)
#define MEMBER_CALL(...) makeMemberCallCreator(__VA_ARGS__)
#define POINTER_CHECKER(x) makePointerChecker(x)
#define CONDITIONAL_FACTORY_ENTRY(Pred, First, Second)                         \
  createConditionalFactory(Pred, First Second 0),
#define ASSIGN_FACTORY_ENTRY(FuncName, L, R)                                   \
  {FuncName, creatAssignExprRewriterFactory(FuncName, L, R)},
#define MEMBER_CALL_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMemberCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define DELETER_FACTORY_ENTRY(FuncName, Arg)                                   \
  {FuncName, createDeleterCallExprRewriterFactory(FuncName, Arg)},

///***************************************************************
/// Examples:
/// cudaBindTexture(0, &tex21, d_data21, &desc21, 32 * sizeof(uint2))
/// -> tex21.attach(d_data21, 32 * sizeof(uint32)
/// using: MEMBER_CALL_FACTORY_ENTRY("cudaBindTexture", DEREF(1), true/* member
/// operator default is arrow*/, "attach", ARG(2), ARG(4), DEREF(3))
///
/// tex2DLayered(&data, tex, 2.0f, 2.0f, 10) -> data = tex.read(10, 2.0f, 2.0f)
/// using: ASSIGN_FACTORY_ENTRY("tex2DLayered", DEREF(0), MEMBER_ALL(ARG(1),
/// false/* member operator default is dot */, "read", ARG(4), ARG(2), ARG(3))
///
/// Macro ASSIGNABLE_FACTORY(x) is used for migration of call expr that has
/// valid return value.
///
/// Macro CONDITIONAL_FACTORY_ENTRY(pred, first, second) is used for conditonal
/// migration. \p pred is expr that can convert to std::function<bool(const
/// CallExpr *)>. \p first and \p second is two candidates factory. If \p pred
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
#define REORDER_FUNC_FACTORY_ENTRY(FuncName, RewriterName, ...)                \
  REWRITER_FACTORY_ENTRY(FuncName, ReorderFunctionRewriterFactory,             \
                         RewriterName, std::vector<unsigned>{__VA_ARGS__})
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportFunctionRewriterFactory, MsgID)
#define TEMPLATED_CALL_FACTORY_ENTRY(FuncName, RewriterName)                     \
  REWRITER_FACTORY_ENTRY(FuncName, TemplatedCallExprRewriterFactory, RewriterName)

const std::unordered_map<std::string,
	std::shared_ptr<CallExprRewriterFactoryBase>>
	CallExprRewriterFactoryBase::RewriterMap = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
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
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED

#define ENTRY_WARP(SOURCEAPINAME, TARGETAPINAME)                               \
  WARP_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesWarp.inc"
#undef ENTRY_WARP

#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_TEXTURE(SOURCEAPINAME, TEXTYPE, ...)                            \
  TEX_FUNCTION_FACTORY_ENTRY(SOURCEAPINAME, TEXTYPE, __VA_ARGS__)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#define ENTRY_BIND(SOURCEAPINAME, ...)            \
  BIND_TEXTURE_FACTORY_ENTRY(SOURCEAPINAME, __VA_ARGS__)
#define ENTRY_REORDER(SOURCEAPINAME, TARGETAPINAME, ...)            \
  REORDER_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME, __VA_ARGS__)
#define ENTRY_TEMPLATED(SOURCEAPINAME, TARGETAPINAME)                            \
  TEMPLATED_CALL_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesTexture.inc"
#undef ENTRY_RENAMED
#undef ENTRY_TEXTURE
#undef ENTRY_UNSUPPORTED
#undef ENTRY_TEMPLATED
#undef UNSUPPORTED_FACTORY_ENTRY

#define ENTRY_HOST(from, to, policy)
#define ENTRY_DEVICE(SOURCEAPINAME, TARGETAPINAME, EXTR) FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_BOTH(SOURCEAPINAME, TARGETAPINAME, EXTR)  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesMapThrust.inc"
#undef ENTRY_HOST
#undef ENTRY_DEVICE
#undef ENTRY_BOTH

};


const std::vector<std::string> MathFuncNameRewriter::SingleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
};

const std::vector<std::string> MathFuncNameRewriter::DoubleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
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
