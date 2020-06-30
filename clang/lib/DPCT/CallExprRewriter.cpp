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

Optional<std::string> FuncNameRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  return buildRewriteString();
}

Optional<std::string> FuncNameRewriter::buildRewriteString() {
  auto S = FuncCallExprRewriter::buildRewriteString();
  if (S.hasValue() && isAssigned(Call))
    return "(" + S.getValue() + ", 0)";
  return S;
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

Optional<std::string> ReorderFunctionIsAssignedRewriter::rewrite() {
  auto S = ReorderFunctionRewriter::rewrite();
  if (S.hasValue() && isAssigned(Call))
    return "(" + S.getValue() + ", 0)";
  return S;
}

void TexFunctionRewriter::setTextureInfo(int TexType) {
  const Expr *Obj = nullptr;
  std::string DataTy;
  int Idx = 0;
  auto &Global = DpctGlobalInfo::getInstance();
  if (Call->getArg(0)->getType()->isPointerType()) {
    DataTy = Global.getUnqualifiedTypeName(
        Call->getArg(0)->getType()->getPointeeType());
    Obj = Call->getArg(1);
    Idx = 1;
  } else {
    DataTy =
        Global.getUnqualifiedTypeName(Call->getType().getUnqualifiedType());
    Obj = Call->getArg(0);
    Idx = 0;
  }

  if (auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(Call)) {
    if (auto ObjInfo =
            DeviceFunctionDecl::LinkRedecls(FD)
                ->addCallee(Call)
                ->addTextureObjectArg(
                    Idx, dyn_cast<DeclRefExpr>(Obj->IgnoreImpCasts()))) {
      ObjInfo->setType(std::move(DataTy), TexType);
    }
  }
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

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterTy, ...)                      \
  {FuncName, std::make_shared<RewriterTy>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define FUNC_NAME_ISASSIGNED_FACTORY_ENTRY(FuncName, RewriterName)             \
  REWRITER_FACTORY_ENTRY(FuncName, FuncNameRewriterFactory, RewriterName)
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
#define REORDER_FUNC_ISASSIGNED_FACTORY_ENTRY(FuncName, RewriterName, ...)     \
  REWRITER_FACTORY_ENTRY(FuncName, ReorderFunctionIsAssignedRewriterFactory,   \
                         RewriterName, std::vector<unsigned>{__VA_ARGS__})
#define TEX_FUNCTION_FACTORY_ENTRY(FuncName, RewriterName, TexType)                     \
  REWRITER_FACTORY_ENTRY(FuncName, TexFunctionRewriterFactory, RewriterName, TexType)
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
#define ENTRY_RENAMED_ISASSIGNED(SOURCEAPINAME, TARGETAPINAME)                 \
  FUNC_NAME_ISASSIGNED_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_TEXTURE(SOURCEAPINAME, TARGETAPINAME, TEXTYPE)                            \
  TEX_FUNCTION_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME, TEXTYPE)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#define ENTRY_REORDER_ISASSIGNED(SOURCEAPINAME, TARGETAPINAME, ...)            \
  REORDER_FUNC_ISASSIGNED_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME, __VA_ARGS__)
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
