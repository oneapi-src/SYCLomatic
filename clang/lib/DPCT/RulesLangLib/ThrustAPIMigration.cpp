//===--------------- ThrustAPIMigration.cpp--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RulesLangLib/ThrustAPIMigration.h"
#include "AnalysisInfo.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RulesLangLib/MapNamesLangLib.h"
#include "TextModification.h"

extern int ThrustVersion;
namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void ThrustAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  // API register
  auto functionName = [&]() { return hasAnyName("on"); };

  // THRUST_200302___CUDA_ARCH_LIST___NS is newly imported inline
  // namespace by thrust library in CUDA header file 12.4.
  auto thrustFuncNameCuda124 = [&]() {
    return hasAnyName("THRUST_200302___CUDA_ARCH_LIST___NS",
                      "THRUST_200302___CUDA_ARCH_LIST___NS::detail",
                      "THRUST_200302___CUDA_ARCH_LIST___NS::system");
  };

  // THRUST_200400___CUDA_ARCH_LIST___NS is newly imported inline
  // namespace by thrust library in CUDA header file 12.5.
  auto thrustFuncNameCuda125 = [&]() {
    return hasAnyName("THRUST_200400___CUDA_ARCH_LIST___NS",
                      "THRUST_200400___CUDA_ARCH_LIST___NS::detail",
                      "THRUST_200400___CUDA_ARCH_LIST___NS::system");
  };

  // THRUST_200500___CUDA_ARCH_LIST___NS is newly imported inline
  // namespace by thrust library in CUDA header file 12.6.
  auto thrustFuncNameCuda126 = [&]() {
    return hasAnyName("THRUST_200500___CUDA_ARCH_LIST___NS",
                      "THRUST_200500___CUDA_ARCH_LIST___NS::detail",
                      "THRUST_200500___CUDA_ARCH_LIST___NS::system");
  };

  auto thrustFuncNameCudaCommon = [&]() {
    return hasAnyName("thrust", "thrust::detail", "thrust::system", "__4");
  };

  int ThrustMajorVersion = ThrustVersion / 100000;
  int ThrustMinorVersion = ThrustVersion / 100 % 1000;

  if (ThrustMajorVersion == 2 && ThrustMinorVersion == 3) {
    // For CUDA-12.4
    MF.addMatcher(
        callExpr(
            anyOf(callee(functionDecl(anyOf(
                      hasDeclContext(namespaceDecl(thrustFuncNameCuda124())),
                      hasDeclContext(namespaceDecl(thrustFuncNameCudaCommon())),
                      functionName()))),
                  callee(unresolvedLookupExpr(
                      hasAnyDeclaration(namedDecl(hasDeclContext(namespaceDecl(
                          anyOf(thrustFuncNameCuda124(),
                                thrustFuncNameCudaCommon())))))))))
            .bind("thrustFuncCall"),
        this);

  } else if (ThrustMajorVersion == 2 && ThrustMinorVersion == 4) {
    // For CUDA-12.5
    MF.addMatcher(
        callExpr(
            anyOf(callee(functionDecl(anyOf(
                      hasDeclContext(namespaceDecl(thrustFuncNameCuda125())),
                      hasDeclContext(namespaceDecl(thrustFuncNameCudaCommon())),
                      functionName()))),
                  callee(unresolvedLookupExpr(
                      hasAnyDeclaration(namedDecl(hasDeclContext(namespaceDecl(
                          anyOf(thrustFuncNameCuda125(),
                                thrustFuncNameCudaCommon())))))))))
            .bind("thrustFuncCall"),
        this);
  } else if (ThrustMajorVersion == 2 && ThrustMinorVersion == 5) {
    // For CUDA-12.6
    MF.addMatcher(
        callExpr(
            anyOf(callee(functionDecl(anyOf(
                      hasDeclContext(namespaceDecl(thrustFuncNameCuda126())),
                      hasDeclContext(namespaceDecl(thrustFuncNameCudaCommon())),
                      functionName()))),
                  callee(unresolvedLookupExpr(
                      hasAnyDeclaration(namedDecl(hasDeclContext(namespaceDecl(
                          anyOf(thrustFuncNameCuda126(),
                                thrustFuncNameCudaCommon())))))))))
            .bind("thrustFuncCall"),
        this);
  } else {
    // For CUDA SDK versions before CUDA-12.4
    MF.addMatcher(
        callExpr(
            anyOf(callee(functionDecl(anyOf(
                      hasDeclContext(namespaceDecl(thrustFuncNameCudaCommon())),

                      functionName()))),
                  callee(unresolvedLookupExpr(
                      hasAnyDeclaration(namedDecl(hasDeclContext(
                          namespaceDecl(thrustFuncNameCudaCommon()))))))))
            .bind("thrustFuncCall"),
        this);
  }

  // THRUST_STATIC_ASSERT macro register
  MF.addMatcher(staticAssertDecl(isExpandedFromMacro("THRUST_STATIC_ASSERT"))
                    .bind("THRUST_STATIC_ASSERT"),
                this);
  MF.addMatcher(typedefDecl(isExpandedFromMacro("THRUST_STATIC_ASSERT"))
                    .bind("THRUST_STATIC_ASSERT"),
                this);
}

void ThrustAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "thrustFuncCall")) {
    if (const UnresolvedLookupExpr *ULE =
            dyn_cast_or_null<UnresolvedLookupExpr>(CE->getCallee()))
      thrustFuncMigration(Result, CE, ULE);
    else
      thrustFuncMigration(Result, CE);
  } else if (const Decl *D =
                 getNodeAsType<Decl>(Result, "THRUST_STATIC_ASSERT")) {
    const SourceManager &SM = DpctGlobalInfo::getSourceManager();
    const SourceLocation BeginLoc = SM.getExpansionLoc(D->getBeginLoc());
    emplaceTransformation(new ReplaceText(
        BeginLoc, std::string("THRUST_STATIC_ASSERT").size(), "static_assert"));
  }
}

void ThrustAPIRule::thrustFuncMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *CE,
                                        const UnresolvedLookupExpr *ULExpr) {
  // handle the regular call expr
  std::string ThrustFuncName;
  if (ULExpr) {
    std::string Namespace;
    if (auto NNS = ULExpr->getQualifier()) {
      if (auto NS = NNS->getAsNamespace()) {
        Namespace = NS->getNameAsString();
      }
    }
    if (!Namespace.empty() && Namespace == "thrust")
      ThrustFuncName = ULExpr->getName().getAsString();
  } else {
    ThrustFuncName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  // Process API: "thrust::cuda::par(thrust_allocator).on(stream)"
  const CXXMemberCallExpr *CMCE = dyn_cast_or_null<CXXMemberCallExpr>(CE);
  if (CMCE) {
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.PrintCanonicalTypes = true;
    auto BaseType = CMCE->getObjectType().getUnqualifiedType().getAsString(PP);
    StringRef BaseTypeRef(BaseType);
    if (BaseTypeRef.starts_with("thrust::cuda_cub::execute_on_stream_base<") &&
        ThrustFuncName == "on") {
      std::string ReplStr = "oneapi::dpl::execution::make_device_policy(" +
                            getDrefName(CMCE->getArg(0)) + ")";
      emplaceTransformation(new ReplaceStmt(CMCE, std::move(ReplStr)));
      return;
    }
  }

  std::string ThrustFuncNameWithNamespace = "thrust::" + ThrustFuncName;

  auto ReplInfo =
      MapNamesLangLib::ThrustFuncNamesMap.find(ThrustFuncNameWithNamespace);
  // For the API migration defined in APINamesThrust.inc
  if (ReplInfo == MapNamesLangLib::ThrustFuncNamesMap.end()) {
    dpct::ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  auto QT = CE->getArg(0)->getType();
  LangOptions LO;
  std::string ArgT = QT.getAsString(PrintingPolicy(LO));

  // For the API migration defined in APINamesMapThrust.inc
  auto HelperFeatureIter =
      MapNamesLangLib::ThrustFuncNamesHelperFeaturesMap.find(
          ThrustFuncNameWithNamespace);
  if (HelperFeatureIter !=
      MapNamesLangLib::ThrustFuncNamesHelperFeaturesMap.end()) {
    requestFeature(HelperFeatureIter->second);
  }

  auto NewName = ReplInfo->second.ReplName;

  bool hasExecutionPolicy =
      ArgT.find("execution_policy_base") != std::string::npos;
  bool PolicyProcessed = false;

  // To migrate "thrust::cuda::par.on" that appears in CE' first arg to
  // "oneapi::dpl::execution::make_device_policy".
  const CallExpr *Call = nullptr;
  if (hasExecutionPolicy) {
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(CE->getArg(0))) {
      if (const auto *MT =
              dyn_cast<MaterializeTemporaryExpr>(ICE->getSubExpr())) {
        if (auto SubICE = dyn_cast<ImplicitCastExpr>(MT->getSubExpr())) {
          Call = dyn_cast<CXXMemberCallExpr>(SubICE->getSubExpr());
        }
      }
    } else if (const auto *SubCE = dyn_cast<CallExpr>(CE->getArg(0))) {
      Call = SubCE;
    } else {
      Call = dyn_cast<CXXMemberCallExpr>(CE->getArg(0));
    }
  }

  if (Call) {
    std::ostringstream OS;
    if (const auto *ME = dyn_cast<MemberExpr>(Call->getCallee())) {
      auto BaseName =
          DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
      if (BaseName == "thrust::cuda_cub::par_t") {
        OS << "oneapi::dpl::execution::make_device_policy(";
        printDerefOp(OS, Call->getArg(0));
        OS << ")";
        emplaceTransformation(new ReplaceStmt(Call, OS.str()));
        PolicyProcessed = true;
      }
    }
  }

  // All the thrust APIs (such as thrust::copy, thrust::fill,
  // thrust::count, thrust::equal) called in device function , should be
  // migrated to oneapi::dpl APIs without a policy on the SYCL side
  if (auto FD = DpctGlobalInfo::getParentFunction(CE)) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      if (hasExecutionPolicy) {
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
      }
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                      ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                      : ULExpr->getEndLoc();
    emplaceTransformation(
        new ReplaceToken(BeginLoc, EndLoc, std::move(NewName)));
    if (PolicyProcessed)
      return;
    if (hasExecutionPolicy) {
      emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
    }
  } else {
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
    if (hasExecutionPolicy)
      return;
  }

  if (CE->getNumArgs() <= 0)
    return;
  auto ExtraParam = ReplInfo->second.ExtraParam;
  if (!ExtraParam.empty()) {
    // This is a temporary fix until, the Intel(R) oneAPI DPC++ Compiler and
    // Intel(R) oneAPI DPC++ Library support creating a SYCL execution policy
    // without creating a unique one for every use
    if (ExtraParam == "oneapi::dpl::execution::sycl") {
      // If no policy is specified and raw pointers are used
      // a host execution policy must be specified to match the thrust
      // behavior
      if (CE->getArg(0)->getType()->isPointerType()) {
        ExtraParam = "oneapi::dpl::execution::seq";
      } else {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        ExtraParam = makeDevicePolicy(CE);
      }
    }
    emplaceTransformation(
        new InsertBeforeStmt(CE->getArg(0), ExtraParam + ", "));
  }
}

void ThrustTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  // TYPE register
  auto ThrustTypeHasNames = [&]() {
    return hasAnyName(
        "thrust::greater_equal", "thrust::less_equal", "thrust::logical_and",
        "thrust::bit_and", "thrust::bit_or", "thrust::minimum",
        "thrust::bit_xor", "thrust::modulus", "thrust::greater",
        "thrust::identity", "thrust::null_type", "thrust::detail::enable_if",
        "thrust::detail::true_type", "thrust::detail::false_type",
        "thrust::detail::integral_constant", "thrust::detail::is_same",
        "thrust::system::detail::bad_alloc", "thrust::iterator_traits",
        "thrust::detail::vector_base", "thrust::optional", "thrust::nullopt",
        "thrust::system::system_error", "thrust::system::error_code",
        "enum thrust::system::errc::errc_t", "thrust::system::error_condition",
        "thrust::device_system_tag", "thrust::pointer",
        "thrust::device_allocator", "thrust::reverse_iterator",
        "thrust::system::errc::errc_t");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(ThrustTypeHasNames())))))
          .bind("thrustTypeLoc"),
      this);

  // CTOR register
  auto hasAnyThrustRecord = []() {
    return cxxRecordDecl(hasName("complex"),
                         hasDeclContext(namespaceDecl(hasName("thrust"))));
  };

  MF.addMatcher(
      cxxConstructExpr(hasType(hasAnyThrustRecord())).bind("thrustCtorExpr"),
      this);

  auto hasFunctionalActor = []() {
    return hasType(qualType(hasDeclaration(
        cxxRecordDecl(hasName("thrust::detail::functional::actor")))));
  };

  MF.addMatcher(
      cxxConstructExpr(anyOf(hasFunctionalActor(),
                             hasType(qualType(hasDeclaration(
                                 typedefNameDecl(hasFunctionalActor()))))))
          .bind("thrustCtorPlaceHolder"),
      this);

  // Var register
  auto hasExprName = [&]() {
    return hasAnyName("seq", "host", "device", "thrust::nullopt");
  };

  MF.addMatcher(declRefExpr(to(varDecl(hasExprName()).bind("varDecl")))
                    .bind("declRefExpr"),
                this);
}

void ThrustTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getAssistNodeAsType<TypeLoc>(Result, "thrustTypeLoc")) {
    ExprAnalysis EA;
    auto DNTL = DpctGlobalInfo::findAncestor<DependentNameTypeLoc>(TL);
    auto NNSL = DpctGlobalInfo::findAncestor<NestedNameSpecifierLoc>(TL);
    if (NNSL) {
      EA.analyze(*TL, *NNSL);
    } else if (DNTL) {
      EA.analyze(*TL, *DNTL);
    } else {
      EA.analyze(*TL);
    }
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (const CXXConstructExpr *CE =
                 getNodeAsType<CXXConstructExpr>(Result, "thrustCtorExpr")) {
    thrustCtorMigration(CE);
  } else if (const CXXConstructExpr *CE = getNodeAsType<CXXConstructExpr>(
                 Result, "thrustCtorPlaceHolder")) {
    // handle constructor expressions with placeholders (_1, _2, etc)
    replacePlaceHolderExpr(CE);
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr")) {
    auto VD = getAssistNodeAsType<VarDecl>(Result, "varDecl");
    if (DRE->hasQualifier()) {
      auto ND =
          DRE->getQualifierLoc().getNestedNameSpecifier()->getAsNamespace();

      if (!ND || ND->getName() != "thrust")
        return;

      if (!VD)
        return;

      const std::string ThrustVarName =
          ND->getNameAsString() + "::" + VD->getName().str();

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, ThrustVarName);
      insertHeaderForTypeRule(ThrustVarName, DRE->getBeginLoc());
      requestHelperFeatureForTypeNames(ThrustVarName);
      if (Replacement == "oneapi::dpl::execution::dpcpp_default")
        Replacement = makeDevicePolicy(DRE);

      if (!Replacement.empty()) {
        emplaceTransformation(new ReplaceToken(
            DRE->getBeginLoc(), DRE->getEndLoc(), std::move(Replacement)));
      }
    } else {
      ExprAnalysis EA(DRE);
      EA.analyze();
      const std::string TyName = EA.getReplacedString();

      std::string ReplStr =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);

      if (!ReplStr.empty()) {
        emplaceTransformation(new ReplaceStmt(DRE, "std::nullopt"));
      }
    }
  } else {
    return;
  }
}

void ThrustTypeRule::replacePlaceHolderExpr(const CXXConstructExpr *CE) {
  unsigned PlaceholderCount = 0;

  auto placeholderStr = [](unsigned Num) {
    return std::string("_") + std::to_string(Num);
  };

  // Walk the expression and replace all placeholder occurrences
  std::function<void(const Stmt *)> walk = [&](const Stmt *S) {
    if (auto DRE = dyn_cast<DeclRefExpr>(S)) {
      auto DREStr = getStmtSpelling(DRE);
      auto TypeStr = DRE->getType().getAsString();
      std::string PlaceHolderTypeStr =
          "const thrust::detail::functional::placeholder<";
      if (TypeStr.find(PlaceHolderTypeStr) == 0) {
        unsigned PlaceholderNum =
            (TypeStr[PlaceHolderTypeStr.length()] - '0') + 1;
        if (PlaceholderNum > PlaceholderCount)
          PlaceholderCount = PlaceholderNum;
        emplaceTransformation(
            new ReplaceStmt(DRE, placeholderStr(PlaceholderNum)));
      }
      return;
    }
    for (auto SI : S->children())
      walk(SI);
  };
  walk(CE);

  if (PlaceholderCount == 0)
    // No placeholders were found, so no replacement is necessary
    return;

  // Construct the lambda wrapper and insert around placeholder expression
  std::string LambdaPrefix = "[=](";
  for (unsigned i = 1; i <= PlaceholderCount; ++i) {
    if (i > 1)
      LambdaPrefix += ",";
    LambdaPrefix += "auto _" + std::to_string(i);
  }
  LambdaPrefix += "){return ";
  emplaceTransformation(new InsertBeforeStmt(CE, std::move(LambdaPrefix)));
  std::string LambdaPostfix = ";}";
  emplaceTransformation(new InsertAfterStmt(CE, std::move(LambdaPostfix)));
}

void ThrustTypeRule::thrustCtorMigration(const CXXConstructExpr *CE) {
  // handle constructor expressions for thrust::complex
  std::string ExprStr = getStmtSpelling(CE);
  if (ExprStr.substr(0, 8) != "thrust::") {
    return;
  }
  auto P = ExprStr.find('<');
  if (P != std::string::npos) {
    auto ReplInfo = MapNamesLangLib::ThrustFuncNamesMap.find(ExprStr);
    if (ReplInfo == MapNamesLangLib::ThrustFuncNamesMap.end()) {
      return;
    }
    std::string ReplName = ReplInfo->second.ReplName;
    if (ReplName == "std::complex") {
      DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Complex);
    }
    emplaceTransformation(
        new ReplaceText(CE->getBeginLoc(), P, std::move(ReplName)));
  }
}

} // namespace dpct
} // namespace clang
