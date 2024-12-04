//===------------------ RandomAPIMigration.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RandomAPIMigration.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RulesMathLib/MapNamesRandom.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;


// Rule for RANDOM enums.
void RandomEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CURAND_STATUS.*"))))
          .bind("RANDOMStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CURAND_ORDERING.*"))))
          .bind("RANDOMOrderingConstants"),
      this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CURAND_RNG.*"))))
                    .bind("RandomTypeEnum"),
                this);
}

void RandomEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RANDOMStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RANDOMOrderingConstants")) {
    std::string EnumStr = DE->getNameInfo().getName().getAsString();
    auto Search = MapNamesRandom::RandomOrderingTypeMap.find(EnumStr);
    if (Search == MapNamesRandom::RandomOrderingTypeMap.end()) {
      report(DE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumStr);
      return;
    }
    emplaceTransformation(new ReplaceStmt(DE, Search->second));
  }
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RandomTypeEnum")) {
    std::string EnumStr = DE->getNameInfo().getName().getAsString();
    auto Search = MapNamesRandom::RandomEngineTypeMap.find(EnumStr);
    if (Search == MapNamesRandom::RandomEngineTypeMap.end()) {
      report(DE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumStr);
      return;
    }
    if (EnumStr == "CURAND_RNG_PSEUDO_XORWOW" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64") {
      report(DE->getBeginLoc(), Diagnostics::DIFFERENT_GENERATOR, false);
    } else if (EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32") {
      report(DE->getBeginLoc(), Diagnostics::DIFFERENT_BASIC_GENERATOR, false);
    }
    emplaceTransformation(new ReplaceStmt(DE, Search->second));
  }
}

// Rule for Random function calls. Currently only support host APIs.
void RandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "curandCreateGenerator", "curandSetPseudoRandomGeneratorSeed",
        "curandSetGeneratorOffset", "curandSetQuasiRandomGeneratorDimensions",
        "curandDestroyGenerator", "curandGenerate", "curandGenerateLongLong",
        "curandGenerateLogNormal", "curandGenerateLogNormalDouble",
        "curandGenerateNormal", "curandGenerateNormalDouble",
        "curandGeneratePoisson", "curandGenerateUniform",
        "curandGenerateUniformDouble", "curandSetStream",
        "curandCreateGeneratorHost", "curandSetGeneratorOrdering");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void RandomFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // Below code can distinguish this kind of function like macro
  //  #define CHECK_STATUS(x) x
  //  CHECK_STATUS(anAPICall());
  bool IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM.isMacroArgExpansion(CE->getEndLoc());

  if (FuncNameBegin.isMacroID() && IsMacroArg) {
    FuncNameBegin = SM.getImmediateSpellingLoc(FuncNameBegin);
    FuncNameBegin = SM.getExpansionLoc(FuncNameBegin);
  } else if (FuncNameBegin.isMacroID()) {
    FuncNameBegin = SM.getExpansionLoc(FuncNameBegin);
  }

  if (FuncCallEnd.isMacroID() && IsMacroArg) {
    FuncCallEnd = SM.getImmediateSpellingLoc(FuncCallEnd);
    FuncCallEnd = SM.getExpansionLoc(FuncCallEnd);
  } else if (FuncCallEnd.isMacroID()) {
    FuncCallEnd = SM.getExpansionLoc(FuncCallEnd);
  }

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);

  if (IsAssigned) {
    requestFeature(HelperFeatureEnum::device_ext);
    insertAroundStmt(CE, MapNames::getCheckErrorMacroName() + "(", ")");
  }

  if (FuncName == "curandCreateGenerator" ||
      FuncName == "curandCreateGeneratorHost") {
    const auto *const Arg0 = CE->getArg(0);
    requestFeature(HelperFeatureEnum::device_ext);
    std::string RHS =
        buildString(" = ", MapNames::getLibraryHelperNamespace(),
                    "rng::create_host_rng(", ExprAnalysis::ref(CE->getArg(1)));
    if (FuncName == "curandCreateGeneratorHost") {
      if (DpctGlobalInfo::useSYCLCompat())
        RHS = buildString(RHS, ", *", MapNames::getDpctNamespace(),
                          "cpu_device().default_queue()");
      else
        RHS = buildString(RHS, ", ", MapNames::getDpctNamespace(),
                          "cpu_device().default_queue()");
    }
    RHS = buildString(RHS, ")");
    if (Arg0->getStmtClass() == Stmt::UnaryOperatorClass) {
      const auto *const UO = cast<const UnaryOperator>(Arg0);
      auto SE = UO->getSubExpr();
      if (UO->getOpcode() == UO_AddrOf &&
          (SE->getStmtClass() == Stmt::DeclRefExprClass ||
           SE->getStmtClass() == Stmt::MemberExprClass)) {
        return emplaceTransformation(new ReplaceStmt(
            CE, false, buildString(ExprAnalysis::ref(SE), RHS)));
      }
    }
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString("*(", ExprAnalysis::ref(CE->getArg(0)), ")", RHS)));
  }
  if (FuncName == "curandDestroyGenerator") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false, buildString(ExprAnalysis::ref(CE->getArg(0)), ".reset()")));
  }
  if (FuncName == "curandSetPseudoRandomGeneratorSeed") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_seed(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_dimensions(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (MapNamesRandom::RandomGenerateFuncMap.find(FuncName) !=
      MapNamesRandom::RandomGenerateFuncMap.end()) {
    auto ArgStr = ExprAnalysis::ref(CE->getArg(1));
    for (unsigned i = 2; i < CE->getNumArgs(); ++i) {
      ArgStr += buildString(", ", ExprAnalysis::ref(CE->getArg(i)));
    }
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(
            ExprAnalysis::ref(CE->getArg(0)),
            "->" +
                MapNamesRandom::RandomGenerateFuncMap.find(FuncName)->second +
                "(",
            ArgStr, ")")));
  }
  if (FuncName == "curandSetGeneratorOffset") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->skip_ahead(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (FuncName == "curandSetStream") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_queue(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

// Rule for device Random function calls.
void DeviceRandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "curand_init", "curand", "curand4", "curand_normal", "curand_normal4",
        "curand_normal2", "curand_normal2_double", "curand_normal_double",
        "curand_log_normal", "curand_log_normal2", "curand_log_normal2_double",
        "curand_log_normal4", "curand_log_normal_double", "curand_uniform",
        "curand_uniform2_double", "curand_uniform4", "curand_uniform_double",
        "curand_poisson", "curand_poisson4", "skipahead", "skipahead_sequence",
        "skipahead_subsequence", "curand_uniform4_double", "curand_normal4_double",
        "curand_log_normal4_double");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);
}

void DeviceRandomFunctionCallRule::runRule(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE)
    return;
  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);
  auto FuncCallLength =
      SM.getCharacterData(FuncCallEnd) - SM.getCharacterData(FuncNameBegin);
  std::string IndentStr = getIndent(FuncNameBegin, SM).str();

  if (FuncName == "curand_init") {
    if (CE->getNumArgs() < 4) {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    std::string Arg0Type = DpctGlobalInfo::getTypeName(
        CE->getArg(0)->getType().getCanonicalType());
    std::string Arg1Type = DpctGlobalInfo::getTypeName(
        CE->getArg(1)->getType().getCanonicalType());
    std::string Arg2Type = DpctGlobalInfo::getTypeName(
        CE->getArg(2)->getType().getCanonicalType());
    std::string DRefArg3Type;

    if (Arg0Type == "unsigned long long" && Arg1Type == "unsigned long long" &&
        Arg2Type == "unsigned long long" &&
        CE->getArg(3)->getType().getCanonicalType()->isPointerType()) {
      DRefArg3Type = DpctGlobalInfo::getTypeName(
          CE->getArg(3)->getType().getCanonicalType()->getPointeeType());
      if (MapNamesRandom::DeviceRandomGeneratorTypeMap.find(DRefArg3Type) ==
          MapNamesRandom::DeviceRandomGeneratorTypeMap.end()) {
        report(FuncNameBegin, Diagnostics::NOT_SUPPORTED_PARAMETER, false,
               FuncName,
               "parameter " + getStmtSpelling(CE->getArg(3)) +
                   " is unsupported");
        return;
      }
    } else {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    auto IsLiteral = [=](const Expr *E) {
      if (dyn_cast<IntegerLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FloatingLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FixedPointLiteral>(E->IgnoreCasts())) {
        return true;
      }
      return false;
    };

    std::string GeneratorType =
        MapNamesRandom::DeviceRandomGeneratorTypeMap.find(DRefArg3Type)->second;
    std::string RNGSeed = ExprAnalysis::ref(CE->getArg(0));
    bool IsRNGSubseqLiteral = IsLiteral(CE->getArg(1));
    std::string RNGSubseq = ExprAnalysis::ref(CE->getArg(1));
    bool IsRNGOffsetLiteral = IsLiteral(CE->getArg(2));
    std::string RNGOffset = ExprAnalysis::ref(CE->getArg(2));
    std::string RNGStateName = getDrefName(CE->getArg(3));

    std::string FirstOffsetArg, SecondOffsetArg;
    if (IsRNGOffsetLiteral) {
      FirstOffsetArg = RNGOffset;
    } else {
      FirstOffsetArg = "static_cast<std::uint64_t>(" + RNGOffset + ")";
    }

    std::string ReplStr;
    if (DRefArg3Type == "curandStateXORWOW") {
      report(FuncNameBegin, Diagnostics::SUBSEQUENCE_IGNORED, false, RNGSubseq);
      ReplStr = RNGStateName + " = " + GeneratorType + "(" + RNGSeed + ", " +
                FirstOffsetArg + ")";
    } else {
      std::string Factor = "8";
      if (GeneratorType == MapNames::getLibraryHelperNamespace() +
                               "rng::device::rng_generator<oneapi::"
                               "mkl::rng::device::philox4x32x10<1>>" &&
          DRefArg3Type == "curandStatePhilox4_32_10") {
        Factor = "4";
      }

      if (needExtraParens(CE->getArg(1))) {
        RNGSubseq = "(" + RNGSubseq + ")";
      }
      if (IsRNGSubseqLiteral) {
        SecondOffsetArg = RNGSubseq + " * " + Factor;
      } else {
        SecondOffsetArg =
            "static_cast<std::uint64_t>(" + RNGSubseq + " * " + Factor + ")";
      }

      ReplStr = RNGStateName + " = " + GeneratorType + "(" + RNGSeed + ", {" +
                FirstOffsetArg + ", " + SecondOffsetArg + "})";
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncCallLength, std::move(ReplStr)));
  } else if (FuncName == "skipahead" || FuncName == "skipahead_sequence" ||
             FuncName == "skipahead_subsequence") {
    if (FuncName == "skipahead") {
      std::string Arg1Type = CE->getArg(1)->getType().getAsString();
      if (Arg1Type != "curandStateMRG32k3a_t *" &&
          Arg1Type != "curandStatePhilox4_32_10_t *" &&
          Arg1Type != "curandStateXORWOW_t *") {
        // Do not support Sobol32 state and Sobol64 state
        report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
        return;
      }
    }
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}


} // namespace dpct
} // namespace clang