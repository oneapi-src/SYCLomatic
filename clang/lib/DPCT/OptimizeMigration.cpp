//===---OptimizeMigration.cpp -----------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "OptimizeMigration.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "Statics.h"
#include "MapNames.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void ForLoopUnrollRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  if (DpctGlobalInfo::isOptimizeMigration()) {
    MF.addMatcher(forStmt().bind("for_loop"), this);
  }
}


// Automatically add "#pragma unroll" to for loops that meet the following conditions:
// 1. The loop is inside a device function.
// 2. The loop does not have any other loop hint attributes.
// 3. No local variable declarations inside the loop, except:
//    a) Variables declared in for loop init-statement.
//    b) Variables that will be in local memory after migration.
//    c) Variables that will be removed after migration.
void ForLoopUnrollRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Context = DpctGlobalInfo::getContext();

  if (auto ForLoop = getNodeAsType<ForStmt>(Result, "for_loop")) {
    auto ForLoopLoc = ForLoop->getBeginLoc();
    if(ForLoopLoc.isMacroID()) {
      return;
    }
    auto FD = getImmediateOuterFuncDecl(ForLoop);
    if (!FD ||
        (!FD->hasAttr<CUDAGlobalAttr>() && !FD->hasAttr<CUDADeviceAttr>())) {
      return;
    }

    if (auto Parent = Context.getParents(*ForLoop)[0].get<AttributedStmt>()) {
      auto Attrs = Parent->getAttrs();
      for (auto Attr : Attrs) {
        if (isa<LoopHintAttr>(Attr)) {
          return;
        }
      }
    }
    if (auto C = dyn_cast<CompoundStmt>(ForLoop->getBody())) {
      auto VarDeclMatcher = findAll(declStmt().bind("ds"));
      auto MatchResult = ast_matchers::match(VarDeclMatcher, *C, Context);
      for (auto &SubResult : MatchResult) {
        const DeclStmt *DS = SubResult.getNodeAs<DeclStmt>("ds");
        if (!DS || Context.getParents(*DS)[0].get<ForStmt>()) {
          continue;
        }
        for (auto D : DS->decls()) {
          if (auto VD = dyn_cast<VarDecl>(D)) {
            if (!VD->hasAttr<CUDASharedAttr>() && VD->isLocalVarDecl() &&
                !isCubVar(VD)) {
              return;
            }
          }
        }
      }
    }
    std::string Repl;
    const char *RawData = SM.getCharacterData(ForLoopLoc.getLocWithOffset(-1));
    while(RawData && (*RawData == ' ')) {
      RawData--;
    }
    if(RawData && ((*RawData != '\r') && (*RawData != '\n'))) {
      Repl = getNL();
    }
    auto IndentStr = getIndent(ForLoopLoc, SM).str();
    Repl = Repl + "#pragma unroll" + getNL() + IndentStr;
    auto Begin = SM.getSpellingLoc(ForLoop->getBeginLoc());
    emplaceTransformation(new InsertText(Begin, Repl));
  }
}

void DeviceConstantVarOptimizeRule::registerMatcher(
    ast_matchers::MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(hasAttr(attr::CUDAConstant),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcher.bind("ConstantVar"), this);
  if (DpctGlobalInfo::isOptimizeMigration()) {
    auto RuntimeSymnolAPIName = [&]() {
      return hasAnyName("cudaGetSymbolAddress", "cudaGetSymbolSize",
                        "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol");
    };

    MF.addMatcher(callExpr(callee(functionDecl(RuntimeSymnolAPIName())))
                      .bind("RuntimeSymnolAPICall"),
                  this);
  }
}

void DeviceConstantVarOptimizeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "ConstantVar")) {
    if (isCubVar(MemVar)) {
      return;
    }
    MemVarInfo::buildMemVarInfo(MemVar);
    return;
  }
  if (auto CE = getNodeAsType<CallExpr>(Result, "RuntimeSymnolAPICall")) {
    if (auto DC = CE->getDirectCallee()) {
      std::string FuncName = getFunctionName(DC);
      int SymbolArgIndex = 1;
      if (FuncName == "cudaMemcpyToSymbol") {
        SymbolArgIndex = 0;
      }
      auto SymbolExpr = CE->getArg(SymbolArgIndex);
      auto DREResults = findDREInScope(SymbolExpr);
      for (auto &Result : DREResults) {
        const DeclRefExpr *MatchedDRE =
            Result.getNodeAs<DeclRefExpr>("VarReference");
        if (!MatchedDRE)
          continue;
        if (auto VD = dyn_cast_or_null<VarDecl>(MatchedDRE->getDecl())) {
          if (VD->hasAttr<CUDAConstantAttr>()) {
            auto &Set = DpctGlobalInfo::getVarUsedByRuntimeSymbolAPISet();
            auto LocInfo = DpctGlobalInfo::getLocInfo(VD->getBeginLoc());
            std::string Key = LocInfo.first + std::to_string(LocInfo.second) +
                              VD->getNameAsString();
            Set.insert(Key);
          }
        }
      }
    }
  }
}

} // namespace dpct
} // namespace clang