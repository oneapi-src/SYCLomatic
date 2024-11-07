//===---GenCodePinHeader.h -------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_GEN_CODEPIN_HEADER_H
#define DPCT_GEN_CODEPIN_HEADER_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class GenCodePinHeaderRule : public NamedMigrationRule<GenCodePinHeaderRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void collectInfoForCodePinDumpFunction(QualType T);
  void collectMemberInfo(QualType T, VarInfoForCodePin &VI,
                         std::vector<QualType> &MembersType, bool IsBaseMember,
                         clang::PrintingPolicy &PrintPolicy);
  void processTemplateTypeForCodePin(
      const ClassTemplateSpecializationDecl *Spec, VarInfoForCodePin &VI,
      std::vector<QualType> &MembersType, bool IsBaseMember,
      clang::PrintingPolicy &PrintPolicy);
  void processCodePinTypeMemberOrBase(QualType MT, std::string Name,
                                      VarInfoForCodePin &VarInfo,
                                      std::vector<QualType> &MembersType,
                                      CodePinVarInfoType InfoType, bool IsBFS,
                                      bool IsBaseMember,
                                      clang::PrintingPolicy &PrintPolicy);
  void saveCodePinTypeDeps(
      std::string &Key, QualType &DepT,
      std::vector<std::pair<std::string, std::vector<std::string>>> &DepsVec,
      bool IsDumpFunc, clang::PrintingPolicy &PrintPolicy);
  std::string getCodePinTypeHashKey(QualType T);
};

} // namespace dpct
} // namespace clang

#endif // !DPCT_GEN_CODEPIN_HEADER_H