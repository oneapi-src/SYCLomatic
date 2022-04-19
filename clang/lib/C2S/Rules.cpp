//===--- Rules.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//
#include "ASTTraversal.h"
#include "CallExprRewriter.h"
#include "Rules.h"
#include "Utility.h"
#include "Error.h"
#include "MapNames.h"
#include "llvm/Support/YAMLTraits.h"
std::vector<std::string> MetaRuleObject::RuleFiles;
std::vector<std::shared_ptr<MetaRuleObject>> MetaRules;

void registerMacroRule(MetaRuleObject &R) {
  auto It = MapNames::MacroRuleMap.find(R.In);
  if (It != MapNames::MacroRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second.Id = R.RuleId;
      It->second.Priority = R.Priority;
      It->second.In = R.In;
      It->second.Out = R.Out;
      It->second.HelperFeature =
        clang::c2s::HelperFeatureEnum::no_feature_helper;
      It->second.Includes = R.Includes;
    }
  } else {
    MapNames::MacroRuleMap.emplace(
        R.In,
        MacroMigrationRule(R.RuleId, R.Priority, R.In, R.Out,
                           clang::c2s::HelperFeatureEnum::no_feature_helper,
                           R.Includes));
  }
}

void registerAPIRule(MetaRuleObject &R) {
  // register rule
  clang::c2s::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
    return new clang::c2s::UserDefinedAPIRule(R.In);
  });
  // create and register rewriter
  // RewriterMap contains entries like {"FunctionName", RewriterFactory}
  // The priority of all the default rules are RulePriority::Fallback and
  // recorded in the RewriterFactory.
  // Search for existing rules of the same function name
  // if there is existing rule,
  //   compare the priority and decide whether to add the rule into the
  //   RewriterMap
  // if there is no existing rule,
  //   add the new rule to the RewriterMap
  auto It =
      clang::c2s::CallExprRewriterFactoryBase::RewriterMap->find(R.In);
  if (It == clang::c2s::CallExprRewriterFactoryBase::RewriterMap->end()) {
    clang::c2s::CallExprRewriterFactoryBase::RewriterMap->emplace(
        R.In, clang::c2s::createUserDefinedRewriterFactory(R.In, R));
  } else if (It->second->Priority > R.Priority) {
    (*clang::c2s::CallExprRewriterFactoryBase::RewriterMap)[R.In] =
        clang::c2s::createUserDefinedRewriterFactory(R.In, R);
  }
}

void registerHeaderRule(MetaRuleObject &R) {
  auto It = MapNames::HeaderRuleMap.find(R.In);
  if (It != MapNames::HeaderRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second = R;
    }
  }
  else {
    MapNames::HeaderRuleMap.emplace(R.In, R);
  }
}

void registerTypeRule(MetaRuleObject &R) {
  auto It = MapNames::TypeNamesMap.find(R.In);
  if (It != MapNames::TypeNamesMap.end()) {
    if (It->second->Priority > R.Priority) {
      It->second->NewName = R.Out;
      It->second->Priority = R.Priority;
      It->second->RequestFeature =
          clang::c2s::HelperFeatureEnum::no_feature_helper;
      It->second->Includes.insert(It->second->Includes.end(),
                                  R.Includes.begin(), R.Includes.end());
    }
  } else {
    clang::c2s::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
      return new clang::c2s::UserDefinedTypeRule(R.In);
    });
    auto RulePtr = std::make_shared<TypeNameRule>(
        R.Out, clang::c2s::HelperFeatureEnum::no_feature_helper, R.Priority);
    RulePtr->Includes.insert(RulePtr->Includes.end(), R.Includes.begin(),
                             R.Includes.end());
    MapNames::TypeNamesMap.emplace(R.In, RulePtr);
  }
}

void importRules(llvm::cl::list<std::string> &RuleFiles) {
  for (auto &RuleFile : RuleFiles) {
    makeCanonical(RuleFile);
    // open the yaml file
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(RuleFile);
    if (!Buffer) {
      llvm::errs() << "error: failed to read " << RuleFile << ": "
                   << Buffer.getError().message() << "\n";
      clang::c2s::ShowStatus(MigrationErrorInvalidRuleFilePath);
      c2sExit(MigrationErrorInvalidRuleFilePath);
    }

    // load rules
    std::vector<std::shared_ptr<MetaRuleObject>> CurrentRules;
    llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
    YAMLIn >> CurrentRules;
    // store the rules, also prevent MetaRuleObjects from being destructed
    MetaRules.insert(MetaRules.end(), CurrentRules.begin(), CurrentRules.end());

    if (YAMLIn.error()) {
      // yaml parsing fail
      clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
      c2sExit(MigrationErrorCannotParseRuleFile);
    }

    MetaRuleObject::setRuleFiles(RuleFile);

    //Register Rules
    for (std::shared_ptr<MetaRuleObject> &r : CurrentRules) {
      switch (r->Kind) {
      case (RuleKind::Macro):
        registerMacroRule(*r);
        break;
      case (RuleKind::API):
        registerAPIRule(*r);
        break;
      case (RuleKind::Header):
        registerHeaderRule(*r);
        break;
      case (RuleKind::TypeRule):
        registerTypeRule(*r);
        break;
      default:
        break;
      }
    }
  }
}

// /RuleOutputString is the string specified in rule's "Out" session
void OutputBuilder::parse(std::string &RuleOutputString) {
  size_t i, StrStartIdx = 0;
  for (i = 0; i < RuleOutputString.length(); i++) {
    switch (RuleOutputString[i]) {
    case '\\': {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      // skip "/" and set the begin of the next string
      i++;
      StrStartIdx = i;
      break;
    }
    case '$':
    {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      SubBuilders.push_back(consumeKeyword(RuleOutputString, i));
      StrStartIdx = i;
    }
    break;
    default:
      break;
    }
  }
  if (i > StrStartIdx) {
    auto StringBuilder = std::make_shared<OutputBuilder>();
    StringBuilder->Kind = Kind::String;
    StringBuilder->Str = RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
    SubBuilders.push_back(StringBuilder);
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::ignoreWhitespaces(std::string &OutStr, size_t &Idx) {
  for (; Idx < OutStr.size(); Idx++) {
    if (OutStr[Idx] != ' ' && OutStr[Idx] != '\r' && OutStr[Idx] != '\n' &&
      OutStr[Idx] != '\t') {
      return;
    }
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeRParen(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);
  if (Idx >= OutStr.size()) {
    llvm::errs() << "rule parse error: in rule " << RuleName
                 << ", expect an ')' at end of 'Out' option value.\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }

  if (OutStr[Idx] != ')') {
    llvm::errs() << "rule parse error : in rule " << RuleName
                 << ", expect an ')' in 'Out' option value around: "
                 << OutStr.substr(Idx, 10) << "\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  } else {
    Idx++;
  }
}



// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeLParen(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);
  if (Idx >= OutStr.size()) {
    llvm::errs() << "rule parse error: in rule " << RuleName
      << ", expect an '(' at end of 'Out' option value.\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }

  if (OutStr[Idx] != '(') {
    llvm::errs() << "rule parse error : in rule " << RuleName
      << ", expect an '(' in 'Out' option value around: "
      << OutStr.substr(Idx, 10) << "\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }
  else {
    Idx++;
  }
}

// /OutStr is the string specified in rule's "Out" session
int OutputBuilder::consumeArgIndex(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);

  if (Idx >= OutStr.size()) {
    llvm::errs() << "rule parse error: in rule " << RuleName
                 << ", expect \'$\' followed by a positive integer at end of "
                    "'Out' option value.\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }

  if (OutStr[Idx] != '$') {
    llvm::errs() << "rule parse error: in rule " << RuleName
                 << ", expect \'$\' followed by a positive integer in 'Out' "
                    "option value around: "
                 << OutStr.substr(Idx, 10) << "\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }
  // consume $
  Idx++;
  auto DollarSignIdx = Idx;
  ignoreWhitespaces(OutStr, Idx);
  int ArgIndex = 0;

  if (Idx >= OutStr.size()) {
    llvm::errs() << "rule parse error: in rule " << RuleName
                 << ", expect a positive integer at end of "
                    "'Out' option value.\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }
  unsigned i = Idx;
  for (; i < OutStr.size(); i++) {
    if (!std::isdigit(OutStr[i])) {
      if (i == Idx) {
        // report unknown KW
        llvm::errs() << "rule parse error: in rule " << RuleName
                     << ", unknown keyword in 'Out' option value around: "
                     << OutStr.substr(i, 10) << "\n";
        clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
        c2sExit(MigrationErrorCannotParseRuleFile);
      } else {
        break;
      }
    }
  }

  // process arg number
  std::string ArgNumStr = OutStr.substr(Idx, i - Idx);
  Idx = i;
  ArgIndex = std::stoi(ArgNumStr);


  if (ArgIndex <= 0) {
    // report invalid ArgIndex
    llvm::errs() << "rule parse error: in rule " << RuleName
                 << ", expect a positive integer in 'Out' option value around: "
                 << OutStr.substr(DollarSignIdx, 10) << "\n";
    clang::c2s::ShowStatus(MigrationErrorCannotParseRuleFile);
    c2sExit(MigrationErrorCannotParseRuleFile);
  }
  // Adjust the index because the arg index in rules starts from $1,
  // and the arg index starts from 0 in CallExpr.
  return ArgIndex - 1;
}

// /OutStr is the string specified in rule's "Out" session
std::shared_ptr<OutputBuilder>
OutputBuilder::consumeKeyword(std::string &OutStr, size_t &Idx) {
  auto ResultBuilder = std::make_shared<OutputBuilder>();
  if (OutStr.substr(Idx, 6) == "$queue") {
    Idx += 6;
    ResultBuilder->Kind = Kind::Queue;
  } else if (OutStr.substr(Idx, 8) == "$context") {
    Idx += 8;
    ResultBuilder->Kind = Kind::Context;
  } else if (OutStr.substr(Idx, 7) == "$device") {
    Idx += 7;
    ResultBuilder->Kind = Kind::Device;
  } else if (OutStr.substr(Idx, 13) == "$type_name_of") {
    Idx += 13;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::TypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 8) == "$addr_of") {
    Idx += 8;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::AddrOf;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 11) == "$deref_type") {
    Idx += 11;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::DerefedTypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 6) == "$deref") {
    Idx += 6;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::Deref;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else {
    ResultBuilder->Kind = Kind::Arg;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
  }
  return ResultBuilder;
}

using namespace clang::ast_matchers;

void clang::c2s::UserDefinedAPIRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(hasName(APIName)))).bind("call"),
                this);
}

void clang::c2s::UserDefinedAPIRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE =
    getAssistNodeAsType<CallExpr>(
      Result, "call")) {
    auto &SM = C2SGlobalInfo::getSourceManager();
    c2s::ExprAnalysis EA;
    EA.analyze(CE);
    auto Range = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    auto Len = Lexer::MeasureTokenLength(
      Range.getEnd(), SM, C2SGlobalInfo::getContext().getLangOpts());
    Len += SM.getDecomposedLoc(Range.getEnd()).second -
      SM.getDecomposedLoc(Range.getBegin()).second;
    auto ReplStr = EA.getReplacedString();
    emplaceTransformation(
      new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));
  }
}

void clang::c2s::UserDefinedTypeRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasName(TypeName))))))
          .bind("typeLoc"),
      this);
}

void clang::c2s::UserDefinedTypeRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "typeLoc")) {
    auto TypeStr =
      C2SGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());

    auto It = MapNames::TypeNamesMap.find(TypeStr);
    if (It == MapNames::TypeNamesMap.end())
      return;

    auto ReplStr = It->second->NewName;

    auto &SM = C2SGlobalInfo::getSourceManager();
    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto Len = Lexer::MeasureTokenLength(
        Range.getEnd(), SM, C2SGlobalInfo::getContext().getLangOpts());
    Len += SM.getDecomposedLoc(Range.getEnd()).second -
           SM.getDecomposedLoc(Range.getBegin()).second;
    emplaceTransformation(
        new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));
    for (auto ItHeader = It->second->Includes.begin();
      ItHeader != It->second->Includes.end(); ItHeader++) {
      C2SGlobalInfo::getInstance().insertHeader(Range.getBegin(), *ItHeader);
    }
  }
}